import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from avalanche.training.strategies.base_strategy import BaseStrategy
from avalanche.training.plugins import StrategyPlugin
from avalanche.training.plugins.evaluation import default_logger
from typing import Optional, Sequence, Union, List

from methods.hat_model import HATModel


class HATStrategy(BaseStrategy):
    """ Average Gradient Episodic Memory (A-GEM) strategy.

    See AGEM plugin for details.
    This strategy does not use task identities.
    """

    def __init__(self, model, optimizer,
                 lamb=0.75, smax=400, clipgrad=10000,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins=None, evaluator=default_logger, eval_every=-1):
        assert isinstance(model, HATModel)
        super().__init__(
            model, optimizer, None,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

        self.lamb = lamb
        self.smax = smax
        self.s = 0
        self.mask_pre = None
        self.mask_back = None
        self.clipgrad = clipgrad

        self.r = 0  # dataset size
        self.i = 0  # number of images seen in this task

    def criterion(self, outputs, targets, masks):
        reg = 0
        count = 0
        if self.mask_pre is not None:
            for m, mp in zip(masks, self.mask_pre):
                aux = 1-mp
                reg += (m*aux).sum()
                count += aux.sum()
        else:
            for m in masks:
                reg += m.sum()
                count += np.prod(m.size()).item()
        reg /= count
        return F.cross_entropy(outputs, targets) + self.lamb * reg

    def train_exp(self, experience, eval_streams=None, **kwargs):
        """ Training loop over a single Experience object.

        :param experience: CL experience information.
        :param eval_streams: list of streams for evaluation.
            If None: use the training experience for evaluation.
            Use [] if you do not want to evaluate during training.
        :param kwargs: custom arguments.
        """
        self.experience = experience
        self.model.train()

        if eval_streams is None:
            eval_streams = [experience]
        for i, exp in enumerate(eval_streams):
            if not isinstance(exp, Sequence):
                eval_streams[i] = [exp]

        # Data Adaptation (e.g. add new samples/data augmentation)
        self._before_train_dataset_adaptation(**kwargs)
        self.train_dataset_adaptation(**kwargs)
        self._after_train_dataset_adaptation(**kwargs)
        self.make_train_dataloader(**kwargs)

        # Model Adaptation (e.g. freeze/add new units)
        self.model = self.model_adaptation()
        self.make_optimizer()

        self._before_training_exp(**kwargs)

        do_final = True
        if self.eval_every > 0 and \
                (self.train_epochs - 1) % self.eval_every == 0:
            do_final = False

        for _ in range(self.train_epochs):
            self._before_training_epoch(**kwargs)

            if self._stop_training:  # Early stopping
                self._stop_training = False
                break

            self.training_epoch(**kwargs)
            self._after_training_epoch(**kwargs)
            self._periodic_eval(eval_streams, do_final=False)

        task_id = self.experience.current_experience
        task = torch.LongTensor([task_id]).cuda()
        mask = self.model.mask(task, s=self.smax)
        for i in range(len(mask)):
            mask[i] = mask[i].data.clone()
        if task_id == 0:
            self.mask_pre = mask
        else:
            for i in range(len(self.mask_pre)):
                self.mask_pre[i] = torch.max(self.mask_pre[i], mask[i])

        # Weights mask
        self.mask_back = {}
        for n, _ in self.model.named_parameters():
            vals = self.model.get_view_for(n, self.mask_pre)
            if vals is not None:
                self.mask_back[n] = 1-vals

        # Final evaluation
        self._periodic_eval(eval_streams, do_final=do_final)
        self._after_training_exp(**kwargs)

    def make_optimizer(self):
        super().make_optimizer()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.optimizer.defaults['lr'])

    def training_epoch(self, thres_cosh=50, thres_emb=6, **kwargs):
        r = len(self.adapted_dataset)
        i = 0

        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.loss = 0

            s = (self.smax-1/self.smax)*i/r+1/self.smax

            # Forward
            self._before_forward(**kwargs)
            task_id = self.experience.current_experience
            task_id = torch.LongTensor([task_id]).to(self.device)
            self.mb_output, masks = self.model(self.mb_x, task_id, self.s)
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self.criterion(self.mb_output, self.mb_y, masks)

            self._before_backward(**kwargs)
            self.optimizer.zero_grad()
            self.loss.backward()
            self._after_backward(**kwargs)

            # Restrict layer gradients in backprop
            if task_id.item() > 0:
                for n, p in self.model.named_parameters():
                    if n in self.mask_back:
                        p.grad.data *= self.mask_back[n]

            # Compensate embedding gradients
            for n, p in self.model.named_parameters():
                if n.startswith('e'):
                    num = torch.cosh(torch.clamp(s*p.data, -thres_cosh, thres_cosh))+1
                    den = torch.cosh(p.data)+1
                    p.grad.data *= self.smax/s*num/den

            # Apply step
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clipgrad)

            # Optimization step
            self._before_update(**kwargs)
            self.optimizer.step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

            # Constrain embeddings
            for n, p in self.model.named_parameters():
                if n.startswith('e'):
                    p.data = torch.clamp(p.data, -thres_emb, thres_emb)

            self.i += len(self.mb_x)

    def eval_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)

            self._before_eval_forward(**kwargs)
            task_id = self.experience.current_experience
            task_id = torch.LongTensor([task_id]).to(self.device)
            self.mb_output, _ = self.model(self.mb_x, task_id, s=self.smax)
            self._after_eval_forward(**kwargs)

            self._after_eval_iteration(**kwargs)

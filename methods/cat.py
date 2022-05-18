from itertools import zip_longest
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from avalanche.training.strategies.base_strategy import BaseStrategy
from avalanche.training.plugins import StrategyPlugin
from avalanche.training.plugins.evaluation import default_logger
from typing import Optional, Sequence, Union, List

from methods.hat_model import HATModel


class CATStrategy(BaseStrategy):
    def __init__(self, num_classes, model, optimizer,
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

        self.check_federated = CheckFederated()

        self.acc_transfer = np.zeros((num_classes, num_classes), dtype=np.float32)
        self.similarity_transfer = np.zeros((num_classes, num_classes), dtype=np.float32)

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

        return self.ce(outputs, targets)+self.lamb*reg

    def train(self, experiences, eval_streams=None, **kwargs):
        """ Training loop. if experiences is a single element trains on it.
        If it is a sequence, trains the model on each experience in order.
        This is different from joint training on the entire stream.
        It returns a dictionary with last recorded value for each metric.

        :param experiences: single Experience or sequence.
        :param eval_streams: list of streams for evaluation.
            If None: use training experiences for evaluation.
            Use [] if you do not want to evaluate during training.

        :return: dictionary containing last recorded value for
            each metric name.
        """
        self.is_training = True
        self._stop_training = False

        self.model.train()
        self.model.to(self.device)

        # Normalize training and eval data.
        if not isinstance(experiences, Sequence):
            experiences = [experiences]
        if eval_streams is None:
            eval_streams = [experiences]

        self._before_training(**kwargs)

        self._periodic_eval(eval_streams, do_final=False, do_initial=True)

        task = self.experience.current_experience
        similarities = []
        history_mask_back = []
        history_mask_pre = []

        for self.experience in experiences:
            candidate_phases = ['mcl']

            for candidate_phase in candidate_phases:
                if candidate_phase == 'mcl':
                    similarity = self.auto_similarity(task)
                    self.check_federated.set_similarities(similarities)
                similarities.append(similarity)
                self.check_federated.set_similarities(similarities)

                self.train_exp(self.experience, eval_streams, candidate_phase,
                               similarity=similarity, history_mask_back=history_mask_back,
                               history_mask_pre=history_mask_pre, check_federated=self.check_federated, **kwargs)

                if candidate_phase == 'mcl':
                    history_mask_back.append(dict((k, v.data.clone()) for k, v in self.mask_back.items()))
                    history_mask_pre.append([m.data.clone() for m in self.mask_pre])
        self._after_training(**kwargs)

        res = self.evaluator.get_last_metrics()
        return res

    def auto_similarity(self, t, experiences):
        """use this to detect similarity by transfer and reference network"""
        if t > 0:
            for pre_task in range(t+1):
                pre_task_torch = torch.autograd.Variable(torch.LongTensor([pre_task]).cuda(), volatile=False)

                gfc1, gfc2 = self.model.mask(pre_task_torch)

                gfc1 = gfc1.detach()
                gfc2 = gfc2.detach()
                pre_mask = [gfc1, gfc2]

                if pre_task == t:  # the last one
                    self.train_exp(t, self.experience, phase='reference', pre_mask=pre_mask, pre_task=pre_task)  # implemented as random mask
                elif pre_task != t:
                    self.train_exp(t, self.experience, phase='transfer', pre_mask=pre_mask, pre_task=pre_task)

                if pre_task == t:  # the last one
                    res = self.eval([self.experience], phase='reference',  # TODO change to val exprience
                                    pre_mask=pre_mask, pre_task=pre_task)
                elif pre_task != t:
                    res = self.eval([self.experience], phase='transfer',
                                    pre_mask=pre_mask, pre_task=pre_task)
                print(res)
                test_acc = res['acc']  # TODO check if works
                exit()

                self.acc_transfer[t, pre_task] = test_acc

        similarity = [0]
        if t > 0:
            acc_list = self.acc_transfer[t][:t]  # t from 0
            print('acc_list: ', acc_list)

            similarity = [0 if (acc_list[acc_id] <= self.acc_transfer[t][t]) else 1 for acc_id in range(len(acc_list))]  # remove all acc < 0.5

            for source_task in range(len(similarity)):
                self.similarity_transfer[t, source_task] = similarity[source_task]

        return similarity

    def train_exp(self, experience, eval_streams=None, phase=None,
                  pre_mask=None, pre_task=None,
                  similarity=None, history_mask_back=None,
                  history_mask_pre=None, check_federated=None, **kwargs):
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
        self.make_optimizer(phase)

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

            self.training_epoch(phase=phase, pre_mask=pre_mask, pre_task=pre_task,
                                similarity=similarity, history_mask_back=history_mask_back,
                                history_mask_pre=history_mask_pre, check_federated=check_federated,
                                **kwargs)
            self._after_training_epoch(**kwargs)
            self._periodic_eval(eval_streams, do_final=False)

        # Final evaluation
        self._periodic_eval(eval_streams, do_final=do_final)
        self._after_training_exp(**kwargs)

        t = self.experience.current_experience
        if phase == 'mcl':
            # Activations mask
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False)
            mask = self.model.mask(task, s=self.smax)

            for i in range(len(mask)):
                mask[i] = torch.autograd.Variable(mask[i].data.clone(), requires_grad=False)

            if t == 0:
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

    def make_optimizer(self, phase=None):
        super().make_optimizer()
        lr = self.optimizer.defaults['lr']
        if phase == 'mcl':
            params = list(self.model.kt.parameters())+list(self.model.mcl.parameters())
            self.optimizer = torch.optim.SGD(params, lr=lr)
        elif phase == 'transfer':
            self.optimizer = torch.optim.SGD(list(self.model.transfer.parameters()), lr=lr)
        elif phase == 'reference':
            self.optimizer = torch.optim.SGD(list(self.model.transfer.parameters()), lr=lr)
        else:
            raise ValueError(f'Invalid pahse: {phase}')

    def training_epoch(self, phase=None, similarity=None, history_mask_pre=None,
                       check_federated=None, pre_mask=None, pre_task=None,
                       thres_emb=6, thres_cosh=50,
                       ** kwargs):
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

            self.mb_output = self.forward()

            t = self.experience.current_experience
            task = torch.LongTensor([t]).to(self.device)
            if phase == 'mcl':
                outputs, masks, outputs_attn = self.model.forward(task, self.mb_x, s=s, phase=phase,
                                                                  similarity=similarity, history_mask_pre=history_mask_pre,
                                                                  check_federated=check_federated)
                output = outputs[t]

                if outputs_attn is None:
                    loss = self.criterion(output, self.mb_y, masks)
                else:
                    output_attn = outputs_attn[t]
                    loss = self.joint_criterion(output, self.mb_y, masks, output_attn)

            elif phase == 'transfer' or phase == 'reference':

                outputs = self.model.forward(task, self.mb_x, s=s, phase=phase,
                                             pre_mask=pre_mask, pre_task=pre_task)
                output = outputs[t]
                loss = self.transfer_criterion(output, self.mb_y)

            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += loss

            self._before_backward(**kwargs)
            self.optimizer.zero_grad()
            self.loss.backward()
            self._after_backward(**kwargs)

            if phase == 'mcl':
                # Restrict layer gradients in backprop
                if t > 0:
                    for n, p in self.model.named_parameters():
                        if n in self.mask_back and p.grad is not None:
                            Tsim_mask = self.model.Tsim_mask(task, history_mask_pre=history_mask_pre, similarity=similarity)
                            Tsim_vals = self.model.get_view_for(n, Tsim_mask).clone()
                            p.grad.data *= torch.max(self.mask_back[n], Tsim_vals)

                # Compensate embedding gradients
                for n, p in self.model.named_parameters():
                    if n.startswith('mcl.e') and p.grad is not None:
                        num = torch.cosh(torch.clamp(s*p.data, -thres_cosh, thres_cosh))+1
                        den = torch.cosh(p.data)+1
                        p.grad.data *= self.smax/s*num/den

            elif phase == 'reference':
                # Compensate embedding gradients
                for n, p in self.model.named_parameters():
                    if n.startswith('transfer.e') and p.grad is not None:
                        num = torch.cosh(torch.clamp(s*p.data, -thres_cosh, thres_cosh))+1
                        den = torch.cosh(p.data)+1
                        p.grad.data *= self.smax/s*num/den

            # Optimization step
            self._before_update(**kwargs)
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

            if phase == 'mcl':
                # Constrain embeddings
                for n, p in self.model.named_parameters():
                    if n.startswith('mcl.e'):
                        p.data = torch.clamp(p.data, -thres_emb, thres_emb)

            elif phase == 'reference':
                # Constrain embeddings
                for n, p in self.model.named_parameters():
                    if n.startswith('transfer.e'):
                        p.data = torch.clamp(p.data, -thres_emb, thres_emb)

    @torch.no_grad()
    def eval(self, exp_list, phase, pre_mask, pre_task, **kwargs):
        """
        Evaluate the current model on a series of experiences and
        returns the last recorded value for each metric.

        :param exp_list: CL experience information.
        :param kwargs: custom arguments.

        :return: dictionary containing last recorded value for
            each metric name
        """
        self.is_training = False
        self.model.eval()

        if not isinstance(exp_list, Sequence):
            exp_list = [exp_list]
        self.current_eval_stream = exp_list

        self._before_eval(**kwargs)
        for self.experience in exp_list:
            # Data Adaptation
            self._before_eval_dataset_adaptation(**kwargs)
            self.eval_dataset_adaptation(**kwargs)
            self._after_eval_dataset_adaptation(**kwargs)
            self.make_eval_dataloader(**kwargs)

            # Model Adaptation (e.g. freeze/add new units)
            self.model = self.model_adaptation()

            self._before_eval_exp(**kwargs)
            self.eval_epoch(phase, pre_mask, pre_task, **kwargs)
            self._after_eval_exp(**kwargs)

        self._after_eval(**kwargs)

        res = self.evaluator.get_last_metrics()

        return res

    def eval_epoch(self, phase, pre_mask, pre_task,
                   similarity=None, history_mask_pre=None, check_federated=None,
                   **kwargs):
        """Evaluation loop over the current `self.dataloader`."""
        for self.mbatch in self.dataloader:
            self._unpack_minibatch()
            self._before_eval_iteration(**kwargs)

            self._before_eval_forward(**kwargs)
            t = self.experience.current_experience
            task = torch.LongTensor([t]).to(self.device)
            if phase == 'mcl':
                outputs, masks, outputs_attn = self.model.forward(task, self.mb_x, s=self.smax, phase=phase,
                                                                  similarity=similarity,
                                                                  history_mask_pre=history_mask_pre,
                                                                  check_federated=check_federated)
                self.mb_output = outputs[t]

            elif phase == 'transfer' or phase == 'reference':
                outputs = self.model.forward(task, self.mb_x, s=self.smax, phase=phase,
                                             pre_mask=pre_mask, pre_task=pre_task)
                output = outputs[t]

            self._after_eval_forward(**kwargs)
            self.loss = self.criterion()

            self._after_eval_iteration(**kwargs)


class CheckFederated():
    def __init__(self):
        pass

    def set_similarities(self, similarities):
        self.similarities = similarities

    def fix_length(self):
        return len(self.similarities)

    def get_similarities(self):
        return self.similarities

    def check_t(self, t):
        if t < len([sum(x) for x in zip_longest(*self.similarities, fillvalue=0)]) and [sum(x) for x in zip_longest(*self.similarities, fillvalue=0)][t] > 0:
            return True

        elif np.count_nonzero(self.similarities[t]) > 0:
            return True

        elif t < len(self.similarities[-1]) and self.similarities[-1][t] == 1:
            return True

        return False

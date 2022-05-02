import copy
import torch
import torch.nn.functional as F

from avalanche.models import avalanche_forward
from avalanche.training.plugins.evaluation import default_logger
from avalanche.training.strategies.base_strategy import BaseStrategy
from torch.utils.data.dataloader import DataLoader
from .rehersal_buffer import RehersalBuffer
from avalanche.benchmarks.utils.data_loader import _default_collate_mbatches_fn
from avalanche.training.plugins import StrategyPlugin


class MirPlugin(StrategyPlugin):

    def __init__(self, patterns_per_exp: int, sample_size: int):
        super().__init__()

        self.patterns_per_experience = int(patterns_per_exp)
        self.sample_size = int(sample_size)

        self.buffers = []  # one AvalancheDataset for each experience.
        self.buffer_dataloader = None
        self.buffer_dliter = None

        self.reference_gradients = None
        self.memory_x, self.memory_y = None, None

    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute reference gradient on memory sample.
        """
        if len(self.buffers) > 0:
            strategy.model.train()
            strategy.optimizer.zero_grad()

            y_hat = avalanche_forward(strategy.model, strategy.mb_x, strategy.mb_task_id)
            loss = F.cross_entropy(y_hat, strategy.mb_y, reduction='mean')

            strategy.optimizer.zero_grad()
            loss.backward()

            mb = self.sample_from_memory()
            bx, by, bt = mb[0], mb[1], mb[-1]
            bx = bx.to(strategy.device)
            by = by.to(strategy.device)
            bt = bt.to(strategy.device)

            grad_dims = []
            for param in strategy.model.parameters():
                grad_dims.append(param.data.numel())
            grad_vector = self.get_grad_vector(strategy.device, strategy.model.parameters, grad_dims)
            model_temp = self.get_future_step_parameters(strategy.model, grad_vector, grad_dims, lr=strategy.optimizer.defaults['lr'])

            with torch.no_grad():
                y_hat_pre = avalanche_forward(strategy.model, bx, bt)
                pre_loss = F.cross_entropy(y_hat_pre, by, reduction='none')

                y_hat_post = avalanche_forward(model_temp, bx, bt)
                post_loss = F.cross_entropy(y_hat_post, by, reduction="none")

                scores = post_loss - pre_loss
                big_ind = scores.sort(descending=True)[1][:strategy.train_mb_size]

            mem_x, mem_y, b_task_ids = bx[big_ind], by[big_ind], bt[big_ind]

            strategy.mbatch[0] = torch.cat([strategy.mbatch[0], mem_x], dim=0)
            strategy.mbatch[1] = torch.cat([strategy.mbatch[1], mem_y], dim=0)
            strategy.mbatch[2] = torch.cat([strategy.mbatch[2], b_task_ids], dim=0)

    def get_grad_vector(self, device, pp, grad_dims):
        """
        gather the gradients in one vector
        """
        grads = torch.Tensor(sum(grad_dims))
        if device.type == 'cuda':
            grads = grads.to(device)

        grads.fill_(0.0)
        cnt = 0
        for param in pp():
            if param.grad is not None:
                beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
                en = sum(grad_dims[:cnt + 1])
                grads[beg: en].copy_(param.grad.data.view(-1))
            cnt += 1
        return grads

    def get_future_step_parameters(self, this_net, grad_vector, grad_dims, lr=1):
        new_net = copy.deepcopy(this_net)
        self.overwrite_grad(new_net.parameters, grad_vector, grad_dims)
        with torch.no_grad():
            for param in new_net.parameters():
                if param.grad is not None:
                    param.data = param.data - lr*param.grad.data
        return new_net

    def overwrite_grad(self, parameters, new_grad, grad_dims):
        cnt = 0
        for param in parameters():
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1

    def after_training_exp(self, strategy, **kwargs):
        """ Update replay memory with patterns from current experience. """
        self.update_memory(strategy.experience.dataset)

    def sample_from_memory(self):
        """
        Sample a minibatch from memory.
        Return a tuple of patterns (tensor), targets (tensor).
        """
        return next(self.buffers_dataloders)

    @torch.no_grad()
    def update_memory(self, dataset):
        """
        Update replay memory with patterns from current experience.
        """
        print('\n\nupdate memory called\n\n')
        if len(dataset) > self.patterns_per_experience:
            indices = torch.randperm(len(dataset))[:self.patterns_per_experience]
            dataset = torch.utils.data.Subset(dataset, indices)
        self.buffers.append(dataset)

        buffer_dataset = RehersalBuffer(self.buffers)

        dataloder = DataLoader(buffer_dataset,
                               batch_size=self.sample_size,
                               num_workers=0,
                               drop_last=False,
                               shuffle=False,
                               collate_fn=_default_collate_mbatches_fn)
        self.buffers_dataloders = iter(dataloder)


class Mir(BaseStrategy):
    """ Average Gradient Episodic Memory (A-GEM) strategy.

    See AGEM plugin for details.
    This strategy does not use task identities.
    """

    def __init__(self, model, optimizer, criterion,
                 patterns_per_exp: int, sample_size: int,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins=None, evaluator=default_logger, eval_every=-1):
        mir = MirPlugin(patterns_per_exp, sample_size)
        if plugins is None:
            plugins = [mir]
        else:
            plugins.append(mir)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size // 2, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

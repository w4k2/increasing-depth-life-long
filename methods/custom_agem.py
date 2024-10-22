import torch
from avalanche.benchmarks.utils.data_loader import _default_collate_mbatches_fn
from avalanche.models import avalanche_forward
from avalanche.training.plugins import StrategyPlugin
from avalanche.training.plugins.evaluation import default_logger
from avalanche.training.strategies.base_strategy import BaseStrategy
from torch.utils.data.dataloader import DataLoader

from .rehersal_buffer import RehersalBuffer


class AGEMPluginModified(StrategyPlugin):
    """ Average Gradient Episodic Memory Plugin.

    AGEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. If the dot product
    between the current gradient and the (average) gradient of a randomly
    sampled set of memory examples is negative, the gradient is projected.
    This plugin does not use task identities.
    """

    def __init__(self, patterns_per_experience: int, sample_size: int):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param sample_size: number of patterns in memory sample when computing
            reference gradient.
        """

        super().__init__()

        self.patterns_per_experience = int(patterns_per_experience)
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
            mb = self.sample_from_memory()
            xref, yref, tid = mb[0], mb[1], mb[-1]
            xref, yref = xref.to(strategy.device), yref.to(strategy.device)

            out = avalanche_forward(strategy.model, xref, tid)
            loss = strategy._criterion(out, yref)
            loss.backward()
            # gradient can be None for some head on multi-headed models
            self.reference_gradients = [
                p.grad.view(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=strategy.device)
                for n, p in strategy.model.named_parameters()]
            self.reference_gradients = torch.cat(self.reference_gradients)
            strategy.optimizer.zero_grad()

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """
        if len(self.buffers) > 0:
            current_gradients = [
                p.grad.view(-1) if p.grad is not None
                else torch.zeros(p.numel(), device=strategy.device)
                for n, p in strategy.model.named_parameters()]
            current_gradients = torch.cat(current_gradients)

            assert current_gradients.shape == self.reference_gradients.shape, \
                "Different model parameters in AGEM projection"

            dotg = torch.dot(current_gradients, self.reference_gradients)
            if dotg < 0:
                alpha2 = dotg / torch.dot(self.reference_gradients,
                                          self.reference_gradients)
                grad_proj = current_gradients - self.reference_gradients * alpha2

                count = 0
                for _, p in strategy.model.named_parameters():
                    n_param = p.numel()
                    if p.grad is not None:
                        p.grad.copy_(grad_proj[count:count+n_param].view_as(p))
                    count += n_param

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
        if len(dataset) > self.patterns_per_experience:
            indices = torch.randperm(len(dataset))[:self.patterns_per_experience]
            dataset = torch.utils.data.Subset(dataset, indices)
        self.buffers.append(dataset)

        buffer_dataset = RehersalBuffer(self.buffers)

        dataloder = DataLoader(buffer_dataset,
                               batch_size=self.sample_size // len(self.buffers),
                               num_workers=4,
                               drop_last=False,
                               shuffle=False,
                               collate_fn=_default_collate_mbatches_fn)
        self.buffers_dataloders = iter(dataloder)


class AGEMModified(BaseStrategy):
    """ Average Gradient Episodic Memory (A-GEM) strategy.

    See AGEM plugin for details.
    This strategy does not use task identities.
    """

    def __init__(self, model, optimizer, criterion,
                 patterns_per_exp: int, sample_size: int = 64,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins=None, evaluator=default_logger, eval_every=-1):
        agem = AGEMPluginModified(patterns_per_exp, sample_size)
        if plugins is None:
            plugins = [agem]
        else:
            plugins.append(agem)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

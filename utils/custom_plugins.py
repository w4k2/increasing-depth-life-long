
from avalanche.models import avalanche_forward
from torch.autograd import grad
from torch.utils.data import random_split
import avalanche
import torch
import torch.optim as optim
import copy

from avalanche.benchmarks.utils.data_loader import \
    GroupBalancedInfiniteDataLoader, _default_collate_mbatches_fn
from avalanche.training.plugins.evaluation import default_logger
from avalanche.training.strategies.base_strategy import BaseStrategy
from avalanche.training.plugins import AGEMPlugin
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Lambda
from avalanche.training.plugins import StrategyPlugin


class DebugingPlugin(StrategyPlugin):
    def __init__(self):
        super().__init__()

    def after_training_iteration(self, strategy, **kwargs):
        strategy.stop_training()


class ConvertedLabelsPlugin(StrategyPlugin):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.task_label_mappings = dict()

    def before_training_exp(self, strategy, **kwargs):
        task_id = strategy.experience.current_experience
        self.adapt_dataloder(strategy, task_id)

    def before_eval_exp(self, strategy, **kwargs):
        task_id = strategy.experience.current_experience
        self.adapt_dataloder(strategy, task_id)

    def adapt_dataloder(self, strategy, task_id):
        if type(strategy.dataloader) == avalanche.benchmarks.utils.data_loader.TaskBalancedDataLoader:
            label_mapping = self.get_label_mapping(strategy.adapted_dataset, task_id)
            for i, dataset in enumerate(strategy.dataloader._dl.datasets):
                dataset.target_transform = Lambda(lambda l: label_mapping[l])
                strategy.dataloader._dl.datasets[i] = dataset
        elif type(strategy.dataloader) == torch.utils.data.DataLoader:
            label_mapping = self.get_label_mapping(strategy.adapted_dataset, task_id)
            strategy.dataloader.dataset.target_transform = Lambda(lambda l: label_mapping[l])
        else:
            print('else in adapt dataloader called')
            exit()

    def get_label_mapping(self, dataset, task_id):
        label_mapping = dict()
        if task_id in self.task_label_mappings:
            label_mapping = self.task_label_mappings[task_id]
        else:
            targets = [t.item() if type(t) == torch.Tensor else t for t in dataset.targets]
            task_classes = sorted(set(targets))
            label_mapping = {class_idx: i for i, class_idx in enumerate(task_classes)}
            self.task_label_mappings[task_id] = label_mapping
        return label_mapping

    def after_training_exp(self, strategy, **kwargs):
        # dataset adaptation for ewc
        task_id = strategy.experience.current_experience
        label_mapping = self.get_label_mapping(strategy.experience.dataset, task_id)
        strategy.experience.dataset.target_transform = Lambda(lambda l: label_mapping[l])


class BaselinePlugin(ConvertedLabelsPlugin):
    """Creates new instance of predefeined model for each task
    Can be used as upper bound for performance
    """

    def __init__(self, model_creation_fn, classes_per_task, device) -> None:
        super().__init__()
        self.model_creation_fn = model_creation_fn
        self.device = device
        self.classes_per_task = classes_per_task
        self.task_models = []

    def before_training_exp(self, strategy, **kwargs):
        super().before_training_exp(strategy)

        task_id = strategy.experience.current_experience
        strategy.model = self.model_creation_fn(num_classes=self.classes_per_task[task_id])
        strategy.model.to(self.device)
        strategy.make_optimizer()

    def after_training_exp(self, strategy, **kwargs):
        task_model = copy.deepcopy(strategy.model)
        for param in task_model.parameters():
            param.requires_grad = False
            param.grad = None
        self.task_models.append(task_model)

    def before_eval_exp(self, strategy, **kwargs):
        task_id = strategy.experience.current_experience
        self.adapt_dataloder(strategy, task_id)

        if task_id < len(self.task_models):
            task_model = self.task_models[task_id]
            task_model.to(self.device)
            task_model = task_model.eval()
            strategy.model = task_model


class StochasticDepthPlugin(ConvertedLabelsPlugin):
    def __init__(self, entropy_threshold, device):
        super().__init__()
        self.device = device
        self.entropy_threshold = entropy_threshold

    def before_training_exp(self, strategy, **kwargs):
        task_id = strategy.experience.current_experience
        self.adapt_dataloder(strategy, task_id)

        num_classes = len(strategy.experience.classes_in_this_experience)
        strategy.model.update_structure(task_id, strategy.dataloader, num_classes, self.device, self.entropy_threshold)

        strategy.optimizer = optim.Adam([{'params': filter(lambda p: p.requires_grad, strategy.model.parameters())}], lr=0.0001, weight_decay=1e-6, amsgrad=False)
        print('training od task id = ', task_id)

    def before_eval_exp(self, strategy, **kwargs):
        task_id = strategy.experience.current_experience
        self.adapt_dataloder(strategy, task_id)
        print('plugin before eval, task idx = ', task_id)
        if task_id in strategy.model.tasks_paths:
            task_path = strategy.model.tasks_paths[task_id]
            print('selected path = ', task_path)
            strategy.model.set_path(task_path)


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
        self.buffers_dataloders = []

        self.reference_gradients = None

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
            self.reference_gradients = self.get_gradiens_vector(strategy.model)
            strategy.optimizer.zero_grad()

    def get_gradiens_vector(self, model):
        gradient_vec = []
        for p in model.parameters():
            if p.requires_grad:
                gradient_vec.append(p.grad.view(-1))
        gradient_vec = torch.cat(gradient_vec)
        return gradient_vec

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """
        if len(self.buffers) > 0:
            current_gradients = self.get_gradiens_vector(strategy.model)

            assert current_gradients.shape == self.reference_gradients.shape, \
                "Different model parameters in AGEM projection"

            dotg = torch.dot(current_gradients, self.reference_gradients)
            if dotg < 0:
                alpha2 = dotg / torch.dot(self.reference_gradients,
                                          self.reference_gradients)
                grad_proj = current_gradients - self.reference_gradients * alpha2

                count = 0
                for p in strategy.model.parameters():
                    n_param = p.numel()
                    if p.grequires_grad:
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
        minibatch = [next(d) for d in self.buffers_dataloders]
        minibatch = _default_collate_mbatches_fn(minibatch)
        return minibatch
        # return next(self.buffer_dliter)

    @torch.no_grad()
    def update_memory(self, dataset):
        """
        Update replay memory with patterns from current experience.
        """
        removed_els = len(dataset) - self.patterns_per_experience
        if removed_els > 0:
            dataset, _ = random_split(dataset,
                                      [self.patterns_per_experience,
                                       removed_els])
        self.buffers.append(dataset)

        self.buffers_dataloders = []
        for old_dataset in self.buffers:
            dataloader = DataLoader(old_dataset,
                                    batch_size=self.sample_size // len(self.buffers),
                                    num_workers=4,
                                    drop_last=True
                                    )
            import itertools
            dataloder_iterator = itertools.cycle(iter(dataloader))
            self.buffers_dataloders.append(dataloder_iterator)

        # self.buffer_dataloader = GroupBalancedInfiniteDataLoader(
        #     self.buffers,
        #     batch_size=self.sample_size // len(self.buffers),
        #     num_workers=4,
        #     pin_memory=False,
        #     persistent_workers=False)
        # self.buffer_dliter = iter(self.buffer_dataloader)


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
        agem = AGEMPlugin(patterns_per_exp, sample_size)
        if plugins is None:
            plugins = [agem]
        else:
            plugins.append(agem)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

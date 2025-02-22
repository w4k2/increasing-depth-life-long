import copy

import avalanche
import torch
import torch.optim as optim
import torch.utils.data
from avalanche.training.plugins import StrategyPlugin
from torchvision.transforms import Lambda


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
    def __init__(self, entropy_threshold, device, lr=0.0001, weight_decay=1e-6):
        super().__init__()
        self.device = device
        self.entropy_threshold = entropy_threshold
        self.lr = lr
        self.weight_decay = weight_decay

    def before_training_exp(self, strategy, **kwargs):
        task_id = strategy.experience.current_experience
        self.adapt_dataloder(strategy, task_id)

        num_classes = len(strategy.experience.classes_in_this_experience)
        strategy.model.update_structure(task_id, strategy.dataloader, num_classes, self.device, self.entropy_threshold)

        strategy.optimizer = optim.Adam(
            params=[{'params': filter(lambda p: p.requires_grad, strategy.model.parameters())}],
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=False)
        print('training od task id = ', task_id)

    def before_eval_exp(self, strategy, **kwargs):
        task_id = strategy.experience.current_experience
        self.adapt_dataloder(strategy, task_id)
        print('plugin before eval, task idx = ', task_id)
        if task_id in strategy.model.tasks_paths:
            task_path = strategy.model.tasks_paths[task_id]
            print('selected path = ', task_path)
            strategy.model.set_path(task_path)

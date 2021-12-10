import avalanche
import torch
import torch.optim as optim
import copy

from torchvision.transforms import Lambda
from avalanche.training.plugins import StrategyPlugin


class ConvertedLabelsPlugin(StrategyPlugin):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.task_label_mappings = dict()
        self.current_train_task_idx = 0
        self.current_eval_task_idx = 0

    def before_training_exp(self, strategy, **kwargs):
        self.adapt_dataloder(strategy, self.current_train_task_idx)

    def before_eval_exp(self, strategy, **kwargs):
        self.adapt_dataloder(strategy, self.current_eval_task_idx)

    def adapt_dataloder(self, strategy, task_id):
        if type(strategy.dataloader) == avalanche.benchmarks.utils.data_loader.TaskBalancedDataLoader:
            label_mapping = self.get_label_mapping(strategy.adapted_dataset, task_id)
            for i, dataset in enumerate(strategy.dataloader._dl.datasets):
                dataset.target_transform = Lambda(lambda l: label_mapping[l])
                strategy.dataloader._dl.datasets[i] = dataset
        elif type(strategy.dataloader) == torch.utils.data.DataLoader:
            label_mapping = self.get_label_mapping(strategy.adapted_dataset, task_id)
            strategy.dataloader.dataset.target_transform = Lambda(lambda l: label_mapping[l])

    def get_label_mapping(self, dataset, task_id):
        label_mapping = dict()
        if task_id in self.task_label_mappings:
            label_mapping = self.task_label_mappings[task_id]
        else:
            task_classes = sorted(set(dataset.targets))
            label_mapping = {class_idx: i for i, class_idx in enumerate(task_classes)}
            self.task_label_mappings[task_id] = label_mapping
        return label_mapping

    def after_training_exp(self, strategy, **kwargs):
        # dataset adaptation for ewc
        label_mapping = self.get_label_mapping(strategy.experience.dataset, self.current_train_task_idx)
        strategy.experience.dataset.target_transform = Lambda(lambda l: label_mapping[l])

        self.current_train_task_idx += 1

    def after_eval_exp(self, strategy, **kwargs):
        self.current_eval_task_idx += 1

    def after_eval(self, strategy, **kwargs):
        self.current_eval_task_idx = 0

    # def after_training_iteration(self, strategy, **kwargs):
    #     strategy.stop_training()


class BaselinePlugin(ConvertedLabelsPlugin):
    """Creates new instance of predefeined model for each task
    Can be used as upper bound for performance
    """

    def __init__(self, base_model, device) -> None:
        super().__init__()
        self.base_model = base_model
        self.device = device
        self.task_models = []
        self.current_train_task_idx = 0
        self.current_eval_task_idx = 0

    def before_training_exp(self, strategy, **kwargs):
        self.adapt_dataloder(strategy, self.current_train_task_idx)

        strategy.model = copy.deepcopy(self.base_model)
        strategy.model.to(self.device)
        strategy.make_optimizer()

    def after_training_exp(self, strategy, **kwargs):
        task_model = copy.deepcopy(strategy.model)
        for param in task_model.parameters():
            param.requires_grad = False
            param.grad = None
        self.task_models.append(task_model)
        self.current_train_task_idx += 1

    def before_eval_exp(self, strategy, **kwargs):
        self.adapt_dataloder(strategy, self.current_eval_task_idx)

        task_model = self.task_models[self.current_eval_task_idx]
        task_model.to(self.device)
        task_model = task_model.eval()
        strategy.model = task_model

    def after_eval_exp(self, strategy, **kwargs):
        self.current_eval_task_idx += 1

    def after_eval(self, strategy, **kwargs):
        self.current_eval_task_idx = 0


class StochasticDepthPlugin(ConvertedLabelsPlugin):
    def __init__(self, device):
        super().__init__()
        self.tasks_paths = dict()
        self.device = device
        self.current_train_task_idx = 0
        self.current_eval_task_idx = 0

    def before_training_exp(self, strategy, **kwargs):
        self.adapt_dataloder(strategy, self.current_train_task_idx)

        current_path = [0]
        if self.current_train_task_idx > 0:
            path = strategy.model.select_most_similar_task(strategy.dataloader, num_classes=10, threshold=0.6)
            print('min entropy path = ', path)
            strategy.model.add_new_node(path)
            strategy.model.to(self.device)
            current_path = strategy.model.get_current_path()

        strategy.optimizer = optim.Adam([{'params': filter(lambda p: p.requires_grad, strategy.model.parameters())}], lr=0.0001, weight_decay=1e-6, amsgrad=False)

        self.tasks_paths[self.current_train_task_idx] = current_path
        self.current_train_task_idx += 1

    def before_eval_exp(self, strategy, **kwargs):
        self.adapt_dataloder(strategy, self.current_eval_task_idx)
        print('plugin before eval, task idx = ', self.current_eval_task_idx)
        task_path = self.tasks_paths[self.current_eval_task_idx]
        print('selected path = ', task_path)
        strategy.model.set_path(task_path)

    def after_eval_exp(self, strategy, **kwargs):
        self.current_eval_task_idx += 1

    def after_eval(self, strategy, **kwargs):
        self.current_eval_task_idx = 0

    # def after_training_iteration(self, strategy, **kwargs):
    #     strategy.stop_training()

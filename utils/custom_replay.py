from avalanche.training.plugins import StrategyPlugin
from avalanche.training.plugins.evaluation import default_logger
from avalanche.training.strategies.base_strategy import BaseStrategy
from torch.utils.data.dataloader import DataLoader
from avalanche.benchmarks.utils.data_loader import _default_collate_mbatches_fn
from utils.custom_plugins import RehersalBuffer
import torch.utils.data


class ReplayPluginModified(StrategyPlugin):
    """
    Experience replay plugin.

    Handles an external memory filled with randomly selected
    patterns and implementing `before_training_exp` and `after_training_exp`
    callbacks. 
    The `before_training_exp` callback is implemented in order to use the
    dataloader that creates mini-batches with examples from both training
    data and external memory. The examples in the mini-batch is balanced 
    such that there are the same number of examples for each experience.    

    The `after_training_exp` callback is implemented in order to add new 
    patterns to the external memory.

    The :mem_size: attribute controls the total number of patterns to be stored 
    in the external memory.
    """

    def __init__(self, mem_size: int = 200, num_workers: int = 10, shuffle: bool = True):
        """
        :param storage_policy: The policy that controls how to add new exemplars
                        in memory
        """
        super().__init__()
        self.mem_size = mem_size
        self.datasets_buffer = []
        self.num_workers = num_workers
        self.shuffle = shuffle

    def before_training_exp(self, strategy, **kwargs):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        if len(self.datasets_buffer) == 0:
            return

        dataset_list = list(self.datasets_buffer)
        dataset_list.append(strategy.adapted_dataset)

        concat_dataset = RehersalBuffer(dataset_list, infinite=False)
        strategy.dataloader = DataLoader(
            concat_dataset,
            num_workers=self.num_workers,
            batch_size=strategy.train_mb_size // (len(self.datasets_buffer)+1),
            shuffle=self.shuffle,
            collate_fn=_default_collate_mbatches_fn
        )

    def after_training_exp(self, strategy: "BaseStrategy", **kwargs):
        self.datasets_buffer.append(strategy.experience.dataset)

        new_size = self.mem_size // (len(self.datasets_buffer)+1)
        for i in range(len(self.datasets_buffer)):
            indicies = torch.randperm(len(self.datasets_buffer[i]))[:new_size]
            self.datasets_buffer[i] = torch.utils.data.Subset(self.datasets_buffer[i], indicies)


class ReplayModified(BaseStrategy):
    """ Experience replay strategy.

    See ReplayPlugin for more details.
    This strategy does not use task identities.
    """

    def __init__(self, model, optimizer, criterion,
                 mem_size: int = 200,
                 train_mb_size: int = 1, train_epochs: int = 1,
                 eval_mb_size: int = None, device=None,
                 plugins=None,
                 evaluator=default_logger, eval_every=-1):
        rp = ReplayPluginModified(mem_size)
        if plugins is None:
            plugins = [rp]
        else:
            plugins.append(rp)
        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device,
            plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

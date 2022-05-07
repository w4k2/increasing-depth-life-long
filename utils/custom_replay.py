import torch
import torch.utils.data
from avalanche.training.plugins import StrategyPlugin
from avalanche.training.plugins.evaluation import default_logger
from avalanche.training.strategies.base_strategy import BaseStrategy
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader


class MutliDataset:
    def __init__(self, datasets):
        self.datasets = datasets
        self.len = sum(len(dataset) for dataset in datasets)

    def __getitem__(self, index):
        dataset_index, sample_index = index
        sample = self.datasets[dataset_index][sample_index]
        return sample

    def __len__(self):
        return self.len


class RehersalSampler():
    def __init__(self, dataset_sizes, dataset_samplers, batch_size, drop_last, oversample_small_tasks=False):
        self.dataset_sizes = dataset_sizes
        self.dataset_active = [True for _ in dataset_sizes]
        self.dataset_samplers = dataset_samplers
        self.dataset_samplers_iter = [iter(sampler) for sampler in dataset_samplers]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.oversample_small_tasks = oversample_small_tasks

        self.len = None
        if self.drop_last:
            self.len = sum(self.dataset_sizes) // self.batch_size
        else:
            self.len = (sum(self.dataset_sizes) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        self.dataset_active = [True for _ in self.dataset_sizes]
        self.dataset_samplers_iter = [iter(sampler) for sampler in self.dataset_samplers]

        batch = []
        num_generated = 0
        i = -1
        while num_generated < len(self):
            i += 1
            i = i % len(self.dataset_sizes)
            if not any(self.dataset_active):
                break
            if not self.dataset_active[i]:
                continue
            try:
                j = next(self.dataset_samplers_iter[i])
            except StopIteration:
                if self.oversample_small_tasks:
                    self.dataset_samplers_iter[i] = iter(self.dataset_samplers[i])
                    j = next(self.dataset_samplers_iter[i])
                else:
                    self.dataset_active[i] = False
                    continue
            idx = (i, j)
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                num_generated += 1
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        return self.len


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
        concat_dataset = MutliDataset(dataset_list)

        sampler = RehersalSampler(dataset_sizes=[len(dataset) for dataset in dataset_list],
                                  dataset_samplers=[RandomSampler(dataset) for dataset in dataset_list],
                                  batch_size=strategy.train_mb_size,
                                  drop_last=False)

        def collate(mbatches):
            batch = []
            for i in range(len(mbatches[0])):
                elems = list()
                for sample in mbatches:
                    elem = sample[i]
                    if type(elem) != torch.Tensor and type(elem) == int:
                        elem = torch.Tensor([elem]).to(torch.long)
                    elems.append(elem)
                t = torch.stack(elems, dim=0)
                if i >= 1:
                    t.squeeze_()
                batch.append(t)
            return batch

        strategy.dataloader = DataLoader(
            concat_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate
        )

    def after_training_exp(self, strategy: "BaseStrategy", **kwargs):
        self.datasets_buffer.append(strategy.experience.dataset)
        new_size = self.mem_size // len(self.datasets_buffer)
        for i in range(len(self.datasets_buffer)):
            self.datasets_buffer[i] = self.dataset_subset(self.datasets_buffer[i], new_size)

    def dataset_subset(self, dataset, new_size):
        indices = torch.randperm(len(dataset))[:new_size]
        subset = None
        if type(dataset) == torch.utils.data.Subset:
            dataset.indices = [dataset.indices[i] for i in indices]
            subset = dataset
        else:
            subset = torch.utils.data.Subset(dataset, indices)
        return subset


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

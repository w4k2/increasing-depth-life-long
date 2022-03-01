import torch
import sys


class RehersalBuffer:
    def __init__(self, datasets, infinite=True):
        self.datasets = datasets
        self.infinite = infinite

    def __getitem__(self, index):
        inputs = []
        targets = []
        task_idx = []

        for dataset in self.datasets:
            batch = dataset[index % len(dataset)]
            inputs.append(batch[0])
            targets.append(batch[1])
            if len(batch) == 3:
                task_idx.append(batch[2])

        inputs = torch.stack(inputs)
        inputs = inputs.to(torch.float32)
        inputs.requires_grad = False
        targets = torch.Tensor(targets).to(torch.int64)
        targets.requires_grad = False
        if len(batch) == 3:
            task_idx = torch.Tensor(task_idx).to(torch.int64)
            task_idx.requires_grad = False
            return inputs, targets, task_idx
        return inputs, targets

    def __len__(self):
        length = None
        if self.infinite:
            length = sys.maxsize
        else:
            length = sum(len(dataset) for dataset in self.datasets)
        return length

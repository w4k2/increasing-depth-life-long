from datasets import FashionMnistDataset
from torch.utils.data import DataLoader, SubsetRandomSampler


class DataProvider:

    def get_fashion_mnist_dataloaders(self, batch_size=128):
        dataset = FashionMnistDataset().load_dataset()
        dataloaders = [
            DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                       sampler=SubsetRandomSampler(indices=list(range(90000)))),
            DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                       sampler=SubsetRandomSampler(indices=list(range(90000, 180000)))),
            DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                       sampler=SubsetRandomSampler(indices=list(range(90000)))),
            DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                       sampler=SubsetRandomSampler(indices=list(range(90000, 180000))))
        ]
        return dataloaders

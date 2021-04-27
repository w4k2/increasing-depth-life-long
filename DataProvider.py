from FashionMnistDataset import *
from torch.utils.data import DataLoader, SubsetRandomSampler


def get_celeba_dataloaders(self, batch_size=128):
    dataset = CelebADataset().load_dataset()
    dataloaders = [
        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                   sampler=SubsetRandomSampler(indices=list(range(143258)))),
        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                   sampler=SubsetRandomSampler(indices=list(range(70000, 200000)))),
        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                   sampler=SubsetRandomSampler(indices=list(range(143258, 277006)))),
        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                   sampler=SubsetRandomSampler(indices=list(range(70000, 200000)))),
        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                   sampler=SubsetRandomSampler(indices=list(range(143258)))),
    ]

    return dataloaders


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


def get_cifar10_dataloaders(self, batch_size=128):
    dataset = Cifar10Dataset().load_dataset()
    dataloaders = [
        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                   sampler=SubsetRandomSampler(indices=list(range(250000, 500000)))),
        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                   sampler=SubsetRandomSampler(indices=list(range(250000)))),
        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                   sampler=SubsetRandomSampler(indices=list(range(250000, 500000)))),
        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                   sampler=SubsetRandomSampler(indices=list(range(250000))))
    ]
    return dataloaders

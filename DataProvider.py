from torch.utils.data import DataLoader, SubsetRandomSampler
from Cifar10Dataset import *
from CelebADataset import *
from FashionMnistDataset import *
from TinyImagenetDataset import *


def get_celeba_dataloaders(batch_size=128):
    dataset = CelebADataset().load_dataset()
    dataloaders = [
        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                   sampler=SubsetRandomSampler(indices=list(range(143258)))),
        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                   sampler=SubsetRandomSampler(indices=list(range(143258, 277006)))),
        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                   sampler=SubsetRandomSampler(indices=list(range(143258)))),
        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                   sampler=SubsetRandomSampler(indices=list(range(143258, 277006)))),
        # DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
        #            sampler=SubsetRandomSampler(indices=list(range(143258, 277006)))),
        # DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
        #            sampler=SubsetRandomSampler(indices=list(range(70000, 200000)))),
        # DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
        #            sampler=SubsetRandomSampler(indices=list(range(143258)))),
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


def get_cifar10_dataloaders(batch_size=128):
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


def get_tinyimagenet_dataloaders(batch_size=128):

    dataloaders = [
        DataLoader(dataset=tiny_imagenet_dataset(['n02123394', 'n02125311', 'n02106662', 'n02113799' 'n02190166',
                   'n02206856'], [0, 0, 1, 1, 2, 2]), batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(dataset=tiny_imagenet_dataset(['n02123045', 'n02129165', 'n02099712', 'n02099601', 'n02226429',
                   'n02231487'], [0, 0, 1, 1, 2, 2]), batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(dataset=tiny_imagenet_dataset(['n02124075', 'n02094433', 'n02085620', 'n02233338', 'n02236044',
                   'n02268443'], [0, 1, 1, 2, 2, 2]), batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(dataset=tiny_imagenet_dataset(['n02123394', 'n02125311', 'n02106662', 'n02113799' 'n02190166',
                   'n02206856'], [0, 0, 1, 1, 2, 2]), batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(dataset=tiny_imagenet_dataset(['n02123045', 'n02129165', 'n02099712', 'n02099601', 'n02226429',
                   'n02231487'], [0, 0, 1, 1, 2, 2]), batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(dataset=tiny_imagenet_dataset(['n02124075', 'n02094433', 'n02085620', 'n02233338', 'n02236044',
                   'n02268443'], [0, 1, 1, 2, 2, 2]), batch_size=batch_size, shuffle=True, drop_last=True),
    ]
    return dataloaders

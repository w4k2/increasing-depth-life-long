from torch.utils.data import DataLoader, SubsetRandomSampler
from Cifar10Dataset import *
# from CelebADataset import *
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
    # dataloaders = list()
    # for _ in range(2):
    #     for _ in range(10):
    #         dataloaders.append(DataLoader(dataset=tiny_imagenet_dataset(['n02123394', 'n02123045', 'n02106662', 'n02113799' 'n02190166',
    #                            'n02206856'], [0, 0, 1, 1, 2, 2]), batch_size=batch_size, shuffle=True, drop_last=True))
    #     for _ in range(10):
    #         dataloaders.append(DataLoader(dataset=tiny_imagenet_dataset(['n02125311', 'n02129165', 'n02099712', 'n02099601',
    #                            'n02226429', 'n02231487'], [0, 0, 1, 1, 2, 2]), batch_size=batch_size, shuffle=True, drop_last=True))

    dataloaders = list()
    single_dataloder_repeats = 5
    for _ in range(3):
        for _ in range(single_dataloder_repeats):
            dataloaders.append(DataLoader(dataset=tiny_imagenet_dataset(
                ['n02129165', 'n02106662', 'n02395406', 'n02486410', 'n03126707', 'n01644900', 'n03662601', 'n03179701', 'n09332890',
                    'n07768694', 'n08496334', 'n02268443', 'n03937543', 'n07711569', 'n03930313', 'n03400231', 'n04540053', 'n01774384'],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
                batch_size=batch_size, shuffle=True, drop_last=True))
        for _ in range(single_dataloder_repeats):
            dataloaders.append(DataLoader(dataset=tiny_imagenet_dataset(
                ['n02125311', 'n02094433', 'n02415577', 'n02480495', 'n07646821', 'n01629819', 'n03599486', 'n04099969', 'n09193705',
                 'n07753592', 'n03763968', 'n07975909', 'n03983396', 'n07583066', 'n02892201', 'n04398044', 'n02802426', 'n01984695'],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
                batch_size=batch_size, shuffle=True, drop_last=True))
        for _ in range(single_dataloder_repeats):
            dataloaders.append(DataLoader(dataset=tiny_imagenet_dataset(
                ['n02124075', 'n02099601', 'n02423022', 'n02481823', 'n02056570', 'n01641577', 'n03670208', 'n03014705', 'n09256479',
                 'n12267677', 'n04254777', 'n02233338', 'n02823428', 'n07873807', 'n02927161', 'n04596742', 'n04023962', 'n01770393'],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
                batch_size=batch_size, shuffle=True, drop_last=True))

    return dataloaders

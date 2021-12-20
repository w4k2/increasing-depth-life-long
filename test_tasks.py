import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as trans
import torchvision.models
from avalanche.benchmarks.classic import PermutedMNIST, SplitCIFAR100, SplitMNIST, SplitCIFAR10, SplitTinyImageNet, CORe50
from avalanche.training.strategies import BaseStrategy, EWC
from avalanche.models import SimpleMLP
import matplotlib.pyplot as plt

from avalanche.benchmarks.generators import nc_benchmark, dataset_benchmark
from avalanche.benchmarks.datasets import MNIST, FashionMNIST, KMNIST, EMNIST, \
    QMNIST, FakeData, CocoCaptions, CocoDetection, LSUN, ImageNet, CIFAR10, \
    CIFAR100, STL10, SVHN, PhotoTour, SBU, Flickr8k, Flickr30k, VOCDetection, \
    VOCSegmentation, Cityscapes, SBDataset, USPS, Kinetics400, HMDB51, UCF101, \
    CelebA, CORe50Dataset, TinyImagenet, CUB200, OpenLORIS


def main():
    benchmark = CORe50(scenario="nic")  # scenarios: 'ni', 'nc', 'nic', 'nicv2_79', 'nicv2_196' or 'nicv2_391'
    # classes_per_task = benchmark.n_classes_per_exp[0]
    # print(classes_per_task)

    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream
    new_test_stream = []

    for i in range(benchmark.n_experiences):
        print('train')
        train_data = train_stream[i]
        print(train_data)
        # print(dir(train_data))
        print(train_data.classes_in_this_experience)
        print(len(train_data.dataset))

        classes_in_exp = train_data.classes_in_this_experience
        classes_in_exp = set(classes_in_exp)
        # classes_in_exp = torch.Tensor(classes_in_exp)

        print('test')
        test_data = test_stream[0]
        # print(dir(test_data))
        # print(test_data)
        # # print(dir(test_data))
        # print(test_data.classes_in_this_experience)
        # print(len(test_data.dataset))

        # print(test_data.dataset)
        # print(dir(test_data.dataset))
        # print(test_data.dataset._dataset)
        # print(dir(test_data.dataset._dataset))
        # print(test_data.dataset._dataset._dataset)

        # print(test_data.dataset._dataset._dataset)
        # print(test_data.dataset._dataset._dataset.targets)
        # targets = test_data.dataset._dataset._dataset.targets
        # targets = torch.Tensor(targets)
        # print(targets.shape)
        # print(classes_in_exp.expand(targets.shape[0], classes_in_exp.shape[0]).shape)
        # idx = (targets.unsqueeze(1) == classes_in_exp.expand(targets.shape[0], classes_in_exp.shape[0])).any(axis=1)
        # print(idx.shape)
        # print(idx)

        images = []
        targets = []
        for img, t in zip(test_data.dataset._dataset._dataset.imgs, test_data.dataset._dataset._dataset.targets):
            if t in classes_in_exp:
                images.append(img)
                targets.append(t)

        test_data.dataset._dataset._dataset.imgs = images
        test_data.dataset._dataset._dataset.targets = targets
        test_data.classes_in_this_experience = classes_in_exp
        new_test_stream.append(test_data)

        # test_data.dataset._dataset._dataset.imgs = [img for img in test_data.dataset._dataset._dataset.imgs if img i]  # torch.masked_select(test_data.dataset._dataset._dataset.imgs, idx)
        # test_data.dataset._dataset._dataset.targets =  # torch.masked_select(test_data.dataset._dataset._dataset.targets, idx)

        # exit()

    classes_per_task = [len(exp.classes_in_this_experience) for exp in benchmark.train_stream]
    print(classes_per_task)

    classes_per_task = [len(exp.classes_in_this_experience) for exp in new_test_stream]
    print(classes_per_task)

# def main():
#     device = torch.device('cuda')
#     benchmark = SplitCIFAR100(n_experiences=10, seed=42)

#     # train_stream = benchmark.train_stream
#     # test_stream = benchmark.test_stream
#     # classes_per_task = benchmark.n_classes_per_exp[0]

#     # train_SVHN = SVHN()

#     mnist_transforms = trans.Compose([trans.ToTensor(), trans.Normalize((0.1307,), (0.3081,)), trans.Lambda(lambda x: torch.cat([x, x, x], dim=0))])
#     cifar_transforms = trans.Compose([trans.ToTensor(), trans.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

#     benchmark = dataset_benchmark(
#         [MNIST('./data/datasets', train=True, transform=mnist_transforms, download=True), CIFAR10('./data/datasets', train=True, transform=cifar_transforms, download=True)],
#         [MNIST('./data/datasets', train=True, transform=mnist_transforms, download=True), CIFAR10('./data/datasets', train=True, transform=cifar_transforms, download=True)],
#         train_transform=None,
#         eval_transform=None,
#     )
#     train_stream = benchmark.train_stream
#     test_stream = benchmark.test_stream
#     classes_per_task = 10

#     print(dir(benchmark))
#     print('n_experiences = ', benchmark.n_experiences)

#     for i in range(benchmark.n_experiences):
#         test_data = test_stream[i]
#         print(dir(test_data))
#         print(test_data.dataset)
#         for j, batch in enumerate(test_data.dataset):
#             if j > 5:
#                 break
#             print(batch[1])
#             # print(batch)
#             # plt.figure()
#             # plt.imshow(batch[0])
#             # break
#         print()

#     # plt.show()
#     # exit()

#     # model = SimpleMLP(num_classes=classes_per_task)
#     model = torchvision.models.resnet18(num_classes=classes_per_task)
#     optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0)
#     criterion = nn.CrossEntropyLoss()
#     ewc_lambda = 100
#     strategy = EWC(model, optimizer, criterion,
#                    ewc_lambda=ewc_lambda, train_mb_size=32, eval_mb_size=32,
#                    device=device, train_epochs=10)

#     results = []
#     for i, train_task in enumerate(train_stream):
#         eval_stream = [test_stream[i]]
#         strategy.train(train_task, eval_stream, num_workers=20)
#         selected_tasks = [test_stream[j] for j in range(0, i+1)]
#         eval_results = strategy.eval(selected_tasks)
#         results.append(eval_results)

#     print(results)


if __name__ == '__main__':
    main()

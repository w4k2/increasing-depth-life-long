import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
from avalanche.benchmarks.classic import PermutedMNIST, SplitCIFAR100, SplitMNIST, SplitCIFAR10, SplitTinyImageNet
from avalanche.training.strategies import BaseStrategy, EWC
from avalanche.models import SimpleMLP

from avalanche.benchmarks.generators import nc_benchmark, dataset_benchmark
from avalanche.benchmarks.datasets import MNIST, FashionMNIST, KMNIST, EMNIST, \
    QMNIST, FakeData, CocoCaptions, CocoDetection, LSUN, ImageNet, CIFAR10, \
    CIFAR100, STL10, SVHN, PhotoTour, SBU, Flickr8k, Flickr30k, VOCDetection, \
    VOCSegmentation, Cityscapes, SBDataset, USPS, Kinetics400, HMDB51, UCF101, \
    CelebA, CORe50Dataset, TinyImagenet, CUB200, OpenLORIS


def main():
    device = torch.device('cuda')
    benchmark = SplitCIFAR100(n_experiences=10, seed=42)

    # train_stream = benchmark.train_stream
    # test_stream = benchmark.test_stream
    # classes_per_task = benchmark.n_classes_per_exp[0]

    train_SVHN = SVHN()

    scenario_custom_task_labels = dataset_benchmark(
        [train_MNIST_task0, train_cifar10_task1],
        [test_MNIST_task0, test_cifar10_task1]
    )

    # model = SimpleMLP(num_classes=classes_per_task)
    model = torchvision.models.resnet18(num_classes=classes_per_task)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    ewc_lambda = 100
    strategy = EWC(model, optimizer, criterion,
                   ewc_lambda=ewc_lambda, train_mb_size=32, eval_mb_size=32,
                   device=device, train_epochs=10)

    results = []
    for i, train_task in enumerate(train_stream):
        eval_stream = [test_stream[i]]
        strategy.train(train_task, eval_stream, num_workers=20)
        selected_tasks = [test_stream[j] for j in range(0, i+1)]
        eval_results = strategy.eval(selected_tasks)
        results.append(eval_results)

    print(results)


if __name__ == '__main__':
    main()

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transf
import stochastic_depth_modified
import stochastic_depth_model
import avalanche

from avalanche.benchmarks.classic import PermutedMNIST, SplitCIFAR100, SplitMNIST, SplitCIFAR10
from avalanche.training.strategies import BaseStrategy
from avalanche_plugins import *


def main():
    args = parse_args()

    device = torch.device(args.device)
    train_stream, test_stream = get_data(args.dataset)
    input_channels = 1 if args.dataset == 'mnist' else 3
    model = stochastic_depth_modified.resnet50_StoDepth_lineardecay(num_classes=10, input_channels=input_channels)
    # model = stochastic_depth_model.resnet50_StoDepth_lineardecay(num_classes=10)
    # model = torchvision.models.resnet18(num_classes=10)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6, amsgrad=False)
    criterion = nn.CrossEntropyLoss()

    strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                            train_epochs=args.n_epochs, plugins=[StochasticDepthPlugin(device)], device=device)

    results = []
    for i, train_task in enumerate(train_stream):
        strategy.train(train_task, num_workers=20)
        selected_tasks = [test_stream[j] for j in range(0, i+1)]
        eval_results = strategy.eval(selected_tasks)
        results.append(eval_results)

    print(results)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='cifar100', choices=('cifar100', 'cifar10', 'mnist'))
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--n_epochs', default=20, type=int)

    args = parser.parse_args()
    return args


def get_data(dataset_name):
    norm_stats = None
    if dataset_name == 'cifar10':
        norm_stats = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    elif dataset_name == 'cifar100':
        norm_stats = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    elif dataset_name == 'mnist':
        norm_stats = (33.3184,), (78.5675,)

    train_transforms = transf.Compose([
        transf.RandomHorizontalFlip(p=0.5),
        transf.Resize((224, 224)),
        transf.ToTensor(),
        transf.Normalize(*norm_stats)
    ])
    eval_transforms = transf.Compose([
        transf.Resize((224, 224)),
        transf.ToTensor(),
        transf.Normalize(*norm_stats)
    ])

    benchmark = None
    if dataset_name == 'cifar10':
        benchmark = SplitCIFAR10(n_experiences=2,
                                 train_transform=train_transforms,
                                 eval_transform=eval_transforms)
    elif dataset_name == 'cifar100':
        benchmark = SplitCIFAR100(n_experiences=10,
                                  train_transform=train_transforms,
                                  eval_transform=eval_transforms)
    elif dataset_name == 'mnist':
        benchmark = SplitMNIST(5,
                               train_transform=train_transforms,
                               eval_transform=eval_transforms
                               )
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream
    return train_stream, test_stream


if __name__ == '__main__':
    main()

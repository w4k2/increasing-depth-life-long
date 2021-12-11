import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transf
import stochastic_depth_lifelong
import stochastic_depth
import mlflow

from avalanche.benchmarks.classic import PermutedMNIST, SplitCIFAR100, SplitMNIST, SplitCIFAR10
from avalanche.training.strategies import BaseStrategy, EWC
from avalanche.models import SimpleMLP
from mlflow_logger import MLFlowLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics

from avalanche.logging import InteractiveLogger
from custom_plugins import *


def main():
    args = parse_args()

    device = torch.device(args.device)
    train_stream, test_stream = get_data(args.dataset, args.seed)
    strategy = get_method(args, device)

    results = []
    for i, train_task in enumerate(train_stream):
        strategy.train(train_task, num_workers=20)
        selected_tasks = [test_stream[j] for j in range(0, i+1)]
        eval_results = strategy.eval(selected_tasks)
        results.append(eval_results)

    print(results)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', default=None, help='mlflow run name')
    parser.add_argument('--method', default='ll-stochastic-depth', choices=('baseline', 'll-stochastic-depth', 'ewc'))
    parser.add_argument('--dataset', default='cifar100', choices=('cifar100', 'cifar10', 'mnist', 'permutation-mnist'))
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_epochs', default=20, type=int)

    args = parser.parse_args()
    return args


def get_data(dataset_name, seed):
    benchmark = None
    if dataset_name == 'cifar10':
        norm_stats = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        train_transforms, eval_transforms = get_transforms(norm_stats)
        benchmark = SplitCIFAR10(n_experiences=2,
                                 train_transform=train_transforms,
                                 eval_transform=eval_transforms,
                                 seed=seed
                                 )
    elif dataset_name == 'cifar100':
        norm_stats = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        train_transforms, eval_transforms = get_transforms(norm_stats)
        benchmark = SplitCIFAR100(n_experiences=10,
                                  train_transform=train_transforms,
                                  eval_transform=eval_transforms,
                                  seed=seed
                                  )
    elif dataset_name == 'mnist':
        norm_stats = (0.1307,), (0.3081,)
        train_transforms, eval_transforms = get_transforms(norm_stats, use_hflip=False)
        benchmark = SplitMNIST(n_experiences=5,
                               train_transform=train_transforms,
                               eval_transform=eval_transforms,
                               seed=seed
                               )
    elif dataset_name == 'permutation-mnist':
        norm_stats = (0.1307,), (0.3081,)
        train_transforms, eval_transforms = get_transforms(norm_stats, use_hflip=False)
        benchmark = PermutedMNIST(n_experiences=10,
                                  train_transform=train_transforms,
                                  eval_transform=eval_transforms,
                                  seed=seed
                                  )

    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream
    return train_stream, test_stream


def get_transforms(norm_stats, use_hflip=True):
    transform_list = [
        transf.Resize((224, 224)),
        transf.ToTensor(),
        transf.Normalize(*norm_stats)
    ]
    if use_hflip:
        transform_list.insert(0, transf.RandomHorizontalFlip(p=0.5))
    train_transforms = transf.Compose(transform_list)
    eval_transforms = transf.Compose([
        transf.Resize((224, 224)),
        transf.ToTensor(),
        transf.Normalize(*norm_stats)
    ])
    return train_transforms, eval_transforms


def get_method(args, device):
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(args.__dict__)
        active_run = mlflow.active_run()
        mlf_logger = MLFlowLogger(active_run.info.run_id)

    input_channels = 1 if 'mnist' in args.dataset else 3
    evaluation_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loggers=[InteractiveLogger(), mlf_logger],
        suppress_warnings=True)

    if args.method == 'baseline':
        model = stochastic_depth.resnet50_StoDepth_lineardecay(num_classes=10)
        plugin = BaselinePlugin(model, device)
        strategy = get_base_strategy(args.batch_size, args.n_epochs, device, model, plugin, evaluation_plugin)
    elif args.method == 'll-stochastic-depth':
        model = stochastic_depth_lifelong.resnet50_StoDepth_lineardecay(num_classes=10, input_channels=input_channels)
        plugin = StochasticDepthPlugin(device)
        strategy = get_base_strategy(args.batch_size, args.n_epochs, device, model, plugin, evaluation_plugin)
    elif args.method == 'ewc':
        # model = SimpleMLP(num_classes=10)
        # model = torchvision.models.resnet18(num_classes=10)
        model = torchvision.models.vgg11(num_classes=10)
        # model = stochastic_depth.resnet50_StoDepth_lineardecay(num_classes=10)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()
        ewc_lambda = 100
        strategy = EWC(model, optimizer, criterion,
                       ewc_lambda=ewc_lambda, train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                       device=device, train_epochs=args.n_epochs, plugins=[], evaluator=evaluation_plugin)
        # ConvertedLabelsPlugin()])
    return strategy


def get_base_strategy(batch_size, n_epochs, device, model, plugin, evaluation_plugin):
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6, amsgrad=False)
    criterion = nn.CrossEntropyLoss()
    strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=batch_size, eval_mb_size=batch_size,
                            train_epochs=n_epochs, plugins=[plugin], device=device, evaluator=evaluation_plugin)
    return strategy


if __name__ == '__main__':
    main()

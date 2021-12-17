import argparse
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transf
import stochastic_depth_lifelong
import stochastic_depth
import mlflow

from avalanche.benchmarks.classic import PermutedMNIST, SplitCIFAR100, SplitMNIST, SplitCIFAR10, SplitTinyImageNet
from avalanche.evaluation.metrics.confusion_matrix import StreamConfusionMatrix
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
    strategy, mlf_logger = get_method(args, device, use_mlflow=not args.debug)

    results = []
    for i, train_task in enumerate(train_stream):
        strategy.train(train_task, num_workers=20)
        selected_tasks = [test_stream[j] for j in range(0, i+1)]
        eval_results = strategy.eval(selected_tasks)
        results.append(eval_results)

    print(results)

    result = compute_conf_matrix(test_stream, strategy)
    mlf_logger.log_conf_matrix(result)
    mlf_logger.log_model(strategy.model)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', default=None, help='mlflow run name')
    parser.add_argument('--experiment', default='Default', help='flow experiment name')

    parser.add_argument('--method', default='ll-stochastic-depth', choices=('baseline', 'll-stochastic-depth', 'ewc'))
    parser.add_argument('--dataset', default='cifar100', choices=('cifar100', 'cifar10', 'mnist', 'permutation-mnist'))
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_epochs', default=20, type=int)
    parser.add_argument('--debug', action='store_true', help='if true, execute only one iteration in training epoch')

    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--entropy_threshold', default=0.7, type=float, help='entropy threshold for adding new node attached directly to backbone')

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
        benchmark = SplitMNIST(n_experiences=10,
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
    elif dataset_name == 'tiny-imagenet':
        norm_stats = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        train_transforms, eval_transforms = get_transforms(norm_stats)
        benchmark = SplitTinyImageNet(n_experiences=10,
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


def get_method(args, device, use_mlflow=True):
    loggers = [InteractiveLogger()]

    mlf_logger = None
    if use_mlflow:
        mlf_logger = MLFlowLogger(experiment_name=args.experiment)
        loggers.append(mlf_logger)

    input_channels = 1 if 'mnist' in args.dataset else 3
    evaluation_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loggers=loggers,
        suppress_warnings=True)

    plugins = list()
    if args.debug:
        plugins.append(DebugingPlugin())

    if args.method == 'baseline':
        model = stochastic_depth.resnet50_StoDepth_lineardecay(num_classes=10)
        plugins.append(BaselinePlugin(model, device))
        strategy = get_base_strategy(args.batch_size, args.n_epochs, device, model, plugins, evaluation_plugin, args.lr, args.weight_decay)
    elif args.method == 'll-stochastic-depth':
        # model = stochastic_depth_lifelong.resnet50_StoDepth_lineardecay(num_classes=10, input_channels=input_channels)
        model = stochastic_depth_lifelong.resnet18_StoDepth_lineardecay(num_classes=10, input_channels=input_channels)
        plugins.append(StochasticDepthPlugin(args.entropy_threshold, device))
        strategy = get_base_strategy(args.batch_size, args.n_epochs, device, model, plugins, evaluation_plugin, args.lr, args.weight_decay)
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
                       device=device, train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin)
        # ConvertedLabelsPlugin()])
    return strategy, mlf_logger


def get_base_strategy(batch_size, n_epochs, device, model, plugins, evaluation_plugin, lr, weight_decay):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False)
    criterion = nn.CrossEntropyLoss()
    strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=batch_size, eval_mb_size=batch_size,
                            train_epochs=n_epochs, plugins=plugins, device=device, evaluator=evaluation_plugin)
    return strategy


def compute_conf_matrix(test_stream, strategy):
    custom_plugin = None
    for plugin in strategy.plugins:
        if issubclass(type(plugin), ConvertedLabelsPlugin):
            custom_plugin = plugin
            break

    conf_matrix_metric = StreamConfusionMatrix(absolute_class_order=False)
    num_classes_per_task = 10
    with torch.no_grad():
        for i, strategy.experience in enumerate(test_stream):
            strategy.eval_dataset_adaptation()
            strategy.make_eval_dataloader()
            strategy.model = strategy.model_adaptation()

            class_mapping = custom_plugin.get_label_mapping(strategy.dataloader.dataset, i)
            for strategy.mbatch in strategy.dataloader:
                getattr(strategy, '_unpack_minibatch')()
                mb_output = strategy.forward()
                new_output = torch.zeros((mb_output.shape[0], 100))
                new_output[:, i*num_classes_per_task:(i+1)*num_classes_per_task] = mb_output

                mb_y = strategy.mb_y
                mb_y = torch.Tensor([class_mapping[l.item()] for l in mb_y])
                mb_y += i * num_classes_per_task
                mb_y = mb_y.to(torch.long)

                conf_matrix_metric.update(mb_y, new_output)

    result = conf_matrix_metric.result()
    return result


if __name__ == '__main__':
    main()

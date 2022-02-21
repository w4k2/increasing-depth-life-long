import argparse
from avalanche.training.strategies.strategy_wrappers import PNNStrategy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transf
import functools
import models.stochastic_depth_lifelong as stochastic_depth_lifelong
import models.stochastic_depth as stochastic_depth
import models.resnet as resnet
import models.reduced_resnet as reduced_resnet
import distutils.util

from avalanche.benchmarks.datasets import MNIST, FashionMNIST, CIFAR10
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.benchmarks.classic import PermutedMNIST, SplitCIFAR100, SplitMNIST, SplitCIFAR10, SplitTinyImageNet, CORe50
from avalanche.evaluation.metrics.confusion_matrix import StreamConfusionMatrix
from avalanche.training.strategies import BaseStrategy, EWC, GEM, Replay, SynapticIntelligence, Cumulative, LwF
from avalanche.models import SimpleMLP
from utils.mlflow_logger import MLFlowLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics

from avalanche.logging import InteractiveLogger
from utils.custom_plugins import *
from utils.custom_replay import *
from utils.custom_cumulative import *
from utils.custom_agem import *

import cProfile


def main():
    args = parse_args()
    run_experiment(args)


def run_experiment(args):
    torch.set_num_threads(1)

    device = torch.device(args.device)
    train_stream, test_stream, classes_per_task = get_data(args.dataset, args.n_experiences, args.seed, args.image_size)
    strategy, mlf_logger = get_method(args, device, classes_per_task, use_mlflow=not args.debug)

    results = []
    for i, train_task in enumerate(train_stream):
        if i >= args.train_on_experiences:
            break
        eval_stream = [test_stream[i]]
        strategy.train(train_task, eval_stream, num_workers=20)
        selected_tasks = [test_stream[j] for j in range(0, i+1)]
        eval_results = strategy.eval(selected_tasks)
        results.append(eval_results)
        forgetting = get_forgetting(eval_results)
        if forgetting is not None and forgetting > args.forgetting_stopping_threshold:
            print(f'Stopping training after task {i} due to large forgetting: {forgetting}')
            break

    # print(results)

    # if args.n_experiences * classes_per_task > 200:
    #     print('to many classes, skipping confusion matrix computation')
    # else:
    #     result = compute_conf_matrix(test_stream, strategy, classes_per_task)
    #     mlf_logger.log_conf_matrix(result)

    if mlf_logger:
        mlf_logger.log_model(strategy.model)
        mlf_logger.log_avrg_accuracy()


def get_forgetting(results):
    forgetting = None
    try:
        forgetting = results['StreamForgetting/eval_phase/test_stream']
    except KeyError:
        pass
    return forgetting


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', default=None, help='mlflow run name')
    parser.add_argument('--experiment', default='Default', help='flow experiment name')
    parser.add_argument('--nested_run', action='store_true', help='create nested run in mlflow')

    parser.add_argument('--method', default='agem', choices=('baseline', 'cumulative', 'll-stochastic-depth', 'ewc', 'si', 'gem', 'agem', 'pnn', 'replay', 'lwf'))
    parser.add_argument('--base_model', default='resnet18', choices=('resnet9', 'resnet18', 'reduced_resnet18', 'resnet50', 'resnet18-stoch', 'resnet50-stoch', 'vgg', 'simpleMLP'))
    parser.add_argument('--pretrained', default=True, type=distutils.util.strtobool, help='if True load weights pretrained on imagenet')
    parser.add_argument('--dataset', default='permutation-mnist', choices=('cifar100', 'cifar10', 'mnist', 'permutation-mnist', 'tiny-imagenet',
                        'cifar10-mnist-fashion-mnist', 'mnist-fashion-mnist-cifar10', 'fashion-mnist-cifar10-mnist', 'cores50'))
    parser.add_argument('--n_experiences', default=50, type=int)
    parser.add_argument('--train_on_experiences', default=50, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--num_workers', default=20, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--n_epochs', default=1, type=int)
    parser.add_argument('--image_size', default=64, type=int)
    parser.add_argument('--debug', action='store_true', help='if true, execute only one iteration in training epoch')

    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.8, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--entropy_threshold', default=0.7, type=float, help='entropy threshold for adding new node attached directly to backbone')  # 0.8 for cifar100
    parser.add_argument('--update_method', default='entropy', choices=('entropy', 'sequential', 'parallel'))
    parser.add_argument('--forgetting_stopping_threshold', default=0.5, type=float)

    args = parser.parse_args()
    return args


def get_data(dataset_name, n_experiences, seed, image_size):
    benchmark = None
    test_stream = None
    if dataset_name == 'cifar10':
        norm_stats = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        train_transforms, eval_transforms = get_transforms(norm_stats, image_size)
        benchmark = SplitCIFAR10(n_experiences=n_experiences,
                                 train_transform=train_transforms,
                                 eval_transform=eval_transforms,
                                 seed=seed
                                 )
        classes_per_task = benchmark.n_classes_per_exp
    elif dataset_name == 'cifar100':
        norm_stats = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        train_transforms, eval_transforms = get_transforms(norm_stats, image_size)
        benchmark = SplitCIFAR100(n_experiences=n_experiences,
                                  train_transform=train_transforms,
                                  eval_transform=eval_transforms,
                                  seed=seed,
                                  return_task_id=True
                                  )
        classes_per_task = benchmark.n_classes_per_exp
    elif dataset_name == 'mnist':
        norm_stats = (0.1307,), (0.3081,)
        train_transforms, eval_transforms = get_transforms(norm_stats, image_size, use_hflip=False, stack_channels=True)
        benchmark = SplitMNIST(n_experiences=n_experiences,
                               train_transform=train_transforms,
                               eval_transform=eval_transforms,
                               seed=seed
                               )
        classes_per_task = benchmark.n_classes_per_exp
    elif dataset_name == 'permutation-mnist':
        norm_stats = (0.1307,), (0.3081,)
        train_transforms, eval_transforms = get_transforms(norm_stats, image_size, use_hflip=False, stack_channels=True)
        benchmark = PermutedMNIST(n_experiences=n_experiences,
                                  train_transform=train_transforms,
                                  eval_transform=eval_transforms,
                                  seed=seed
                                  )
        classes_per_task = benchmark.n_classes_per_exp
    elif dataset_name == 'tiny-imagenet':
        norm_stats = (0.4443, 0.4395, 0.4250), (0.3138, 0.3181, 0.3182)
        train_transforms, eval_transforms = get_transforms(norm_stats, image_size)
        benchmark = SplitTinyImageNet(n_experiences=n_experiences,
                                      train_transform=train_transforms,
                                      eval_transform=eval_transforms,
                                      seed=seed,
                                      return_task_id=True,
                                      )
        classes_per_task = benchmark.n_classes_per_exp
    elif dataset_name == 'cifar10-mnist-fashion-mnist':
        benchmark = get_multidataset_benchmark(('cifar', 'mnist', 'fmnist'), image_size)
        classes_per_task = [10, 10, 10]
    elif dataset_name == 'mnist-fashion-mnist-cifar10':
        benchmark = get_multidataset_benchmark(('mnist', 'fmnist', 'cifar'), image_size)
        classes_per_task = [10, 10, 10]
    elif dataset_name == 'fashion-mnist-cifar10-mnist':
        benchmark = get_multidataset_benchmark(('fmnist', 'cifar', 'mnist'), image_size)
        classes_per_task = [10, 10, 10]
    elif dataset_name == 'cores50':
        norm_stats = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        train_transforms, eval_transforms = get_transforms(norm_stats, image_size)
        benchmark = CORe50(scenario="nc", train_transform=train_transforms, eval_transform=eval_transforms)
        classes_per_task = [len(exp.classes_in_this_experience) for exp in benchmark.train_stream]

        test_stream = []
        for i in range(benchmark.n_experiences):
            train_data = benchmark.train_stream[i]
            classes_in_exp = train_data.classes_in_this_experience
            classes_in_exp = set(classes_in_exp)
            test_data = benchmark.test_stream[0]

            images = []
            targets = []
            for img, t in zip(test_data.dataset._dataset._dataset.imgs, test_data.dataset._dataset._dataset.targets):
                if t in classes_in_exp:
                    images.append(img)
                    targets.append(t)

            test_data.dataset._dataset._dataset.imgs = images
            test_data.dataset._dataset._dataset.targets = targets
            test_data.classes_in_this_experience = classes_in_exp
            test_stream.append(test_data)

    train_stream = benchmark.train_stream
    if not test_stream:
        test_stream = benchmark.test_stream
    return train_stream, test_stream, classes_per_task


def get_multidataset_benchmark(order, image_size):
    cifar10_norm_stats = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
    mnist_norm_stats = (0.1307,), (0.3081,)
    fmnist_norm_stats = (0.2860,), (0.3530,)

    cifar10_train_transforms, cifar10_eval_transforms = get_transforms(cifar10_norm_stats, image_size)
    mnist_train_transforms, mnist_eval_transforms = get_transforms(mnist_norm_stats, image_size, use_hflip=False, stack_channels=True)
    fmnist_train_transforms, fmnist_eval_transforms = get_transforms(fmnist_norm_stats, image_size, use_hflip=False, stack_channels=True)

    train_datasets = []
    test_datasets = []

    for dataset_name in order:
        if dataset_name == 'cifar':
            train_datasets.append(CIFAR10('./data/datasets', train=True, transform=cifar10_train_transforms, download=True))
            test_datasets.append(CIFAR10('./data/datasets', train=False, transform=cifar10_eval_transforms, download=True))
        elif dataset_name == 'mnist':
            train_datasets.append(MNIST('./data/datasets', train=True, transform=mnist_train_transforms, download=True))
            test_datasets.append(MNIST('./data/datasets', train=False, transform=mnist_eval_transforms, download=True))
        elif dataset_name == 'fmnist':
            train_datasets.append(FashionMNIST('./data/datasets', train=True, transform=fmnist_train_transforms, download=True))
            test_datasets.append(FashionMNIST('./data/datasets', train=False, transform=fmnist_eval_transforms, download=True))
        else:
            raise ValueError("Invalid dataset name")

    benchmark = dataset_benchmark(train_datasets, test_datasets)
    return benchmark


def get_transforms(norm_stats, image_size, use_hflip=True, stack_channels=False):
    train_list = [
        transf.Resize((image_size, image_size)),
        transf.ToTensor(),
        transf.Normalize(*norm_stats)
    ]
    if use_hflip:
        train_list.insert(0, transf.RandomHorizontalFlip(p=0.5))
    if stack_channels:
        train_list.append(transf.Lambda(lambda x: torch.cat([x, x, x], dim=0)))
    train_transforms = transf.Compose(train_list)

    eval_list = [
        transf.Resize((image_size, image_size)),
        transf.ToTensor(),
        transf.Normalize(*norm_stats)
    ]
    if stack_channels:
        eval_list.append(transf.Lambda(lambda x: torch.cat([x, x, x], dim=0)))
    eval_transforms = transf.Compose(eval_list)
    return train_transforms, eval_transforms


def get_method(args, device, classes_per_task, use_mlflow=True):
    loggers = [InteractiveLogger()]

    mlf_logger = None
    if use_mlflow:
        mlf_logger = MLFlowLogger(experiment_name=args.experiment, nested=args.nested_run, run_name=args.run_name)
        mlf_logger.log_parameters(args.__dict__)
        loggers.append(mlf_logger)

    input_channels = 3
    evaluation_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loggers=loggers,
        suppress_warnings=True)

    plugins = list()
    if args.debug:
        plugins.append(DebugingPlugin())

    if args.method == 'baseline':
        print('classes_per_task = ', classes_per_task)
        model_creation_fn = functools.partial(get_base_model, model_name=args.base_model, input_channels=input_channels, pretrained=args.pretrained)
        plugins.append(BaselinePlugin(model_creation_fn, classes_per_task, device))
        model = get_base_model(args.base_model, classes_per_task[0], input_channels)
        strategy = get_base_strategy(args.batch_size, args.n_epochs, device, model, plugins, evaluation_plugin, args.lr, args.weight_decay)
    elif args.method == 'cumulative':
        model = resnet.resnet18_multihead(num_classes=classes_per_task[0], input_channels=input_channels, pretrained=args.pretrained)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = CumulativeModified(model, optimizer, criterion,
                                      train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                                      device=device, train_epochs=args.n_epochs, plugins=plugins,
                                      evaluator=evaluation_plugin, eval_every=-1
                                      )

    elif args.method == 'll-stochastic-depth':
        model = get_base_model_ll(args.base_model, classes_per_task[0], input_channels, pretrained=args.pretrained, update_method=args.update_method)
        plugins.append(StochasticDepthPlugin(args.entropy_threshold, device))
        strategy = get_base_strategy(args.batch_size, args.n_epochs, device, model, plugins, evaluation_plugin, args.lr, args.weight_decay)
    elif args.method == 'ewc':
        model = resnet.resnet18_multihead(num_classes=classes_per_task[0], input_channels=input_channels, pretrained=args.pretrained)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()
        plugins.append(ConvertedLabelsPlugin())
        ewc_lambda = 1000
        strategy = EWC(model, optimizer, criterion,
                       ewc_lambda=ewc_lambda, train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                       device=device, train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin)
    elif args.method == 'si':
        model = resnet.resnet18_multihead(num_classes=classes_per_task[0], input_channels=input_channels, pretrained=args.pretrained)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
        criterion = nn.CrossEntropyLoss()
        plugins.append(ConvertedLabelsPlugin())
        strategy = SynapticIntelligence(model, optimizer, criterion,
                                        si_lambda=1000, train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                                        device=device, train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin)
    elif args.method == 'gem':
        model = resnet.resnet18_multihead(num_classes=classes_per_task[0], input_channels=input_channels, pretrained=args.pretrained)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = GEM(model, optimizer, criterion, patterns_per_exp=250,
                       train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                       train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'agem':
        model = resnet.resnet18_multihead(num_classes=classes_per_task[0], input_channels=input_channels, pretrained=args.pretrained)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = AGEMModified(model, optimizer, criterion, patterns_per_exp=250, sample_size=256,
                                train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                                train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'pnn':
        num_channels = 1 if args.dataset in ('mnist', 'permutation-mnist') else 3
        in_features = args.image_size * args.image_size * num_channels
        strategy = PNNStrategy(num_layers=4, in_features=in_features, hidden_features_per_column=256,
                               lr=args.lr, momentum=args.momentum, train_mb_size=args.batch_size, eval_mb_size=args.batch_size,
                               train_epochs=args.n_epochs, device=device, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'replay':
        model = resnet.resnet18_multihead(num_classes=classes_per_task[0], input_channels=input_channels, pretrained=args.pretrained)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = ReplayModified(model, optimizer, criterion, mem_size=250*args.n_experiences,
                                  train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                                  train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)
    elif args.method == 'lwf':
        model = resnet.resnet18_multihead(num_classes=classes_per_task[0], input_channels=input_channels, pretrained=args.pretrained)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        strategy = LwF(model, optimizer, criterion, alpha=1.0, temperature=1.0,
                       train_mb_size=args.batch_size, eval_mb_size=args.batch_size, device=device,
                       train_epochs=args.n_epochs, plugins=plugins, evaluator=evaluation_plugin, eval_every=-1)

    return strategy, mlf_logger


def get_base_model(model_name, num_classes=10, input_channels=3, pretrained=False):
    if model_name == 'resnet18':
        model = resnet.resnet18(num_classes=num_classes, input_channels=input_channels, pretrained=pretrained)
    elif model_name == 'reduced_resnet18':
        model = reduced_resnet.resnet18(num_classes=num_classes, input_channels=input_channels, pretrained=pretrained)
    elif model_name == 'resnet50':
        model = resnet.resnet50(num_classes=num_classes, input_channels=input_channels)
    elif model_name == 'resnet18-stoch':
        model = stochastic_depth.resnet18_StoDepth_lineardecay(num_classes=num_classes)
    elif model_name == 'resnet50-stoch':
        model = stochastic_depth.resnet50_StoDepth_lineardecay(num_classes=num_classes)
    elif model_name == 'vgg':
        model = torchvision.models.vgg11(num_classes=num_classes)
    elif model_name == 'simpleMLP':
        model = SimpleMLP(num_classes=num_classes)
    else:
        raise ValueError('Invalid model name')
    return model


def get_base_model_ll(model_name, num_classes, input_channels, pretrained=False, update_method='entropy'):
    if 'resnet9' in model_name:
        model = stochastic_depth_lifelong.resnet9_StoDepth_lineardecay(num_classes=num_classes, input_channels=input_channels, update_method=update_method)
    elif 'resnet18' in model_name:
        model = stochastic_depth_lifelong.resnet18_StoDepth_lineardecay(num_classes=num_classes, input_channels=input_channels, pretrained=pretrained, update_method=update_method)
    elif 'resnet50' in model_name:
        model = stochastic_depth_lifelong.resnet50_StoDepth_lineardecay(num_classes=num_classes, input_channels=input_channels, update_method=update_method)
    else:
        raise ValueError('Invalid model name for ll-stochastic-depth method')
    return model


def get_base_strategy(batch_size, n_epochs, device, model, plugins, evaluation_plugin, lr, weight_decay):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False)
    criterion = nn.CrossEntropyLoss()
    strategy = BaseStrategy(model, optimizer, criterion, train_mb_size=batch_size, eval_mb_size=batch_size,
                            train_epochs=n_epochs, plugins=plugins, device=device, evaluator=evaluation_plugin,
                            eval_every=-1)
    return strategy


def compute_conf_matrix(test_stream, strategy, classes_per_task):
    custom_plugin = None
    for plugin in strategy.plugins:
        if issubclass(type(plugin), ConvertedLabelsPlugin):
            custom_plugin = plugin
            break

    conf_matrix_metric = StreamConfusionMatrix(absolute_class_order=False)
    with torch.no_grad():
        for i, strategy.experience in enumerate(test_stream):
            strategy.eval_dataset_adaptation()
            strategy.make_eval_dataloader()
            strategy.model = strategy.model_adaptation()

            class_mapping = custom_plugin.get_label_mapping(strategy.dataloader.dataset, i)
            for strategy.mbatch in strategy.dataloader:
                getattr(strategy, '_unpack_minibatch')()
                mb_output = strategy.forward()
                new_output = torch.zeros((mb_output.shape[0], 10 * classes_per_task))
                new_output[:, i*classes_per_task:(i+1)*classes_per_task] = mb_output

                mb_y = strategy.mb_y
                mb_y = torch.Tensor([class_mapping[l.item()] for l in mb_y])
                mb_y += i * classes_per_task
                mb_y = mb_y.to(torch.long)

                conf_matrix_metric.update(mb_y, new_output)

    result = conf_matrix_metric.result()
    return result


if __name__ == '__main__':
    main()
    # cProfile.run('main()')

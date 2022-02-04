# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision.models

# from torchvision.models import resnet18
# from avalanche.training import AGEM
# from avalanche.benchmarks.classic import SplitCIFAR100, PermutedMNIST
# from torchvision import transforms as transf
# # from models import MLP, MultiHeadReducedResNet18


# # # norm_stats = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
# # norm_stats = (0.1307,), (0.3081,)
# # train_transforms = transf.Compose([transf.Resize((32, 32)), transf.ToTensor(), transf.Normalize(*norm_stats)])
# # eval_transforms = transf.Compose([transf.Resize((32, 32)), transf.ToTensor(), transf.Normalize(*norm_stats)])
# # # benchmark = SplitCIFAR100(n_experiences=20, train_transform=train_transforms, eval_transform=eval_transforms, seed=42, return_task_id=True)
# # benchmark = PermutedMNIST(n_experiences=20, train_transform=train_transforms, eval_transform=eval_transforms, seed=42,)  # return_task_id=True)
# # train_stream = benchmark.train_stream
# # test_stream = benchmark.test_stream

# # # model = torchvision.models.resnet18(num_classes=5)
# # model = nn.Sequential(
# #     nn.Conv2d(1, 32, 3, bias=False),
# #     nn.ReLU(inplace=True),
# #     nn.Conv2d(32, 64, 3, bias=False),
# #     nn.ReLU(inplace=True),
# #     nn.Conv2d(64, 128, 3, bias=False),
# #     nn.ReLU(),
# #     nn.Flatten(),
# #     nn.Linear(26*26*128, 10)
# # )
# # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0, momentum=0.8)
# # criterion = nn.CrossEntropyLoss()
# # strategy = AGEM(model, optimizer, criterion, patterns_per_exp=50, sample_size=1300,
# #                 train_mb_size=10, eval_mb_size=10, device='cuda', train_epochs=1)

# # results = []
# # for train_task in train_stream:
# #     strategy.train(train_task, num_workers=2)
# #     results.append(strategy.eval(test_stream))

# import avalanche as avl


# class MLP(nn.Module):
#     def __init__(self, input_size=28 * 28, hidden_size=256, hidden_layers=2,
#                  output_size=10, drop_rate=0, relu_act=True):
#         super().__init__()
#         self._input_size = input_size

#         layers = nn.Sequential(*(nn.Linear(input_size, hidden_size),
#                                  nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
#                                  nn.Dropout(p=drop_rate)))
#         for layer_idx in range(hidden_layers - 1):
#             layers.add_module(
#                 f"fc{layer_idx + 1}", nn.Sequential(
#                     *(nn.Linear(hidden_size, hidden_size),
#                       nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
#                       nn.Dropout(p=drop_rate))))

#         self.features = nn.Sequential(*layers)
#         self.classifier = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = x.contiguous()
#         x = x.view(x.size(0), self._input_size)
#         x = self.features(x)
#         x = self.classifier(x)
#         return x


# def get_average_metric(metric_dict: dict, metric_name: str = 'Top1_Acc_Stream'):
#     """
#     Compute the average of a metric based on the provided metric name.
#     The average is computed across the instance of the metrics containing the
#     given metric name in the input dictionary.
#     :param metric_dict: dictionary containing metric name as keys and metric value as value.
#         This dictionary is usually returned by the `eval` method of Avalanche strategies.
#     :param metric_name: the metric name (or a part of it), to be used as pattern to filter the dictionary
#     :return: a number representing the average of all the metric containing `metric_name` in their name
#     """

#     avg_stream_acc = []
#     for k, v in metric_dict.items():
#         if k.startswith(metric_name):
#             avg_stream_acc.append(v)
#     return sum(avg_stream_acc) / float(len(avg_stream_acc))


# # args = create_default_args({'cuda': 0, 'patterns_per_exp': 250, 'hidden_size': 256,
# #                             'hidden_layers': 2, 'epochs': 1, 'dropout': 0,
# #                             'sample_size': 256,
# #                             'learning_rate': 0.1, 'train_mb_size': 10}, override_args)
# class args:
#     patterns_per_exp = 250
#     hidden_size = 256
#     hidden_layers = 2
#     epochs = 1
#     dropout = 0
#     sample_size = 256
#     learning_rate = 0.1
#     train_mb_size = 10


# device = torch.device("cuda")

# benchmark = avl.benchmarks.PermutedMNIST(17)
# model = MLP(hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
#             drop_rate=args.dropout)
# criterion = nn.CrossEntropyLoss()

# interactive_logger = avl.logging.InteractiveLogger()

# evaluation_plugin = avl.training.plugins.EvaluationPlugin(
#     avl.evaluation.metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
#     loggers=[interactive_logger], benchmark=benchmark)

# cl_strategy = avl.training.AGEM(
#     model, optim.SGD(model.parameters(), lr=args.learning_rate), criterion,
#     patterns_per_exp=args.patterns_per_exp, sample_size=args.sample_size,
#     train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
#     device=device, evaluator=evaluation_plugin)

# for experience in benchmark.train_stream:
#     cl_strategy.train(experience)
#     res = cl_strategy.eval(benchmark.test_stream)

# avg_stream_acc = get_average_metric(res)
# print(f"AGEM-PMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

# # target_acc = float(get_target_result('agem', 'pmnist'))
# # if args.check:
# #     self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.02)

import torch
import torch.utils.data
import torchvision
import itertools


def main():
    buffer = []
    dataset = torchvision.datasets.MNIST('./data/datasets', train=True, transform=torchvision.transforms.ToTensor())
    dataloder = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=0)
    dataloder_iterator = itertools.cycle(iter(dataloder))
    buffer.append(dataloder_iterator)

    minibatch = [next(d) for d in buffer]
    print(minibatch)


if __name__ == '__main__':
    main()

from avalanche.models import MultiHeadClassifier, MultiTaskModule
import torch
import torch.nn as nn
from torch import relu
import torch.optim as optim
import torchvision.models

from torch.nn.functional import avg_pool2d
from torchvision.models import resnet18
from avalanche.training import AGEM
from avalanche.benchmarks.classic import SplitCIFAR100, PermutedMNIST
from torchvision import transforms as transf


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

import avalanche as avl


class MLP(nn.Module):
    def __init__(self, input_size=28 * 28, hidden_size=256, hidden_layers=2,
                 output_size=10, drop_rate=0, relu_act=True):
        super().__init__()
        self._input_size = input_size

        layers = nn.Sequential(*(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                                 nn.Dropout(p=drop_rate)))
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    *(nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                      nn.Dropout(p=drop_rate))))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.InstanceNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf

        self.conv1 = conv3x3(1, nf * 1)
        self.bn1 = nn.InstanceNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 1, 28, 28))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        return out


"""
END: FROM GEM CODE
"""


class MultiHeadReducedResNet18(MultiTaskModule):
    """
    As from GEM paper, a smaller version of ResNet18, with three times less feature maps across all layers.
    It employs multi-head output layer.
    """

    def __init__(self, size_before_classifier=160):
        super().__init__()
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], 20)
        self.classifier = MultiHeadClassifier(size_before_classifier)

    def forward(self, x, task_labels):
        out = self.resnet(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out, task_labels)


def get_average_metric(metric_dict: dict, metric_name: str = 'Top1_Acc_Stream'):
    """
    Compute the average of a metric based on the provided metric name.
    The average is computed across the instance of the metrics containing the
    given metric name in the input dictionary.
    :param metric_dict: dictionary containing metric name as keys and metric value as value.
        This dictionary is usually returned by the `eval` method of Avalanche strategies.
    :param metric_name: the metric name (or a part of it), to be used as pattern to filter the dictionary
    :return: a number representing the average of all the metric containing `metric_name` in their name
    """

    avg_stream_acc = []
    for k, v in metric_dict.items():
        if k.startswith(metric_name):
            avg_stream_acc.append(v)
    return sum(avg_stream_acc) / float(len(avg_stream_acc))


# args = create_default_args({'cuda': 0, 'patterns_per_exp': 250, 'hidden_size': 256,
#                             'hidden_layers': 2, 'epochs': 1, 'dropout': 0,
#                             'sample_size': 256,
#                             'learning_rate': 0.1, 'train_mb_size': 10}, override_args)
class args:
    patterns_per_exp = 250
    hidden_size = 256
    hidden_layers = 2
    epochs = 1
    dropout = 0
    sample_size = 256
    learning_rate = 0.1
    train_mb_size = 10


device = torch.device("cuda")

benchmark = avl.benchmarks.PermutedMNIST(17)
# model = MLP(hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
#             drop_rate=args.dropout)
model = MultiHeadReducedResNet18()
criterion = nn.CrossEntropyLoss()

interactive_logger = avl.logging.InteractiveLogger()

evaluation_plugin = avl.training.plugins.EvaluationPlugin(
    avl.evaluation.metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
    loggers=[interactive_logger], benchmark=benchmark)

cl_strategy = avl.training.AGEM(
    model, optim.SGD(model.parameters(), lr=args.learning_rate), criterion,
    patterns_per_exp=args.patterns_per_exp, sample_size=args.sample_size,
    train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
    device=device, evaluator=evaluation_plugin)

for i, experience in enumerate(benchmark.train_stream):
    cl_strategy.train(experience)
    selected_tasks = [benchmark.test_stream[j] for j in range(0, i+1)]
    res = cl_strategy.eval(selected_tasks)

avg_stream_acc = get_average_metric(res)
print(f"AGEM-PMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")

# target_acc = float(get_target_result('agem', 'pmnist'))
# if args.check:
#     self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.02)


# if __name__ == '__main__':
#     main()

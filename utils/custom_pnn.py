# this file use newer version of pnn in avalanche and is based on:
# https://github.com/ContinualAI/avalanche/blob/master/avalanche/training/supervised/strategy_wrappers.py
# https://github.com/ContinualAI/avalanche/blob/e5562e3f3e1aaf7e9623698f7fd493ff97b4bf64/avalanche/models/pnn.py

from avalanche.training.strategies.base_strategy import BaseStrategy
from avalanche.training.plugins import StrategyPlugin
from typing import Optional, Sequence, List, Union
from avalanche.training.plugins.evaluation import default_logger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence
from avalanche.models import MultiTaskModule, DynamicModule
from avalanche.models import MultiHeadClassifier


class LinearAdapter(nn.Module):
    """
    Linear adapter for Progressive Neural Networks.
    """

    def __init__(self, in_features, out_features_per_column, num_prev_modules):
        """
        :param in_features: size of each input sample
        :param out_features_per_column: size of each output sample
        :param num_prev_modules: number of previous modules
        """
        super().__init__()
        # Eq. 1 - lateral connections
        # one layer for each previous column. Empty for the first task.
        self.lat_layers = nn.ModuleList([])
        for _ in range(num_prev_modules):
            m = nn.Linear(in_features, out_features_per_column)
            self.lat_layers.append(m)

    def forward(self, x):
        assert len(x) == self.num_prev_modules
        hs = []
        for ii, lat in enumerate(self.lat_layers):
            hs.append(lat(x[ii]))
        return sum(hs)


class MLPAdapter(nn.Module):
    """
    MLP adapter for Progressive Neural Networks.
    """

    def __init__(
        self,
        in_features,
        out_features_per_column,
        num_prev_modules,
        activation=F.relu,
    ):
        """
        :param in_features: size of each input sample
        :param out_features_per_column: size of each output sample
        :param num_prev_modules: number of previous modules
        :param activation: activation function (default=ReLU)
        """
        super().__init__()
        self.num_prev_modules = num_prev_modules
        self.activation = activation

        if num_prev_modules == 0:
            return  # first adapter is empty

        # Eq. 2 - MLP adapter. Not needed for the first task.
        self.V = nn.Linear(in_features * num_prev_modules, out_features_per_column)
        self.alphas = nn.Parameter(torch.randn(num_prev_modules))
        self.U = nn.Linear(out_features_per_column, out_features_per_column)

    def forward(self, x):
        if self.num_prev_modules == 0:
            return 0  # first adapter is empty

        assert len(x) == self.num_prev_modules
        assert len(x[0].shape) == 2, (
            "Inputs to MLPAdapter should have two dimensions: "
            "<batch_size, num_features>."
        )
        for i, el in enumerate(x):
            x[i] = self.alphas[i] * el
        x = torch.cat(x, dim=1)
        x = self.U(self.activation(self.V(x)))
        return x


class ConvAdapter(nn.Module):
    """
    Conv adapter for Progressive Neural Networks.
    Dimensionality reduction is performed via 1x1 convolutions.
    """

    def __init__(
        self,
        in_features,
        out_features_per_column,
        num_prev_modules,
        activation=F.relu,
    ):
        """
        :param in_features: size of each input sample
        :param out_features_per_column: size of each output sample
        :param num_prev_modules: number of previous modules
        :param activation: activation function (default=ReLU)
        """
        super().__init__()
        self.num_prev_modules = num_prev_modules
        self.activation = activation

        if num_prev_modules == 0:
            return  # first adapter is empty

        # Eq. 2 - MLP adapter. Not needed for the first task.
        self.V = nn.Conv2d(in_features * num_prev_modules, out_features_per_column, 1)
        self.alphas = nn.Parameter(torch.randn(num_prev_modules))
        self.U = nn.Conv2d(out_features_per_column, out_features_per_column, 3, padding=1)
        # self.U = ResBlock(out_features_per_column, out_features_per_column)

    def forward(self, x):
        if self.num_prev_modules == 0:
            return 0  # first adapter is empty

        assert len(x) == self.num_prev_modules
        assert len(x[0].shape) == 4, (
            "Inputs to MLPAdapter should have four dimensions: "
            "<batch_size, num_channels, h, w>."
        )
        for i, el in enumerate(x):
            x[i] = self.alphas[i] * el
        x = torch.cat(x, dim=1)
        x = self.U(self.activation(self.V(x)))
        return x


class ResBlock(nn.Module):
    def __init__(self, in_features, out_features, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1, stride=stride)  # TODO add stride = 2
        self.bn1 = nn.InstanceNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.InstanceNorm2d(out_features)
        self.downsample = None
        if stride != 1 or in_features != out_features:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride),
                nn.InstanceNorm2d(out_features),
            )

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        x += identity
        return x


class PNNColumn(nn.Module):
    """
    Progressive Neural Network column.
    """

    def __init__(
        self,
        in_features,
        out_features_per_column,
        num_prev_modules,
        layers_type="conv",
        adapter="mlp",
    ):
        """
        :param in_features: size of each input sample
        :param out_features_per_column:
            size of each output sample (single column)
        :param num_prev_modules: number of previous columns
        :param adapter: adapter type. One of {'linear', 'mlp'} (default='mlp')
        """
        super().__init__()
        self.in_features = in_features
        self.out_features_per_column = out_features_per_column
        self.num_prev_modules = num_prev_modules

        if layers_type == 'conv':
            self.itoh = nn.Conv2d(in_features, out_features_per_column, kernel_size=3, padding=1)
        elif layers_type == 'res_block':
            self.itoh = ResBlock(in_features, out_features_per_column)
        else:
            self.itoh = nn.Linear(in_features, out_features_per_column)

        if layers_type in ('conv', 'res_block'):
            self.adapter = ConvAdapter(in_features, out_features_per_column, num_prev_modules)
        elif adapter == "linear":
            self.adapter = LinearAdapter(in_features, out_features_per_column, num_prev_modules)
        elif adapter == "mlp":
            self.adapter = MLPAdapter(in_features, out_features_per_column, num_prev_modules)
        else:
            raise ValueError("`adapter` must be one of: {'mlp', `linear', 'conv'}.")

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
            param.grad = None

    def forward(self, x):
        prev_xs, last_x = x[:-1], x[-1]
        hs = self.adapter(prev_xs)
        hs += self.itoh(last_x)
        return hs


class PNNLayer(MultiTaskModule):
    """Progressive Neural Network layer.

    The adaptation phase assumes that each experience is a separate task.
    Multiple experiences with the same task label or multiple task labels
    within the same experience will result in a runtime error.
    """

    def __init__(self, in_features, out_features_per_column, adapter="mlp", layers_type="res_block"):
        """
        :param in_features: size of each input sample
        :param out_features_per_column:
            size of each output sample (single column)
        :param adapter: adapter type. One of {'linear', 'mlp'} (default='mlp')
        """
        super().__init__()
        self.in_features = in_features
        self.out_features_per_column = out_features_per_column
        self.layers_type = layers_type
        self.adapter = adapter

        # convert from task label to module list order
        self.task_to_module_idx = {}
        first_col = PNNColumn(in_features, out_features_per_column, 0, adapter=adapter, layers_type=layers_type)
        self.columns = nn.ModuleList([first_col])

    @property
    def num_columns(self):
        return len(self.columns)

    def adaptation(self, dataset: AvalancheDataset):
        """Training adaptation for PNN layer.

        Adds an additional column to the layer.

        :param dataset:
        :return:
        """
        super().train_adaptation(dataset)
        task_labels = dataset.targets_task_labels
        if isinstance(task_labels, ConstantSequence):
            # task label is unique. Don't check duplicates.
            task_labels = [task_labels[0]]
        else:
            task_labels = set(task_labels)
        assert len(task_labels) == 1, (
            "PNN assumes a single task for each experience. Please use a "
            "compatible benchmark."
        )
        # extract task label from set
        task_label = next(iter(task_labels))
        if task_label in self.task_to_module_idx:
            return  # we already added the column for the current task.

        if len(self.task_to_module_idx) == 0:
            # we have already initialized the first column.
            # No need to call add_column here.
            self.task_to_module_idx[task_label] = 0
        else:
            self.task_to_module_idx[task_label] = self.num_columns
            self._add_column()

    def _add_column(self):
        """Add a new column."""
        # Freeze old parameters
        for param in self.parameters():
            param.requires_grad = False
            param.grad = None
        self.columns.append(
            PNNColumn(
                self.in_features,
                self.out_features_per_column,
                self.num_columns,
                adapter=self.adapter,
                layers_type=self.layers_type,
            )
        )

    def forward_single_task(self, x, task_label):
        """Forward.

        :param x: list of inputs.
        :param task_label:
        :return:
        """
        col_idx = self.task_to_module_idx[task_label]
        hs = []
        for ii in range(col_idx + 1):
            el = self.columns[ii](x[: ii + 1])
            el = F.relu(el)
            hs.append(el)
        return hs


class PNN(MultiTaskModule):
    """
    Progressive Neural Network.

    The model assumes that each experience is a separate task.
    Multiple experiences with the same task label or multiple task labels
    within the same experience will result in a runtime error.
    """

    def __init__(
        self,
        num_layers=1,
        in_features=784,
        hidden_features_per_column=100,
        layers_type="conv",
        classifier_in_size=None,
        adapter="mlp",
    ):
        """
        :param num_layers: number of layers (default=1)
        :param in_features: size of each input sample
        :param hidden_features_per_column: number of hidden units for each column
        :param layers_type: layer type. One of {'conv', 'linear', 'res_block'} (default='res_block')
        :param adapter: adapter type. One of {'linear', 'mlp'} (default='mlp')
        """
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        self.in_features = in_features
        self.out_features_per_columns = hidden_features_per_column
        self.layers_type = layers_type

        self.layers = nn.ModuleList()
        first_layer_type = 'conv' if layers_type == 'res_block' else layers_type
        self.layers.append(PNNLayer(in_features, hidden_features_per_column, layers_type=first_layer_type))
        for _ in range(num_layers - 1):
            layer = PNNLayer(
                hidden_features_per_column,
                hidden_features_per_column,
                layers_type=layers_type,
                adapter=adapter,
            )
            self.layers.append(layer)
        self.flatten = nn.Flatten()
        classifier_in_size = hidden_features_per_column if layers_type == "linear" else classifier_in_size
        self.classifier = MultiHeadClassifier(classifier_in_size)

    def adaptation(self, dataset: AvalancheDataset):
        super().train_adaptation(dataset)
        for param in self.classifier.parameters():
            param.requires_grad = False
            param.grad = None

    def forward_single_task(self, x, task_label):
        """Forward.

        :param x:
        :param task_label:
        :return:
        """
        x = x.contiguous()
        if self.layers_type == 'linear':
            x = x.view(x.size(0), self.in_features)

        num_columns = self.layers[0].num_columns
        col_idx = self.layers[-1].task_to_module_idx[task_label]

        x = [x for _ in range(num_columns)]
        for layer in self.layers:
            x = layer(x, task_label)
        x = x[col_idx]
        x = self.flatten(x)
        y = self.classifier(x, task_label)
        return y


class PNNPlugin(StrategyPlugin):
    """Progressive Neural Network strategy.
    To use this strategy you need to instantiate a PNN model.
    """

    def __init__(self, lr, weight_decay):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay

    def before_training_exp(self, strategy, **kwargs):
        strategy.optimizer = optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, strategy.model.parameters())}],
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=False
        )


class PNNModified(BaseStrategy):
    """ Progressive Neural Networks strategy.

    See PNN plugin for details.
    """

    def __init__(self, model,
                 optimizer,
                 criterion,
                 lr,
                 weight_decay,
                 train_mb_size: int = 1,
                 train_epochs: int = 1,
                 eval_mb_size: int = 1,
                 device="cpu",
                 plugins=None,
                 evaluator=default_logger,
                 eval_every=-1,
                 **base_kwargs
                 ):
        """Init.
        :param model: PyTorch model.
        :param optimizer: PyTorch optimizer.
        :param criterion: loss function.
        :param train_mb_size: mini-batch size for training.
        :param train_epochs: number of training epochs.
        :param eval_mb_size: mini-batch size for eval.
        :param device: PyTorch device where the model will be allocated.
        :param plugins: (optional) list of StrategyPlugins.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations. None to remove logging.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param **base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """
        assert isinstance(model, PNN), "PNNStrategy requires a PNN model."

        pnn_plugin = PNNPlugin(lr, weight_decay)
        if plugins is None:
            plugins = [pnn_plugin]
        else:
            plugins.append(pnn_plugin)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size, train_epochs=train_epochs,
            eval_mb_size=eval_mb_size, device=device, plugins=plugins,
            evaluator=evaluator, eval_every=eval_every)

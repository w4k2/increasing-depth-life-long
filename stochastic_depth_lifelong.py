from os import stat
import torch.nn as nn
from torch.nn.modules.instancenorm import InstanceNorm2d
import torch.utils.model_zoo as model_zoo
import torch

import avalanche.models

__all__ = ['ResNet_StoDepth', 'resnet18_StoDepth_lineardecay', 'resnet34_StoDepth_lineardecay', 'resnet50_StoDepth_lineardecay', 'resnet101_StoDepth_lineardecay',
           'resnet152_StoDepth_lineardecay']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class StoDepth_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, prob, inplanes, planes, stride=1, downsample=None):
        super(StoDepth_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))

    def forward(self, x):
        # print('StoDepth_BasicBlock forward call')
        identity = x.clone()

        if self.training:
            if torch.equal(self.m.sample(), torch.ones(1)):

                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
            else:
                # Resnet does not use bias terms
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False

                if self.downsample is not None:
                    identity = self.downsample(x)

                out = identity
        else:

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.prob*out + identity

        out = self.relu(out)

        return out


class StoDepth_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, prob, inplanes, planes, stride=1, downsample=None):
        super(StoDepth_Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.InstanceNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))

    def forward(self, x):
        # print('StoDepth_Bottleneck forward call')
        identity = x.clone()

        if self.training:
            if torch.equal(self.m.sample(), torch.ones(1)):
                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True
                self.conv3.weight.requires_grad = True

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)

                out = self.conv3(out)
                out = self.bn3(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
            else:
                # Resnet does not use bias terms
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False
                self.conv3.weight.requires_grad = False

                if self.downsample is not None:
                    identity = self.downsample(x)

                out = identity
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = self.prob*out + identity

        out = self.relu(out)

        return out


def make_layer(inplanes, block, planes, blocks, stride=1, use_downsample=True):
    layers = []
    if use_downsample:
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                nn.InstanceNorm2d(planes * block.expansion),
            )
        layers.append(block(1.0, inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(1.0, inplanes, planes))

    return nn.Sequential(*layers), inplanes


class Node(nn.Module):
    def __init__(self, task_inplanes, block, layers, num_classes):
        super().__init__()
        self.task_inplanes = task_inplanes
        self.block = block
        self.layers = layers
        self.num_classes = num_classes
        self.layer3, inplanes = make_layer(task_inplanes, block, 256, layers[2], use_downsample=False)
        self.layer4, _ = make_layer(inplanes, block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.all_children = []
        self.current_child = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        feature_maps = self.layer3(x)
        if self.current_child:
            x, _ = self.current_child(feature_maps)
        else:
            x = self.layer4(feature_maps)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        return x, feature_maps

    def add_new_leaf(self, path, num_classes):
        if self.current_child == None or len(path) == 0:
            node = Node(self.task_inplanes, self.block, self.layers, num_classes)
            self.current_child = node
            self.all_children.append(node)
        else:
            self.current_child.add_new_leaf(path[1:], num_classes)

    def get_all_paths(self):
        all_paths = []
        for i, node in enumerate(self.all_children):
            all_paths.append([i])
            children_paths = node.get_all_paths()
            for path in children_paths:
                all_paths.append([i] + path)
        return all_paths

    def set_path(self, path):
        if len(path) > 0:
            self.current_child = self.all_children[path[0]]
            self.current_child.set_path(path[1:])
        else:
            self.current_child = None

    def get_current_path(self):
        current_path = []
        for i, node in enumerate(self.all_children):
            if self.current_child == node:
                current_path.append(i)
        if self.current_child:
            current_path.extend(self.current_child.get_current_path())
        return current_path

    def current_depth(self):
        depth = len(self.layer3) + len(self.layer4)
        if self.current_child:
            depth += self.current_child.current_depth()
        return depth

    def update_probs(self, prob_now, prob_step):
        for block in self.layer3:
            block.prob = prob_now
            prob_now = prob_now - prob_step

        for block in self.layer4:
            block.prob = prob_now
            prob_now = prob_now - prob_step

        if self.current_child:
            self.current_child.update_probs(prob_now, prob_step)


class ResNet_StoDepth(nn.Module):

    def __init__(self, block, prob_begin, prob_end, layers, num_classes=1000, input_channels=3, zero_init_residual=False):
        super().__init__()
        inplanes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.prob_begin = prob_begin
        self.prob_end = prob_end

        self.layer1, inplanes = make_layer(inplanes, block, 64, layers[0])
        self.layer2, inplanes = make_layer(inplanes, block, 128, layers[1], stride=2)
        downsample = nn.Sequential(
            conv1x1(inplanes, 256 * block.expansion, stride=2),
            nn.InstanceNorm2d(256 * block.expansion),
        )
        self.downsample_block = block(1.0, inplanes, planes=256, stride=2, downsample=downsample)
        self.task_inplanes = inplanes

        self.block = block
        self.layers = layers

        self.nodes = nn.ModuleList([])
        self.current_node = None
        self.add_new_node([], freeze_previous=False, num_classes=num_classes)

        self.tasks_paths = dict()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, StoDepth_Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, StoDepth_BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def update_structure(self, task_id, dataloader, num_classes, device, entropy_threshold):
        current_path = [0]
        if task_id > 0:
            path = self.select_most_similar_task(dataloader, num_classes=num_classes, device=device, threshold=entropy_threshold)
            print('min entropy path = ', path)
            self.add_new_node(path, num_classes)
            self.to(device)
            current_path = self.get_current_path()

        self.tasks_paths[task_id] = current_path

    def add_new_node(self, path, num_classes, freeze_previous=True):
        if freeze_previous:
            self.freeze_current_path()

        if len(path) == 0:
            node = Node(self.task_inplanes, self.block, self.layers, num_classes)
            self.current_node = node
            self.nodes.append(node)
        else:
            self.set_path(path)
            if freeze_previous:
                self.freeze_current_path()
            self.current_node.add_new_leaf(path[1:], num_classes)
        self.update_probs()

    def freeze_current_path(self):
        for param in self.parameters():
            param.requires_grad = False
            param.grad = None

    def select_most_similar_task(self, dataloder, num_classes, device='cuda', threshold=0.5):
        all_paths = self.get_all_paths()
        min_entropy_raio = 1.0
        min_entropy_path = []
        max_entropy = self.entropy(torch.Tensor([[1.0/num_classes for _ in range(num_classes)]]))

        for path in all_paths:
            self.set_path(path)
            self.to(device)
            self.eval()
            avrg_entropy = []
            with torch.no_grad():
                for inp, _, _ in dataloder:
                    inp = inp.to(device)
                    y_pred = self.forward(inp)
                    y_pred = torch.softmax(y_pred, dim=1)
                    entropy = self.entropy(y_pred)
                    avrg_entropy.append(entropy)
                    # break
            avrg_entropy = torch.mean(torch.cat(avrg_entropy)).item()
            entropy_ratio = avrg_entropy / max_entropy
            print(f'path = {path}, entropy_ratio = {entropy_ratio}')

            if entropy_ratio <= min_entropy_raio:
                min_entropy_raio = entropy_ratio
                min_entropy_path = path

        print('min entropy ratio = ', min_entropy_raio)
        if min_entropy_raio >= threshold:
            min_entropy_path = []
        return min_entropy_path

    def get_all_paths(self):
        all_paths = []
        for i, node in enumerate(self.nodes):
            all_paths.append([i])
            node_paths = node.get_all_paths()
            for path in node_paths:
                all_paths.append([i] + path)
        return all_paths

    @staticmethod
    def entropy(p):
        log_p = torch.log2(p)
        entropy = - torch.sum(log_p * p, dim=1)
        return entropy

    def set_path(self, path):
        self.current_node = self.nodes[path[0]]
        self.current_node.set_path(path[1:])
        self.update_probs()

    def update_probs(self):
        depth = self.current_depth()
        prob_now = self.prob_begin
        prob_delta = self.prob_begin - self.prob_end
        prob_step = prob_delta / (depth - 1)

        for block in self.layer1:
            block.prob = prob_now
            prob_now = prob_now - prob_step

        for block in self.layer2:
            block.prob = prob_now
            prob_now = prob_now - prob_step

        if self.current_node:
            self.current_node.update_probs(prob_now, prob_step)

    def current_depth(self):
        depth = len(self.layer1) + len(self.layer2)
        if self.current_node:
            depth += self.current_node.current_depth()
        return depth

    def get_current_path(self):
        current_path = []
        for i, node in enumerate(self.nodes):
            if self.current_node == node:
                current_path.append(i)
        if len(current_path) == 0:
            raise ValueError("Current node not found")
        current_path.extend(self.current_node.get_current_path())
        return current_path

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.downsample_block(x)

        pred, _ = self.current_node(x)

        return pred


def resnet18_StoDepth_lineardecay(pretrained=False, prob_begin=1, prob_end=0.5, **kwargs):
    """Constructs a ResNet_StoDepth-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth(StoDepth_BasicBlock, prob_begin, prob_end, [2, 2, 2, 2], **kwargs)
    if pretrained:
        import torchvision
        model_tmp = torchvision.models.resnet18(pretrained=True)
        state_dict = model_tmp.state_dict()
        state_dict = {key: value for key, value in state_dict.items() if key in model.state_dict() and type(model._modules[key.rsplit('.')[0]]) != InstanceNorm2d}
        state_dict["downsample_block.conv1.weight"] = model_tmp.state_dict()['layer3.0.conv1.weight']
        state_dict["downsample_block.conv2.weight"] = model_tmp.state_dict()['layer3.0.conv2.weight']
        state_dict["downsample_block.downsample.0.weight"] = model_tmp.state_dict()['layer3.0.downsample.0.weight']
        model.load_state_dict(state_dict, strict=False)
        freeze_module(model.conv1)
        freeze_module(model.layer1)
        freeze_module(model.layer2)
        freeze_module(model.downsample_block)
    return model


def freeze_module(m):
    for param in m.parameters():
        param.requires_grad = False
        param.grad = None


def resnet9_StoDepth_lineardecay(pretrained=False, prob_begin=1, prob_end=0.5, **kwargs):
    """Constructs a ResNet_StoDepth-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth(StoDepth_BasicBlock, prob_begin, prob_end, [1, 1, 1, 1], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34_StoDepth_lineardecay(pretrained=False, prob_begin=1, prob_end=0.5, **kwargs):
    """Constructs a ResNet_StoDepth-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth(StoDepth_BasicBlock, prob_begin, prob_end, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50_StoDepth_lineardecay(pretrained=False, prob_begin=1, prob_end=0.5, **kwargs):
    """Constructs a ResNet_StoDepth-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth(StoDepth_Bottleneck, prob_begin, prob_end, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101_StoDepth_lineardecay(pretrained=False, prob_begin=1, prob_end=0.5, **kwargs):
    """Constructs a ResNet_StoDepth-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth(StoDepth_Bottleneck, prob_begin, prob_end, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152_StoDepth_lineardecay(pretrained=False, prob_begin=1, prob_end=0.5, **kwargs):
    """Constructs a ResNet_StoDepth-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth(StoDepth_Bottleneck, prob_begin, prob_end, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


if __name__ == '__main__':
    def model_paramters(model: torch.nn.Module):
        unique = {p.data_ptr(): p for p in model.parameters()}.values()
        return sum(p.numel() for p in unique)

    model = resnet18_StoDepth_lineardecay(num_classes=10, pretrained=True)
    # print(model_paramters(model))

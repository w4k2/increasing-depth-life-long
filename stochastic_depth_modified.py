import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch

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

    def __init__(self, prob, multFlag, inplanes, planes, stride=1, downsample=None):
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
        self.multFlag = multFlag

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

            if self.multFlag:
                out = self.prob*out + identity
            else:
                out = out + identity

        out = self.relu(out)

        return out


class StoDepth_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, prob, multFlag, inplanes, planes, stride=1, downsample=None):
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
        self.multFlag = multFlag

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

            if self.multFlag:
                out = self.prob*out + identity
            else:
                out = out + identity

        out = self.relu(out)

        return out


def make_layer(inplanes, multFlag, prob_now, prob_step, block, planes, blocks, stride=1, use_downsample=True):
    layers = []
    if use_downsample:
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                nn.InstanceNorm2d(planes * block.expansion),
            )
        layers.append(block(prob_now, multFlag, inplanes, planes, stride, downsample))
    prob_now = prob_now - prob_step
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(prob_now, multFlag, inplanes, planes))
        prob_now = prob_now - prob_step

    return nn.Sequential(*layers), inplanes, prob_now


class Node(nn.Module):
    def __init__(self, task_inplanes, multFlag, task_base_prob, prob_step, block, layers, num_classes):
        super().__init__()
        self.layer3, inplanes, prob_now = make_layer(task_inplanes, multFlag, task_base_prob, prob_step, block, 256, layers[2], use_downsample=False)
        self.layer4, _, _ = make_layer(inplanes, multFlag, prob_now, prob_step, block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.all_children = []
        self.current_child = None

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

    def add_new_leaf(self, node):
        if self.current_child == None:
            self.current_child = node
        else:
            self.current_child.add_new_leaf(node)

# class Tree(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.root = None


class ResNet_StoDepth(nn.Module):

    def __init__(self, block, prob_begin, prob_end, multFlag, layers, max_depth=5, num_classes=1000, zero_init_residual=False):
        super().__init__()
        inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.multFlag = multFlag
        prob_now = prob_begin
        self.prob_delta = prob_begin - prob_end
        self.prob_step = self.prob_delta/(sum(layers)-1)

        self.layer1, inplanes, prob_now = make_layer(inplanes, multFlag, prob_now, self.prob_step, block, 64, layers[0])
        self.layer2, inplanes, prob_now = make_layer(inplanes, multFlag, prob_now, self.prob_step, block, 128, layers[1], stride=2)
        downsample = nn.Sequential(
            conv1x1(inplanes, 256 * block.expansion, stride=2),
            nn.InstanceNorm2d(256 * block.expansion),
        )
        self.downsample_block = block(prob_now, self.multFlag, inplanes, planes=256, stride=2, downsample=downsample)
        self.task_base_prob = prob_now - self.prob_step
        self.task_inplanes = inplanes

        self.block = block
        self.layers = layers
        self.num_classes = num_classes
        self.current_node = None
        self.add_new_node(freeze_previous=False)

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

    # TODO check multFlag evaluation
    def add_new_node(self, freeze_previous=True):
        if freeze_previous:
            for param in self.parameters():
                param.requires_grad = False

        node = Node(self.task_inplanes, self.multFlag, self.task_base_prob, self.prob_step, self.block, self.layers, self.num_classes)
        if self.current_node == None:
            self.current_node = node
        else:
            self.current_node.add_new_leaf(node)

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


def resnet18_StoDepth_lineardecay(pretrained=False, prob_begin=1, prob_end=0.5, multFlag=True, **kwargs):
    """Constructs a ResNet_StoDepth-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth(StoDepth_BasicBlock, prob_begin, prob_end, multFlag, [4, 4, 4, 4], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34_StoDepth_lineardecay(pretrained=False, prob_begin=1, prob_end=0.5, multFlag=True, **kwargs):
    """Constructs a ResNet_StoDepth-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth(StoDepth_BasicBlock, prob_begin, prob_end, multFlag, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50_StoDepth_lineardecay(pretrained=False, prob_begin=1, prob_end=0.5, multFlag=True, **kwargs):
    """Constructs a ResNet_StoDepth-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth(StoDepth_Bottleneck, prob_begin, prob_end, multFlag, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101_StoDepth_lineardecay(pretrained=False, prob_begin=1, prob_end=0.5, multFlag=True, **kwargs):
    """Constructs a ResNet_StoDepth-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth(StoDepth_Bottleneck, prob_begin, prob_end, multFlag, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152_StoDepth_lineardecay(pretrained=False, prob_begin=1, prob_end=0.5, multFlag=True, **kwargs):
    """Constructs a ResNet_StoDepth-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_StoDepth(StoDepth_Bottleneck, prob_begin, prob_end, multFlag, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


if __name__ == '__main__':
    model = resnet18_StoDepth_lineardecay()
    model = model.to('cuda')
    inp = torch.randn(1, 3, 224, 224)
    inp = inp.to('cuda')

    output = model(inp)
    # print(output)

    def print_probs(layer):
        for i in range(len(layer)):
            print(layer[i].prob)

    for layer_name in ['layer1', 'layer2']:
        print(layer_name)
        layer = getattr(model, layer_name)
        print_probs(layer)
    print('downsample')
    print(model.downsample_block.prob)

    def print_node_probs(node):
        print('layer 3')
        print_probs(node.layer3)
        print('layer 4')
        print_probs(node.layer4)
        if node.current_child is not None:
            print_node_probs(node.current_child)

    print_node_probs(model.current_node)
    # print(model.state_dict())

    print('new task')
    model.add_new_node()
    print_node_probs(model.current_node)
    # print(model.state_dict())

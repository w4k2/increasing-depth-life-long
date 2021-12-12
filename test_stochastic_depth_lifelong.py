import pytest
import torch

from stochastic_depth_lifelong import *


def test_stoch_depth_cuda_inference():
    model = resnet18_StoDepth_lineardecay()
    model = model.to('cuda')
    inp = torch.randn(1, 3, 224, 224)
    inp = inp.to('cuda')

    output = model(inp)

    assert output.shape == (1, 1000)


def get_probs(layer):
    probs = []
    for i in range(len(layer)):
        p = layer[i].prob
        probs.append(p)
        print(p)
    return probs


def test_backbone_probs_are_correct():
    prob_begin = 1.0
    prob_end = 0.5
    model = resnet18_StoDepth_lineardecay(prob_begin=prob_begin, prob_end=prob_end)

    all_probs = []
    for layer_name in ['layer1', 'layer2']:
        print(layer_name)
        layer = getattr(model, layer_name)
        all_probs.extend(get_probs(layer))
    print('downsample')
    print(model.downsample_block.prob)

    assert all_probs[0] == 1
    assert all(p <= prob_begin and p >= prob_end for p in all_probs)


def get_node_probs(node):
    probs = list()
    print('layer 3')
    probs.extend(get_probs(node.layer3))
    print('layer 4')
    probs.extend(get_probs(node.layer4))
    if node.current_child is not None:
        probs.extend(get_node_probs(node.current_child))
    return probs


def test_node_probs_are_correct():
    prob_begin = 1.0
    prob_end = 0.5
    model = resnet18_StoDepth_lineardecay(prob_begin=prob_begin, prob_end=prob_end)

    probs = get_node_probs(model.current_node)
    assert all(p <= prob_begin and p >= prob_end for p in probs)
    # assert probs[-1] == prob_end


def test_node_probs_with_additional_task_are_correct():
    prob_begin = 1.0
    prob_end = 0.5
    model = resnet18_StoDepth_lineardecay(prob_begin=prob_begin, prob_end=prob_end)
    for _ in range(3):
        model.add_new_node(model.get_current_path(), num_classes=2)

    probs = get_node_probs(model.current_node)
    assert all(p <= prob_begin and p >= prob_end for p in probs)
    # assert probs[-1] == prob_end


def test_affter_adding_100_tasks_inference_is_possible():
    model = resnet18_StoDepth_lineardecay()
    model = model.to('cuda')
    for _ in range(100):
        model.add_new_node(model.get_current_path(), num_classes=2)
    # get_node_probs(model.current_node)

    model = model.to('cuda')
    inp = torch.randn(32, 3, 224, 224)
    inp = inp.to('cuda')
    output = model(inp)
    # print(output)

    assert output.shape == (32, 2)

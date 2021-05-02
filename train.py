import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as trans
import matplotlib.pyplot as plt
import tqdm
import skmultiflow

import stochastic_depth_model
import DataProvider


def main():
    dataloaders = DataProvider.get_tinyimagenet_dataloaders(batch_size=32)
    device = torch.device('cuda')
    crtierion = nn.CrossEntropyLoss()
    crtierion = crtierion.to(device)

    model = stochastic_depth_model.resnet18_StoDepth_lineardecay(num_classes=19)
    # model.layer3[1].prob = 0.0
    # model.layer3[1].m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([0.0]))
    # model = torchvision.models.resnet50(num_classes=3)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0000001)
    drift_detector = skmultiflow.drift_detection.DDM(out_control_level=1.5)

    batch_acc = []

    model.train()
    # deactivate_layers(model, 1)
    # activate_layers(model, 0)

    activate_frist_half = True

    dataloader_index = 0
    for dataloader in dataloaders:
        pbar = tqdm.tqdm(dataloader, total=len(dataloader))

        # for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        #     print(layer_name)
        #     layer = getattr(model, layer_name)
        #     for i in range(len(layer)):
        #         print(layer[i].prob)

        for i, (img, target) in enumerate(pbar):
            model.zero_grad()
            img = img.to(device)
            # target = torch.argmax(target, keepdim=False, dim=1)
            target = target.to(device)

            pred = model(img)
            acc = sum(torch.argmax(torch.softmax(pred, dim=1), keepdim=False, dim=1) == target) / target.shape[0]
            acc = acc.item()
            batch_acc.append(acc)

            loss = crtierion(pred, target)
            loss.backward()

            loss_value = loss.mean(dim=0).item()
            if i % 10 == 0:
                pbar.set_description(f'loss = {loss_value} acc = {acc}')
            drift_detector.add_element(loss_value)
            if drift_detector.detected_change():
                print(f'Change has been detected in batch: {i}')

                # if dataloader_index % 10 == 0:
                #     activate_frist_half = not activate_frist_half
                # if activate_frist_half:
                #     deactivate_layers(model, 1)
                #     activate_layers(model, 0)
                # else:
                #     deactivate_layers(model, 0)
                #     activate_layers(model, 1)

            optimizer.step()
        dataloader_index += 1

    plot_acc(batch_acc)


def deactivate_layers(model, index):
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for i in range(len(layer)):
            if i % 2 == index:
                layer[i].prob = 0.1
                layer[i].m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([0.1]))


def activate_layers(model, index):
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for i in range(len(layer)):
            if i % 2 == index:
                layer[i].prob = 0.9
                layer[i].m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([0.9]))


def plot_acc(batch_acc):
    plt.plot(batch_acc)
    plt.xlabel('batches')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':
    main()

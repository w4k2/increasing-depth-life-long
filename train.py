import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as trans
import matplotlib.pyplot as plt
import tqdm
import skmultiflow
import itertools

import stochastic_depth_model
import stochastic_depth_modified
import fhddm
import DataProvider


def main():
    batch_size = 32
    dataloaders = DataProvider.get_tinyimagenet_dataloaders(batch_size=batch_size)
    device = torch.device('cuda')
    crtierion = nn.CrossEntropyLoss()
    crtierion = crtierion.to(device)

    # model = stochastic_depth_model.resnet18_StoDepth_lineardecay(num_classes=19)
    model = stochastic_depth_modified.resnet101_StoDepth_lineardecay(num_classes=19)
    # model = torchvision.models.resnet18(num_classes=19)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-7)
    drift_detector = fhddm.FHDDM(n=5*batch_size)

    batch_acc = []
    batch_loss = []

    model.train()
    model.activate_task(0)

    tasks = itertools.cycle((0, 1, 2))

    dataloader_index = 0
    for dataloader in dataloaders:
        if dataloader_index % 10 == 0:
            task_index = next(tasks)
            model.activate_task(task_index)

        # for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        #     print(layer_name)
        #     layer = getattr(model, layer_name)
        #     for i in range(len(layer)):
        #         print(layer[i].prob)

        pbar = tqdm.tqdm(dataloader, total=len(dataloader))

        for i, (img, target) in enumerate(pbar):
            model.zero_grad()
            img = img.to(device)
            # target = torch.argmax(target, keepdim=False, dim=1)
            target = target.to(device)

            pred = model(img)

            predictions = torch.argmax(torch.softmax(pred, dim=1), keepdim=False, dim=1)
            correct_predictions = predictions == target
            for prediction in correct_predictions:
                _, drift_detected = drift_detector.run(prediction)
                if drift_detected:
                    print(f'Change has been detected in batch: {i}')

            acc = sum(torch.argmax(torch.softmax(pred, dim=1), keepdim=False, dim=1) == target) / target.shape[0]
            acc = acc.item()
            batch_acc.append(acc)

            loss = crtierion(pred, target)
            loss.backward()

            loss_value = loss.mean(dim=0).item()
            batch_loss.append(loss_value)
            if i % 10 == 0:
                pbar.set_description(f'loss = {loss_value} acc = {acc}')

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

    plot(batch_acc, batch_loss)


def plot(batch_acc, batch_loss):
    plt.figure(1)
    plt.plot(batch_acc)
    plt.xlabel('batches')
    plt.ylabel('accuracy')

    plt.figure(2)
    plt.plot(batch_loss, 'r')
    plt.xlabel('batches')
    plt.ylabel('loss value')

    plt.show()


if __name__ == '__main__':
    main()

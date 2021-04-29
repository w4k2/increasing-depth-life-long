import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as trans
import matplotlib.pyplot as plt
import tqdm

import stochastic_depth_model
import DataProvider


def main():
    dataloaders = DataProvider.get_tinyimagenet_dataloaders(batch_size=32)
    device = torch.device('cuda')
    crtierion = nn.CrossEntropyLoss()
    crtierion = crtierion.to(device)

    # model = stochastic_depth_model.resnet18_StoDepth_lineardecay(num_classes=2)
    # model.layer3[1].prob = 0.0
    # model.layer3[1].m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([0.0]))
    model = torchvision.models.resnet50(num_classes=3)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0000001)

    batch_acc = []

    model.train()
    for dataloader in dataloaders:
        pbar = tqdm.tqdm(dataloader, total=len(dataloader))
        for i, (img, target) in enumerate(pbar):
            model.zero_grad()
            img = img.to(device)
            # target = torch.argmax(target, keepdim=False, dim=1)
            target = target.to(device)

            pred = model(img)
            acc = sum(torch.argmax(torch.softmax(pred, dim=1), keepdim=False, dim=1) == target) / target.shape[0]
            acc = acc.item()
            if i % 100 == 0:
                pbar.set_description(f'acc = {acc}')
            batch_acc.append(acc)

            loss = crtierion(pred, target)
            loss.backward()

            optimizer.step()

    plot_acc(batch_acc)


def plot_acc(batch_acc):
    plt.plot(batch_acc)
    plt.xlabel('batches')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':
    main()

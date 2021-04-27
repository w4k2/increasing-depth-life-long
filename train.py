import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as trans

import stochastic_depth_model
import DataProvider


def main():
    dataloaders = DataProvider.get_celeba_dataloaders(batch_size=32)
    device = torch.device('cuda')
    crtierion = nn.CrossEntropyLoss()
    crtierion = crtierion.to(device)

    model = stochastic_depth_model.resnet18_StoDepth_lineardecay(num_classes=2)
    model.layer3[1].prob = 0.0
    model.layer3[1].m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([0.0]))
    # model = torchvision.models.resnet50(num_classes=10)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0000001)

    model.train()
    for dataloader in dataloaders:

        for img, target in dataloader:
            model.zero_grad()
            img = img.to(device)
            target = target.to(device)

            pred = model(img)
            acc = sum(torch.argmax(torch.softmax(pred, dim=1), keepdim=False, dim=1) == target) / target.shape[0]
            print(acc)

            loss = crtierion(pred, target)
            loss.backward()

            optimizer.step()

        print('\n\ndrift\n\n')


if __name__ == '__main__':
    main()

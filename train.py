import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as trans

import model

if __name__ == '__main__':
    transforms = trans.Compose([
        trans.Resize(224),
        trans.ToTensor(),
        trans.Normalize((0.4913997551666284, 0.48215855929893703, 0.4465309133731618), (0.24703225141799082, 0.24348516474564, 0.26158783926049628))
    ])
    train_dataset = torchvision.datasets.CIFAR10('.', train=True, download=True, transform=transforms)
    test_dataset = torchvision.datasets.CIFAR10('.', train=False, download=True, transform=transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=20)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=20)
    device = torch.device('cuda')
    crtierion = nn.CrossEntropyLoss()
    crtierion = crtierion.to(device)

    model = model.resnet50_StoDepth_lineardecay(num_classes=10)
    model.layer3[1].prob = 0.0
    model.layer3[1].m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([0.0]))
    # model = torchvision.models.resnet50(num_classes=10)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    epochs = 20

    for epoch in range(epochs):
        model.train()
        for img, target in train_dataloader:
            model.zero_grad()
            img = img.to(device)
            target = target.to(device)

            pred = model(img)

            loss = crtierion(pred, target)
            loss.backward()

            optimizer.step()

        model.eval()
        with torch.no_grad():
            corret = 0.0
            num_samples = 0
            for img, target in test_dataloader:
                img = img.to(device)
                target = target.to(device)
                pred = model(img)
                corret += sum(torch.argmax(torch.softmax(pred, dim=1), keepdim=False, dim=1) == target)
                num_samples += target.shape[0]
            acc = corret / num_samples
            print('acc = ', acc)

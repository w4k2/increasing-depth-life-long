import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import tqdm
import itertools
import argparse

import stochastic_depth_model
import stochastic_depth_modified
from cifar10 import get_dataloder
from collections import namedtuple
from sklearn.metrics import accuracy_score


def main():
    args = parse_args()

    device = torch.device('cuda')
    crtierion = nn.CrossEntropyLoss()
    crtierion = crtierion.to(device)
    weight_decay = 0.000001
    lr_milestones = (100,)
    lr = 0.0001

    # model = stochastic_depth_model.resnet18_StoDepth_lineardecay(num_classes=19)
    # model = stochastic_depth_modified.resnet101_StoDepth_lineardecay(num_classes=19)
    model = stochastic_depth_modified.resnet50_StoDepth_lineardecay(num_classes=3)
    # model = torchvision.models.resnet18(num_classes=19)

    tasks = [(0, 1, 2), (3, 4, 0), (5, 6, 0), (7, 8, 0), (9, 1, 0)]

    for i, task_classes in enumerate(tasks):
        train_dataloder = get_dataloder(args, task_classes, train=True, shuffle=True, flip=False)
        test_dataloader = get_dataloder(args, task_classes, train=False, shuffle=False, flip=False)
        if i > 0:
            path = model.select_most_similar_task(train_dataloder)
            print('min entropy path = ', path)
            model.add_new_node(path)

        model, results = train(model, train_dataloder, test_dataloader, lr=lr, n_epochs=args.n_epochs,
                               lr_milestones=lr_milestones, weight_decay=weight_decay, device=device, num_layers=args.num_layers)
        plot_metrics(results)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--n_epochs', default=20, type=int)
    parser.add_argument('--num_layers', default=1, type=int)

    args = parser.parse_args()
    return args


def train(model, train_dataloder, test_dataloader, lr: float = 0.001, n_epochs: int = 150,
          lr_milestones: tuple = (), weight_decay: float = 1e-6, device: str = 'cuda', num_layers=1):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

    Results = namedtuple('Results',
                         ['loss_values', 'train_acc_values', 'test_acc_values'])

    loss_values = []
    train_acc_values = []
    test_acc_values = []
    train_acc = None
    test_acc = None

    loss_fn = nn.CrossEntropyLoss()

    pbar = tqdm.tqdm(range(n_epochs))
    for epoch in pbar:
        model.train()
        for inputs, labels in train_dataloder:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            y_pred = model(inputs)
            loss = loss_fn(y_pred, labels)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            pbar_update = 'loss = {:.4f} '.format(loss_value)
            if train_acc is not None and test_acc is not None:
                pbar_update += 'train_acc = {:.4f} test_acc = {:.4f}'.format(train_acc, test_acc)
            pbar.set_description(pbar_update)

            break

        loss_values.append(loss_value)
        if test_dataloader is not None:
            y_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1).detach().cpu().numpy()
            labels = labels.cpu().numpy()
            train_acc = accuracy_score(labels, y_pred)
            train_acc_values.append(train_acc)
            test_acc = evaluate(model, test_dataloader, device=device)
            test_acc_values.append(test_acc)

        scheduler.step()

    results = Results(loss_values, train_acc_values, test_acc_values)
    return model, results


def evaluate(model, test_dataloder, device: str = 'cuda'):
    all_labels = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloder:
            inputs = inputs.to(device)
            y_pred = model(inputs)
            y_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            all_labels.append(labels)
            all_preds.append(y_pred)
            break

    all_labels = torch.concat(all_labels).cpu().numpy()
    all_preds = torch.concat(all_preds).cpu().numpy()
    acc = accuracy_score(all_labels, all_preds)
    return acc


def plot_metrics(results):
    _, axs = plt.subplots(2)
    axs[0].plot(results.loss_values, label='loss')
    axs[0].set(xlabel='epochs', ylabel='loss')
    axs[0].legend()
    axs[1].plot(results.train_acc_values, 'b-', label='train Acc')
    axs[1].plot(results.test_acc_values, 'b--', label='test Acc')
    axs[1].set(xlabel='epochs', ylabel='accuracy')
    axs[1].set_ylim(0, 1)
    axs[1].legend()


if __name__ == '__main__':
    main()

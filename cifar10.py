import torch
import numpy as np
import torchvision
import copy


def get_cifar10(train=True, use_horizontal_flip=False, target_transform=None):

    transforms_list = [
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ]
    if use_horizontal_flip:
        transforms_list.insert(0, torchvision.transforms.RandomHorizontalFlip(p=0.5))

    transforms = torchvision.transforms.Compose(transforms_list)

    cifar_dataset = torchvision.datasets.CIFAR10(root='../repo/data', download=True, train=train, transform=transforms, target_transform=target_transform)
    return cifar_dataset


def select_classes(dataset, class_idx):
    if type(class_idx) == int:
        class_idx = [class_idx]
    new_dataset = copy.deepcopy(dataset)
    selected_indexes = []
    for idx in class_idx:
        selected_indexes.append(np.argwhere(np.array(new_dataset.targets) == idx).flatten())
    class_indexes = np.concatenate(selected_indexes, axis=0)
    new_dataset.data = [new_dataset.data[i] for i in class_indexes]
    new_dataset.targets = [new_dataset.targets[i] for i in class_indexes]
    return new_dataset


def get_dataloder(args, task_classes, train=True, shuffle=False, flip=False):
    labels_mapping = {class_idx: i for i, class_idx in enumerate(task_classes)}
    dataset = get_cifar10(train=train, use_horizontal_flip=flip, target_transform=torchvision.transforms.Lambda(lambda l: labels_mapping[l]))
    dataset = select_classes(dataset, task_classes)
    dataloder = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size,
                                            shuffle=shuffle,
                                            num_workers=args.num_workers)

    return dataloder


if __name__ == '__main__':
    import torchvision
    import matplotlib.pyplot as plt

    base_dataset = torchvision.datasets.CIFAR10(root='./data', download=True, transform=torchvision.transforms.ToTensor())
    frogs_dataset = select_classes(base_dataset, 6)

    images = []
    for i in range(9):
        img, _ = frogs_dataset[i]
        images.append(img)

    grid = torchvision.utils.make_grid(images)
    grid_np = grid.numpy()
    plt.imshow(np.transpose(grid_np, (1, 2, 0)))
    plt.show()

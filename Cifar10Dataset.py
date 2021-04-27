from sklearn.utils import shuffle
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import torch


class Cifar10Dataset:

    def load_dataset(self):

        dataset = datasets.CIFAR10(transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.CenterCrop((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            download=True, root='./data/datasets/cifar10')

        mask = np.asarray(dataset.targets) > 4

        x1 = dataset.data[~mask]
        y1 = np.asarray(dataset.targets)[~mask]
        x2 = dataset.data[mask]
        y2 = np.asarray(dataset.targets)[mask]

        animal_indices = [2, 3, 4, 5, 6, 7]

        for i, (y1_elem, y2_elem) in enumerate(zip(y1, y2)):
            if y1_elem in animal_indices:
                y1[i] = 1
            else:
                y1[i] = 0

            if y2_elem in animal_indices:
                y2[i] = 1
            else:
                y2[i] = 0

        x1 = np.repeat(x1, 10, axis=0)
        y1 = np.repeat(y1, 10, axis=0)
        x2 = np.repeat(x2, 10, axis=0)
        y2 = np.repeat(y2, 10, axis=0)

        x1, y1 = shuffle(x1, y1)
        x2, y2 = shuffle(x2, y2)

        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))

        dataset.data = x
        dataset.targets = torch.squeeze(torch.nn.functional.one_hot(torch.from_numpy(y).to(torch.int64)))
        return dataset

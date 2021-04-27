from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import torch


class CelebADataset:

    def load_dataset(self):

        dataset = datasets.CelebA(transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.CenterCrop((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            target_type='attr', download=True, root='./data/datasets/celeba')

        mask = torch.zeros(dataset.attr.shape[0], dtype=torch.bool)
        for i, elem in enumerate(dataset.attr):
            if elem[20] == 1:
                mask[i] = True
            else:
                mask[i] = False

        x1 = dataset.filename[~mask]
        y1 = dataset.attr[~mask]
        x2 = dataset.filename[mask]
        y2 = dataset.attr[mask]

        ros = RandomOverSampler(sampling_strategy='minority')

        attr = torch.zeros((len(y1), 1), dtype=torch.long)
        for i, elem in enumerate(y1):
            if elem[9] == 1:
                attr[i] = 1
            else:
                attr[i] = 0

        x1, y1 = ros.fit_resample(x1.reshape(-1, 1), attr)

        attr = torch.zeros((len(y2), 1), dtype=torch.long)
        for i, elem in enumerate(y2):
            if elem[9] == 1:
                attr[i] = 1
            else:
                attr[i] = 0

        x2, y2 = ros.fit_resample(x2.reshape(-1, 1), attr)

        x1, y1 = shuffle(x1, y1)
        x2, y2 = shuffle(x2, y2)

        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))

        dataset.filename = np.squeeze(x)
        dataset.attr = torch.squeeze(torch.nn.functional.one_hot(torch.from_numpy(y)))
        return dataset

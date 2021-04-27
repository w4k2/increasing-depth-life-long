from sklearn.utils import shuffle
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import torch


class CustomFashionMnistDataset(datasets.FashionMNIST):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class FashionMnistDataset:

    def load_dataset(self):

        dataset = CustomFashionMnistDataset(transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.CenterCrop((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))]),
            download=True, root='./data/datasets/fashion-mnist')

        first_concept_elements = [0, 1, 2, 5, 7]
        mask = []
        for target in dataset.targets:
            if target in first_concept_elements:
                mask.append(True)
            else:
                mask.append(False)

        mask = np.asarray(mask)

        x1 = dataset.data[~mask]
        y1 = dataset.targets[~mask]
        x2 = dataset.data[mask]
        y2 = dataset.targets[mask]

        with_sleeves_indices = [0, 2, 3, 4, 6]

        for i, (y1_elem, y2_elem) in enumerate(zip(y1, y2)):
            if y1_elem in with_sleeves_indices:
                y1[i] = 1
            else:
                y1[i] = 0

            if y2_elem in with_sleeves_indices:
                y2[i] = 1
            else:
                y2[i] = 0

        x1 = np.repeat(x1, 3, axis=0)
        y1 = np.repeat(y1, 3, axis=0)
        x2 = np.repeat(x2, 3, axis=0)
        y2 = np.repeat(y2, 3, axis=0)

        x1, y1 = shuffle(x1, y1)
        x2, y2 = shuffle(x2, y2)

        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))

        dataset.data = torch.from_numpy(x)
        dataset.targets = torch.squeeze(torch.nn.functional.one_hot(torch.from_numpy(y).to(torch.int64)))
        return dataset

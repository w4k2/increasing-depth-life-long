import pathlib
import sklearn.model_selection
import PIL
import PIL.Image as Image


class NOTMNIST:
    def __init__(self, dataset_path, train=True, transforms=tuple(), seed=42) -> None:
        self.dataset_path = pathlib.Path(dataset_path) / 'notMNIST_large'
        if not self.dataset_path.exists():
            raise RuntimeError(f'Invalid dataset_path for NOTMNIST: {dataset_path}')
        self.train = train
        self.transforms = transforms
        self.seed = seed

        filepaths = list()
        labels = list()
        for label, class_dir in enumerate(self.dataset_path.iterdir()):
            class_filepaths = list(class_dir.iterdir())
            class_filepaths = self._check_for_corrupted_images(class_filepaths)
            filepaths.extend(class_filepaths)
            labels.extend([label for _ in range(len(class_filepaths))])

        train_filepaths, test_filepaths, train_labels, test_labels = sklearn.model_selection.train_test_split(filepaths, labels, test_size=0.2, random_state=seed, stratify=labels)

        if self.train:
            self.filepaths = train_filepaths
            self.targets = train_labels
        else:
            self.filepaths = test_filepaths
            self.targets = test_labels

    def _check_for_corrupted_images(self, class_filepaths):
        filtered_filepaths = []
        for filepath in class_filepaths:
            try:
                Image.open(filepath)
                filtered_filepaths.append(filepath)
            except PIL.UnidentifiedImageError:
                continue
        return filtered_filepaths

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        filepath = self.filepaths[idx]
        img = Image.open(filepath)
        if self.transforms:
            img = self.transforms(img)
        label = self.targets[idx]
        return img, label


if __name__ == '__main__':
    dataset = NOTMNIST('./data/datasets')

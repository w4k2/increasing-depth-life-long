import torchvision
import torchvision.transforms as transforms

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class SelectedClassesDataset:
    def __init__(self, root, classes_folders, class_indexes, transform=None, target_transform=None):
        self.root = root
        self.loader = torchvision.datasets.folder.default_loader
        self.transform = transform
        self.target_transform = target_transform
        class_to_idx = {class_name: i for i, class_name in zip(class_indexes, classes_folders)}
        self.samples = self.make_dataset(self.root, class_to_idx)

    @ staticmethod
    def make_dataset(
        directory,
        class_to_idx,
        is_valid_file=None,
    ):
        return torchvision.datasets.folder.make_dataset(directory, class_to_idx, extensions=IMG_EXTENSIONS, is_valid_file=is_valid_file)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def tiny_imagenet_dataset(classes_folders, class_indexes):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = SelectedClassesDataset('/home/jkozal/Documents/ML/datasets/tiny-imagenet/tiny-imagenet-200/train/', classes_folders, class_indexes, transform)
    return dataset

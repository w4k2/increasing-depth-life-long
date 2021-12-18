import torch
import torchvision.transforms as trans

from avalanche.benchmarks.classic import SplitTinyImageNet, SplitCIFAR100, SplitMNIST


benchmark = SplitTinyImageNet(n_experiences=10, )
train_stream = benchmark.train_stream
test_stream = benchmark.test_stream

print(dir(train_stream))
exit()

for task in train_stream:
    print(dir(task.dataset))
    print(task.dataset)
    print(len(task.dataset))

    targets = torch.unique(torch.Tensor(task.dataset.targets))
    print(len(targets))
    print(targets)
    # print(task.dataset.targets_task_labels)
    # print(task.dataset.taks_labels)

    # # print(task.task_labels)
    # images = []
    # for data, _, _ in task.dataset:
    #     # print(data)
    #     # print(data.shape)
    #     images.append(data)
    # images = torch.stack(images, dim=0)

    # images_flattened = images.transpose(0, 1)
    # print('images_flattened = ', images_flattened.shape)
    # images_flattened = images_flattened.reshape((3, -1))
    # mean = torch.mean(images_flattened, dim=1)
    # std = torch.std(images_flattened, dim=1)

    # print('mean = ', mean)
    # print('std = ', std)

    # for data, _, _ in task.dataset:

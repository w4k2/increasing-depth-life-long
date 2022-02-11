from torchvision.transforms import ToTensor
from avalanche.benchmarks.classic import PermutedMNIST, SplitCIFAR100, SplitMNIST, SplitCIFAR10, SplitTinyImageNet, CORe50
import torch

benchmark = SplitTinyImageNet(n_experiences=1,
                              train_transform=ToTensor(),
                              eval_transform=None,
                              seed=42
                              )


train_stream = benchmark.train_stream
task = train_stream[0]

images = []
for image, _, _ in task.dataset:
    images.append(image)

    print(len(images))

    # if len(images) > 5:
    #     break

images = torch.stack(images, dim=0)
images = torch.transpose(images, 0, 1)
images = torch.reshape(image, (3, -1))

# mean = torch.mean(images, dim=1)
# print(mean)

std, mean = torch.std_mean(images, dim=1)

print('mean = ', mean)
print('std = ', std)

import torchvision
import torchvision.transforms as trans
import matplotlib.pyplot as plt
import copy

dataset = torchvision.datasets.CIFAR100(root='/home/jkozal/Documents/PWr/stochastic_depth/data/datasets/', download=True)
sampled_img, _ = dataset[0]
plt.subplot(1, 3, 1)
plt.imshow(sampled_img)

crop = trans.RandomResizedCrop(size=64, scale=(0.8, 1.0), ratio=(0.75, 1.33))  # , interpolation=2)
img_aug = crop(dataset[0][0])
plt.subplot(1, 3, 2)
plt.imshow(img_aug)

resize = trans.Resize(size=64)
img_resized = resize(dataset[0][0])
plt.subplot(1, 3, 3)
plt.imshow(img_resized)
plt.show()

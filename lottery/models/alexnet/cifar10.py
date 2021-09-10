import torch
import torchvision.transforms as transforms
from torchvision import datasets

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_set = datasets.CIFAR10("../data", train=False, download=True, transform=transforms)

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("../data", train=True, download=True, transform=transforms),
    batch_size=128,
    num_workers=2,
    shuffle=True,
    pin_memory=True,
)

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=128, num_workers=2, shuffle=True, pin_memory=True
)

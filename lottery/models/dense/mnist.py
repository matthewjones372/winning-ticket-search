import warnings

import torch
from torchvision import datasets
from torchvision.transforms import transforms

from lottery.models.data_path import DATA_PATH

warnings.filterwarnings("ignore", category=UserWarning)

forward_transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

training_set = datasets.MNIST(
    DATA_PATH, train=True, download=True, transform=forward_transform
)
test_set = datasets.MNIST(DATA_PATH, train=False, transform=forward_transform)

forward_train_loader = torch.utils.data.DataLoader(
    training_set, batch_size=512, num_workers=4, shuffle=True, pin_memory=True
)

forward_test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=512, num_workers=4, shuffle=True, pin_memory=True
)

forward_representative_loader = torch.utils.data.DataLoader(
    test_set, batch_size=1, num_workers=1, shuffle=True, pin_memory=False
)

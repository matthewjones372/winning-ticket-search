# Winning Ticket Search for Pytorch

### Info

Implementation of the Winning Ticket Search algorithm for my Masters in Machine Learning. As described
by [Frankle](https://arxiv.org/abs/1803.03635) in the The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural
Networks paper.

# Usage

#### Searching for a winning ticket

##### Example of finding a winning ticket for AlexNet

```python
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.models import AlexNet
import torchvision.transforms as transforms

from lottery.WinningTicket import WinningTicket
from lottery.training.ClassificationTrainer import ClassificationTrainer

# Define the model to search train and find a winning ticket for
model = AlexNet(num_classes=10)

# Define your pytorch train and test data loaders with transforms
# there are pre-defined models and datasets under the models directory for convenience

transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transforms)
test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transforms)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=128,
    num_workers=2,
    shuffle=True,
    pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=128,
                                          num_workers=2,
                                          shuffle=True,
                                          pin_memory=True)

# Select the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define a model trainer with your desired loss function
model_trainer = ClassificationTrainer(
    loss_func=nn.CrossEntropyLoss(),
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
)

# Create a winning ticket instance
winning_ticket = WinningTicket(
    model=model,
    model_trainer=model_trainer,
    device=device,
    # base name is the filename/directory where metrics will be output if logging is switch on
    base_name="ALEXNET_CIFAR10"
)

# search for a winning ticket
winning_ticket.search(
    prune_iterations=40,
    training_iterations=15,
    percentage_prune=5
)
# or by the models desired sparsity
winning_ticket.search_by_target_sparsity(
    target_sparsity=10,
    training_iterations=15,
    percentage_prune=5
)
```

#### Example Finding a Winning Ticket for a QAT model

TODO

# Installation

It is recommended that you install Pytorch and Pytorch Vision first with your set-up in mind. If you want to use a
CPU build, or a CUDA build. Set-up instructions for PyTorch and PyTorch Vision can be
found [here](https://pytorch.org/). If you want to use the quantised models, opt for version >= 1.9 as this is still a
beta feature and can be a bit un-stable.

Currently, the implementation of winning ticket search is not in a public repo:
Please download the latest release and install using [pip](https://pypi.org/project/pip/).

We recommend that use Python version 3.9, versions from 3.8 have been tested.

#### 1. Install Pytorch and Pytorch Vision (Optional if already installed)
(This step is not ideal, working on trying to get Poetry to not override the installed Pytorch version)
```shell
$ python3 -m pip install torch torchvision
```

##### If you wish to use CUDA

##### For CUDA 10
```shell
$ python3 -m pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```

##### For CUDA 11
```shell
$ python3 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 2. Install TorchMetrics
```shell
$ python3 -m pip install torchmetrics
```

#### 3. Install Winning Ticket Search

Finally you can install winning ticket search...

```shell
$ python3 -m pip install lottery-1.0.0-py3-none-any.whl
```

# Developing

This has been built using [Poetry](https://python-poetry.org/) if you want to run the tests and create a build it's
recommended that you install it on your local machine. 

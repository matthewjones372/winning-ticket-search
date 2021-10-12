# 🎟️ Winning Ticket Search

[![Run tests](https://github.com/matthewjones372/winning-ticket-search/actions/workflows/run_test.yml/badge.svg)](https://github.com/matthewjones372/winning-ticket-search/actions/workflows/run_test.yml)

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

```python
import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms

from lottery.WinningQatTicket import WinningQatTicket
from lottery.WinningTicket import WinningTicket
from lottery.training.ClassificationTrainer import ClassificationTrainer



## Define your QAT model with QuantStub and DeQuantStub
class DenseNetworkQAT(nn.Module):
    def __init__(self, num_classes=10):
        super(DenseNetworkQAT, self).__init__()
        self.classifier = nn.Sequential(
            torch.quantization.QuantStub(),
            nn.Linear(28 * 28, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
            torch.quantization.DeQuantStub()
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


model = DenseNetworkQAT(num_classes=10)

# or simply wrap your non QAT model with the Quant wrapper which will insert the layers for you
normal_model = DenseNetwork(num_classes=10)
model = torch.quantization.QuantWrapper(normal_model)

forward_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

training_set = datasets.MNIST('../data', train=True, download=True, transform=forward_transform)
test_set = datasets.MNIST('../data', train=False, transform=forward_transform)

forward_train_loader = torch.utils.data.DataLoader(training_set,
                                                   batch_size=512,
                                                   num_workers=4,
                                                   shuffle=True,
                                                   pin_memory=True)

forward_test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=512,
                                                  num_workers=4,
                                                  shuffle=True,
                                                  pin_memory=True)
# Select the device for training
device = torch.device("cuda")

# define a model trainer with your desired loss function
# set is_quantised to true so model is converted at time of testing
model_trainer = ClassificationTrainer(
    loss_func=nn.CrossEntropyLoss(),
    train_loader=forward_train_loader,
    test_loader=forward_test_loader,
    device=device,
    is_quantised_model=True
)

# Create a winning ticket instance
winning_ticket = WinningQatTicket(
    model=model,
    model_trainer=model_trainer,
    device=device,
    base_name="DENSE_QAT_MNIST",
)

# search for a winning ticket
winning_ticket.search(
    prune_iterations=40, training_iterations=15, percentage_prune=5
)
```

# Installation

It is recommended that you install Pytorch and Pytorch Vision first with your set-up in mind. If you want to use a
CPU build, or a CUDA build. Set-up instructions for PyTorch and PyTorch Vision can be
found [here](https://pytorch.org/). If you want to use the quantised models, opt for version >= 1.9 as this is still a
beta feature and can be a bit un-stable.

Currently, the implementation of winning ticket search is not in a public repo:
Please download the latest release and install using [pip](https://pypi.org/project/pip/).

We recommend that you use Python version 3.9, only versions from 3.8 to 3.9 have been tested.

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

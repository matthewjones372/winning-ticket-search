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

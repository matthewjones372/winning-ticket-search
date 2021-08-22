import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.models import AlexNet
import torchvision.transforms as transforms

from lottery.WinningTicket import WinningTicket
from lottery.training.ClassificationTrainer import ClassificationTrainer

if __name__ == "__main__":
    # Define the class to search train and search for a winning ticket
    model = AlexNet(num_classes=10)

    # Define your pytorch train and test data loaders with transforms
    # there are pre-defined models and datasets under the models directory
    transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_set = datasets.CIFAR10(
        "./data", train=False, download=True, transform=transforms
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("./data", train=True, download=True, transform=transforms),
        batch_size=128,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, num_workers=2, shuffle=True, pin_memory=True
    )

    # Select the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define a model trainer with your loss function
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
        base_name="ALEXNET_CIFAR10",
    )

    # search for a winning ticket
    winning_ticket.search(
        prune_iterations=40, training_iterations=15, percentage_prune=5
    )

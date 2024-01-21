"""
Trains and evaluates an improved version of LeNet-5 on MNIST dataset.
Reference: https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
"""

import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

batch_size = 64
device = "cpu"
epochs = 5
learning_rate = 1e-3

def main():
    train_dataset = MNIST("datasets", train=True, download=True, transform=ToTensor())
    test_dataset = MNIST("datasets", train=False, download=True, transform=ToTensor())
    train_dataloader= DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader= DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model.pth"))

    model.eval()
    x, y = test_dataset[0][0], test_dataset[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = pred[0].argmax(0), y
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>4f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>4f} \n")



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":
    main()
"""
Trains and evaluates an improved version of LeNet-5 on MNIST dataset.
Reference: https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
"""

import torch
from torch import nn
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

batch_size = 64
device = "cpu"
epochs = 50
learning_rate = 1e-3

def main():
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])
    train_dataset = MNIST("datasets", train=True, download=True, transform=transform)
    test_dataset = MNIST("datasets", train=False, download=True, transform=transform)
    train_dataloader= DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader= DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = LeNet5().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    model = LeNet5().to(device)
    model.load_state_dict(torch.load("model.pth"))

    model.eval()
    X, y = next(iter(test_dataloader))
    with torch.no_grad():
        X = X.to(device)
        pred = model(X)
        predicted, actual = pred[0].argmax(0), y[0]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

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


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)), # LeCun talks about s2 having 12 trainable parameters. I guess each of the 6 planes has a linear transformation (2 params) applied elementwise.
            nn.Tanh(), # It should be f(a) = A*tanh(S*a) where A is 1.7159 and S ?
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            # LeCun takls about some kind of dropout mask between c3 and s4
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)), # LeCun talks about s2 having 12 trainable parameters. I guess each of the 6 planes has a linear transformation (2 params) applied elementwise.
            nn.Tanh(), # It should be f(a) = A*tanh(S*a) where A is 1.7159 and S ?
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.Tanh(), # It should be f(a) = A*tanh(S*a) where A is 1.7159 and S ?
            nn.Linear(84, 10),
            # The output layer should be composed of Euclidean Radial Basis Function units (RBF)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    main()
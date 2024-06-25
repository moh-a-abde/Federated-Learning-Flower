import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, num_classes: int, input_dim: int) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(net, trainloader, optimizer, scheduler, epochs, device: str):
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for epoch in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.Adam()
            loss = criterion(net(features), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

def test(net, testloader, device: str):
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            features, labels = data[0].to(device), data[1].to(device)
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

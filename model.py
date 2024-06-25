import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes: int, input_dim: int) -> None:
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def train(net, trainloader, optimizer, epochs, device: str):
        
    # Train network on training set
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    for _ in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            loss_Adam = []
            n_iter = 20
            loss = criterion(net(features), labels)
            # store loss into list
            loss_Adam.append(loss.item())
            # zeroing gradients after each iteration
            optimizer.zero_grad()
            # backward pass for computing the gradients of the loss w.r.t to learnable parameters
            loss.backward()
            # updateing the parameters after each iteration
            optimizer.step()

def test(net, testloader, device: str):
    #  Validate network on entire test set

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

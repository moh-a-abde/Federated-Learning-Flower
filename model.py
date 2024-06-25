import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes: int, input_dim: int) -> None:
        super(Net, self).__init__()
        # Increase the number of neurons in hidden layers
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Add batch normalization layers
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply dropout and batch normalization after each layer
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

def train(net, trainloader, optimizer, epochs, device: str):
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print epoch loss and adjust learning rate
        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        scheduler.step(epoch_loss)

def test(net, testloader, device: str):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total_loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            outputs = net(features)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / len(testloader.dataset)
    avg_loss = total_loss / len(testloader)
    return avg_loss, accuracy

# Add a function for early stopping
def early_stopping(val_losses, patience=10):
    if len(val_losses) > patience:
        if all(val_losses[-1] > val_losses[-(i+2)] for i in range(patience)):
            return True
    return False

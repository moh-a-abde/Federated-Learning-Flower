import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time

class Net(nn.Module):
    def __init__(self, num_classes: int, input_dim: int) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def train_nn(net, trainloader, optimizer, epochs, device: str):
    print('Training Neural Network:')
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    net.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=2, verbose=True)
    val_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        scheduler.step(epoch_loss)
        val_losses.append(epoch_loss)
        if early_stopping(val_losses):
            print("Early stopping triggered")
            break

def test_nn(net, testloader, device: str):
    print('Testing Neural Network:')
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

def early_stopping(val_losses, patience=10):
    if len(val_losses) > patience:
        if all(val_losses[-1] > val_losses[-(i+2)] for i in range(patience)):
            return True
    return False

def train_xgboost():
    
    # Load the dataset
    file_path = 'data/zeek_live_data_merged.csv'
    data = pd.read_csv(file_path)

    # Encode categorical features
    label_encoder = LabelEncoder()
    # Print unique values in the 'label' column
    unique_labels = data['label'].unique()
    print("Unique values in 'label' column:", unique_labels)
    data['label'] = label_encoder.fit_transform(data['label'])

    # Separate the 'ts' column into a different dataset.
    ts_data = data[['ts']]

    # Select features and target
    X = data.drop(columns=['label', 'ts', 'uid'])
    y = data['label']

    # Convert categorical features to numerical
    X = pd.get_dummies(X, columns=['id.orig_h', 'id.resp_h', 'proto', 'conn_state', 'history'])

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Set the test size
    tsz = 0.20

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=tsz, stratify=y, random_state=42)
    l = len(set(y))
    startTrainTime = time.time()
    train = xgb.DMatrix(X_train, label=y_train)
    test = xgb.DMatrix(X_test, label=y_test)
    param = {
        'max_depth': 6,
        'eta': 0.35,
        'objective': 'multi:softmax',
        'num_class': l,
        'eval_metric': 'merror',
        'tree_method': 'hist'
    }
    cv_params = {
        'params': param,
        'dtrain': train,
        'num_boost_round': 20,
        'nfold': 10,
        'metrics': {'merror'},
        'early_stopping_rounds': 10
    }
    cv_results = xgb.cv(**cv_params)
    print(cv_results)
    best_num_boost_round = cv_results.shape[0]
    model = xgb.train(param, train, num_boost_round=best_num_boost_round)
    TrainTime = (time.time() - startTrainTime)
    startTestTime = time.time()
    predictions = model.predict(train)
    TestTime = (time.time() - startTestTime)
    print(f"Training Time: {TrainTime} seconds")
    print(f"Testing Time: {TestTime} seconds")
    accuracy = accuracy_score(y_train, predictions)
    report = classification_report(y_train, predictions)
    print('XGBoost Model Training Metrics:')
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)
    return model

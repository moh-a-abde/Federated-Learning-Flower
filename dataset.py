import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor, Normalize, Compose
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        
        # Convert categorical variables
        self.categorical_features = ['id.resp_p', 'id.resp_h', 'proto', 'query', 'uid']
        self.numerical_features = ['id.orig_p']

        # Define the column transformer with handle_unknown='ignore' for OneHotEncoder
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ])
        
        # Fit and transform the dataset
        self.data_transformed = self.preprocessor.fit_transform(self.data)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data_transformed[idx]
        features = row.astype('float32')
        label = self.data.iloc[idx, -1]  # Assuming the label is the last column
        if self.transform:
            features = self.transform(features)
        return torch.tensor(features), torch.tensor(label, dtype=torch.long)

def get_csv_dataset(csv_file: str, transform=None):
    dataset = CSVDataset(csv_file, transform=transform)
    return dataset

def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1, csv_file: str = '/mnt/data/live_data_part1.csv'):
    # Transformation and dataset loading
    tr = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    dataset = get_csv_dataset(csv_file, transform=tr)
    
    # Check if dataset is loaded
    print(f"Total number of samples: {len(dataset)}")

    # Split dataset into partitions
    num_images = len(dataset) // num_partitions
    remainder = len(dataset) % num_partitions
    partition_len = [num_images + 1 if i < remainder else num_images for i in range(num_partitions)]
    
    datasets = random_split(dataset, partition_len, torch.Generator().manual_seed(2024))

    # Creating train and validation loaders
    trainloaders = []
    valloaders = []
    for dataset_ in datasets:
        num_total = len(dataset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(dataset_, [num_train, num_val], torch.Generator().manual_seed(2024))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    # Creating test loader
    testloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
    
    return trainloaders, valloaders, testloader

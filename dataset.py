import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from typing import List

class PreprocessedCSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        
        self.categorical_features = ['id.orig_h', 'id.resp_h', 'proto', 'history', 'uid', 'conn_state']
        self.numerical_features = ['id.orig_p', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'missed_bytes',
                                   'local_resp', 'local_orig', 'resp_bytes', 'orig_bytes', 'duration', 'id.resp_p']
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ]
        )
        
        self.features = self.data.drop(columns=['label', 'ts'])
        self.labels = self.data['label']
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        
        self.features_transformed = self.preprocessor.fit_transform(self.features)
        if not isinstance(self.features_transformed, np.ndarray):
            self.features_transformed = self.features_transformed.toarray()
        self.input_dim = self.features_transformed.shape[1]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.features_transformed[idx].astype('float32')
        label = self.labels_encoded[idx]
        if self.transform:
            features = self.transform(features)
        return torch.tensor(features), torch.tensor(label, dtype=torch.long)

def get_csv_dataset(csv_file: str, transform=None):
    dataset = PreprocessedCSVDataset(csv_file, transform=transform)
    return dataset

def prepare_dataset(num_partitions: int, batch_size: int, num_classes: int, val_ratio: float = 0.1, csv_files: List[str] = []):
    trainloaders, valloaders, datasets = [], [], []
    
    for csv_file in csv_files:
        dataset = get_csv_dataset(csv_file)
        datasets.append(dataset)
        
        num_total = len(dataset)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val
        
        for_train, for_val = random_split(dataset, [num_train, num_val], torch.Generator().manual_seed(2024))
        
        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))
    
    testloader = DataLoader(datasets[0], batch_size=64, shuffle=False, num_workers=2)  # Use one of the datasets for the test loader

    return trainloaders, valloaders, testloader

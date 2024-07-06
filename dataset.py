import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from typing import List

class PreprocessedCSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # Load and preprocess data
        self.data = pd.read_csv(csv_file)
        
        # Define categorical and numerical features
        self.categorical_features = ['id.orig_h', 'id.resp_h', 'proto', 'history', 'uid', 'conn_state']
        self.numerical_features = ['id.orig_p', 'orig_pkts',	'orig_ip_bytes',	'resp_pkts', 'missed_bytes'
        , 'local_resp', 'local_orig', 'resp_bytes', 'orig_bytes', 'duration', 'id.resp_p']
        
        # Define the column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ]
        )
        
        # Separate features and labels
        self.features = self.data.drop(columns=['label', 'ts'])
        self.labels = self.data['label']
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        
        # Fit and transform the features
        self.features_transformed = self.preprocessor.fit_transform(self.features)
        self.input_dim = self.features_transformed.shape[1]  # Define input_dim based on the transformed features' shape
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.features_transformed[idx].astype('float32').todense()
        features = np.asarray(features).flatten()
        label = self.labels_encoded[idx]
        if self.transform:
            features = self.transform(features)
        return torch.tensor(features), torch.tensor(label, dtype=torch.long)

def get_csv_dataset(csv_file: str, transform=None):
    dataset = PreprocessedCSVDataset(csv_file, transform=transform)
    return dataset

def prepare_dataset(num_partitions: int, batch_size: int, num_classes: int, val_ratio: float = 0.1, csv_files: List[str] = []):
    trainloaders, valloaders, testloader = [], [], None
    datasets = []
    
    for csv_file in csv_files:
        dataset = get_csv_dataset(csv_file)
        datasets.append(dataset)
        
        num_total = len(dataset)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val
        
        for_train, for_val = random_split(dataset, [num_train, num_val], torch.Generator().manual_seed(2024))
        
        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    return trainloaders, valloaders, testloader, datasets

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

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

def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1, num_classes: int, csv_file: str = 'data/zeek_live_data_labeled-2.csv'):
    # Load and preprocess the dataset
    dataset = get_csv_dataset(csv_file)
    
    # Check if dataset is loaded
    print(f"Total number of samples: {len(dataset)}")

    # Ensure no partition is empty by adjusting the partition lengths
    num_images = len(dataset) // num_partitions
    remainder = len(dataset) % num_partitions
    partition_len = [num_images + 1 if i < remainder else num_images for i in range(num_partitions)]
    
    # Ensure that no partition length is zero
    partition_len = [length for length in partition_len if length > 0]
    
    datasets = random_split(dataset, partition_len, torch.Generator().manual_seed(2024))

    # Creating train and validation loaders
    trainloaders = []
    valloaders = []
    for dataset_ in datasets:
        num_total = len(dataset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        if num_train == 0:
            continue  # Skip if no training data
        
        for_train, for_val = random_split(dataset_, [num_train, num_val], torch.Generator().manual_seed(2024))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    # Creating test loader
    testloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)
    
    return trainloaders, valloaders, testloader, dataset.input_dim

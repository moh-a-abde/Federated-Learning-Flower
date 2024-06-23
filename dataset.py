import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor, Normalize, Compose
import pandas as pd

class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = row[:-1].values.astype('float32')
        label = row[-1]
        if self.transform:
            features = self.transform(features)
        return torch.tensor(features), torch.tensor(label, dtype=torch.long)

def get_csv_dataset(csv_file: str, transform=None):
    dataset = CSVDataset(csv_file, transform=transform)
    return dataset

def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1, csv_file: str = '/data/live_data_part1.csv'):

    tr = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    dataset = get_csv_dataset(csv_file, transform=tr)

    # Split dataset into 'num_partitions' datasets
    num_images = len(dataset) // num_partitions
    partition_len = [num_images] * num_partitions
    datasets = random_split(dataset, partition_len, torch.Generator().manual_seed(2024))

    # Create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    for dataset_ in datasets:
        num_total = len(dataset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(dataset_, [num_train, num_val], torch.Generator().manual_seed(2024))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))

    # Create a testloader
    testloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

    return trainloaders, valloaders, testloader

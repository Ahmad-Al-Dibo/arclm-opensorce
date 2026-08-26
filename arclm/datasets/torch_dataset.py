"""
Dataset Module - Data handling and batching
"""

import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """Dataset for language modeling with sliding window"""
    
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx : idx + self.block_size])
        y = torch.tensor(self.data[idx + 1 : idx + self.block_size + 1])
        return x, y


def create_dataloader(encoded_data, block_size, batch_size, shuffle=True):
    """Create DataLoader from encoded data"""
    dataset = TextDataset(encoded_data, block_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return loader

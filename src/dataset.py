import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class MultiUserCSIDataset(Dataset):
    def __init__(self, annotation_file, data_dir, fixed_length=3000, transform=None):
        self.annotations = pd.read_csv(annotation_file)
        self.data_dir = data_dir
        self.fixed_length = fixed_length
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        sample_name = row['sample_label']
        
        file_path = os.path.join(self.data_dir, f"{sample_name}.npy")
        csi_tensor = np.load(file_path).astype(np.float32) # Shape: [Time, Tx, Rx, Subcarriers]
        
        # Handle variable time lengths via padding or truncation to fixed_length (3000)
        current_length = csi_tensor.shape[0]
        if current_length < self.fixed_length:
            # Pad with zeros along the time axis (axis 0)
            pad_size = self.fixed_length - current_length
            csi_tensor = np.pad(csi_tensor, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode='constant')
        elif current_length > self.fixed_length:
            # Truncate if it exceeds
            csi_tensor = csi_tensor[:self.fixed_length, :, :, :]
            
        if self.transform:
            csi_tensor = self.transform(csi_tensor)
            
        x = torch.tensor(csi_tensor)
        return x, sample_name
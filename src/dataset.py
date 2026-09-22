import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pywt # Make sure PyWavelets is installed for DWT

class MultiUserCSIDatasetWithPreprocessing(Dataset):
    def __init__(self, annotation_file, data_dir, fixed_length=3000, transform=None):
        self.annotations = pd.read_csv(annotation_file)
        self.data_dir = data_dir
        self.fixed_length = fixed_length
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        sample_name = row['label']
        
        file_path = os.path.join(self.data_dir, f"{sample_name}.npy")
        csi_tensor = np.load(file_path).astype(np.float32) # Raw Shape: [Time, Tx, Rx, Subcarriers]
        
        # 1. Placeholder / Implementation for CFO & SFO Correction
        # (Removes phase/frequency offsets across subcarriers and time axes)
        # csi_tensor = apply_cfo_sfo_correction(csi_tensor)
        
        # 2. Handle variable time lengths via padding or truncation to fixed_length (3000)
        current_length = csi_tensor.shape[0]
        if current_length < self.fixed_length:
            pad_size = self.fixed_length - current_length
            csi_tensor = np.pad(csi_tensor, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode='constant')
        elif current_length > self.fixed_length:
            csi_tensor = csi_tensor[:self.fixed_length, :, :, :]
            
        # 3. Apply Discrete Wavelet Transform (DWT) for temporal compression / denoising if needed
        # Example decomposition along the time axis
        # coeffs = pywt.wavedec(csi_tensor, 'db1', level=1, axis=0)
        # csi_tensor = coeffs[0] # or combined representation
        
        # 4. Reshape from [Time, Tx, Rx, Subcarriers] to [Channels, Time, Subcarriers] or similar 
        # To match expected shape: (B, C, T, S) where C combines Tx * Rx antennas (e.g., 3 * 3 = 9 channels)
        time_steps, tx, rx, subcarriers = csi_tensor.shape
        channels = tx * rx
        
        # Reshape to [Time, Channels, Subcarriers] then permute to [Channels, Time, Subcarriers] -> (C, T, S)
        csi_tensor = csi_tensor.reshape(time_steps, channels, subcarriers)
        csi_tensor = np.transpose(csi_tensor, (1, 0, 2)) # Shape: [C, T, S]
        
        # 5. Standardized zero-mean, unit variance normalization per sample
        mean = np.mean(csi_tensor, axis=(1, 2), keepdims=True)
        std = np.std(csi_tensor, axis=(1, 2), keepdims=True)
        csi_tensor = (csi_tensor - mean) / (std + 1e-5)
            
        if self.transform:
            csi_tensor = self.transform(csi_tensor)
            
        x = torch.tensor(csi_tensor, dtype=torch.float32)
        return x, sample_name
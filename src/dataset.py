"""
CAPA-Fi: Multi-User CSI Dataset with Preprocessing (Member 1 / Member 2)

Loads WiMANS CSI data and extracts M=6 target slots from source_split.csv:
    x:            [C, T, S] = [3, 3000, 30]
    target_slots: [M] = [6] with values in {0..6}
                  0 = NO_PERSON
                  1..6 = defined project activity classes
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# WiMANS Activity to Class Index mapping
# Config-defined 7 classes:
# 0 = NO_PERSON (unoccupied slot / background)
# 1..6 = The six activities from configs/default_config.yaml (1:Walk, 2:Sit, 3:Stand, 4:Fall, 5:Wave, 6:Pick-up)
ACTIVITY_TO_CLASS: Dict[str, int] = {
    # 0: NO_PERSON / background
    "no_person": 0,
    "none": 0,
    "nothing": 0,
    # 1: Walk
    "walk": 1,
    # 2: Sit
    "sit": 2,
    "sit_down": 2,
    # 3: Stand
    "stand": 3,
    "stand_up": 3,
    # 4: Fall / Lie down
    "fall": 4,
    "lie_down": 4,
    # 5: Wave
    "wave": 5,
    # 6: Pick-up
    "pick_up": 6,
    "pickup": 6,
}

CLASS_TO_ACTIVITY: Dict[int, str] = {
    0: "NO_PERSON",
    1: "Walk",
    2: "Sit",
    3: "Stand",
    4: "Fall",
    5: "Wave",
    6: "Pick-up",
}


def map_activity_to_class(activity_name: Optional[str]) -> int:
    """Maps an activity string to class index in {0..6}."""
    if activity_name is None or pd.isna(activity_name):
        return 0
    clean = str(activity_name).strip().lower()
    return ACTIVITY_TO_CLASS.get(clean, 0)


def extract_target_slots(row: pd.Series, num_slots: int = 6) -> torch.Tensor:
    """
    Extracts M target slots from the six user activity columns in source_split.csv:
    user_1_activity ... user_6_activity.

    Active users are placed into slots; remaining slots are padded with class 0 (NO_PERSON).

    Examples:
        0 users -> [NO_PERSON, NO_PERSON, NO_PERSON, NO_PERSON, NO_PERSON, NO_PERSON]
        1 user  -> [activity, NO_PERSON, NO_PERSON, NO_PERSON, NO_PERSON, NO_PERSON]
        2 users -> [act1, act2, NO_PERSON, NO_PERSON, NO_PERSON, NO_PERSON]
    """
    active_classes = []
    for i in range(1, 7):
        col = f"user_{i}_activity"
        if col in row and pd.notna(row[col]):
            act_str = str(row[col]).strip().lower()
            if act_str and act_str not in ("nan", ""):
                cls_idx = ACTIVITY_TO_CLASS.get(act_str, 0)
                if cls_idx > 0:
                    active_classes.append(cls_idx)

    # Pad remaining slots with 0 (NO_PERSON)
    slots = [0] * num_slots
    for idx, cls_idx in enumerate(active_classes[:num_slots]):
        slots[idx] = cls_idx

    return torch.tensor(slots, dtype=torch.long)


class MultiUserCSIDatasetWithPreprocessing(Dataset):
    """
    Multi-user CSI Dataset with preprocessing matching Member 1 and Member 2 contract:
        - Output x: [C, T, S] = [3, 3000, 30]
        - Output target_slots: [M] = [6]
    """

    def __init__(
        self,
        annotation_file: str,
        data_dir: str = "data/processed/WiMANS",
        fixed_length: int = 3000,
        num_slots: int = 6,
        in_channels: int = 3,
        subcarriers: int = 30,
        transform=None,
    ):
        self.annotation_file = annotation_file
        self.annotations = pd.read_csv(annotation_file)
        self.data_dir = data_dir
        self.fixed_length = fixed_length
        self.num_slots = num_slots
        self.in_channels = in_channels
        self.subcarriers = subcarriers
        self.transform = transform

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.annotations.iloc[idx]
        sample_name = str(row["label"])

        # Target slots: [6]
        target_slots = extract_target_slots(row, num_slots=self.num_slots)

        file_path = os.path.join(self.data_dir, f"{sample_name}.npy")
        if os.path.isfile(file_path):
            csi_tensor = np.load(file_path).astype(np.float32)  # Raw: [Time, Tx, Rx, Subcarriers]

            # 1. Handle variable time lengths via padding or truncation to fixed_length (3000)
            current_length = csi_tensor.shape[0]
            if current_length < self.fixed_length:
                pad_size = self.fixed_length - current_length
                csi_tensor = np.pad(
                    csi_tensor, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode="constant"
                )
            elif current_length > self.fixed_length:
                csi_tensor = csi_tensor[: self.fixed_length, :, :, :]

            # 2. Reshape from [Time, Tx, Rx, Subcarriers] to [Channels, Time, Subcarriers]
            time_steps, tx, rx, subcarriers = csi_tensor.shape
            channels = tx * rx
            csi_tensor = csi_tensor.reshape(time_steps, channels, subcarriers)
            csi_tensor = np.transpose(csi_tensor, (1, 0, 2))  # [Channels, Time, Subcarriers]

            # Ensure channel count matches in_channels (3)
            if csi_tensor.shape[0] > self.in_channels:
                csi_tensor = csi_tensor[: self.in_channels, :, :]
            elif csi_tensor.shape[0] < self.in_channels:
                pad_ch = self.in_channels - csi_tensor.shape[0]
                csi_tensor = np.pad(csi_tensor, ((0, pad_ch), (0, 0), (0, 0)), mode="constant")

            # 3. Standardized zero-mean, unit variance normalization per sample
            mean = np.mean(csi_tensor, axis=(1, 2), keepdims=True)
            std = np.std(csi_tensor, axis=(1, 2), keepdims=True)
            csi_tensor = (csi_tensor - mean) / (std + 1e-5)
        else:
            # Synthetic fallback when .npy files are not yet on disk
            csi_tensor = np.random.randn(
                self.in_channels, self.fixed_length, self.subcarriers
            ).astype(np.float32)

        if self.transform:
            csi_tensor = self.transform(csi_tensor)

        x = torch.tensor(csi_tensor, dtype=torch.float32)
        return x, target_slots
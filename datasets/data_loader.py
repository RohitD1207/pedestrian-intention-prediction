import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

class PIEDataset(Dataset):
    def __init__(self, annotation_file, crop_dir, sequence_length=16):
        self.annotations = pd.read_csv(annotation_file).to_dict("records")
        self.crop_dir = crop_dir 
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations[idx]
        video_name = str(row["video"])
        target_frame = int(row["frame"])
        label = int(row["label"])
        pid = str(row["pedestrian_id"])
        sequence = []
        for i in range(self.sequence_length):
            frame_idx = target_frame - (self.sequence_length - 1) + i
            img_path = os.path.join(self.crop_dir, video_name, pid, f"{frame_idx:06d}.jpg")       
            image_np = cv2.imread(img_path) if os.path.exists(img_path) else None
            
            if image_np is None:
                image = torch.zeros((3, 224, 224))
            else:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                image = torch.from_numpy(image_np).permute(2,0,1).float() / 255.0
                image = TF.resize(image, [224, 224])
            
            sequence.append(image)

        sequence = torch.stack(sequence)
        return sequence, torch.tensor(label), pid
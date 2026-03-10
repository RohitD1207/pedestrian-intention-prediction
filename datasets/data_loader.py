import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms


class PIEDataset(Dataset):

    def __init__(self, annotation_file, video_dir, sequence_length=16):

        self.annotations = pd.read_csv(annotation_file)
        self.video_dir = video_dir
        self.sequence_length = sequence_length

        self.video_cache = {}

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.annotations)

    """def _load_frame(self, video_path, frame_number):

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Failed to read frame")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame"""

    def _load_frame(self, video_name, frame_number):

        if video_name not in self.video_cache:
            video_path = os.path.join(self.video_dir, video_name + ".mp4")
            self.video_cache[video_name] = cv2.VideoCapture(video_path)

        cap = self.video_cache[video_name]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()

        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_number} from {video_name}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame



    def __getitem__(self, idx):

        row = self.annotations.iloc[idx]

        video_name = row["video"]
        frame = int(row["frame"])
        label = int(row["label"])

        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]

        sequence = []

        for i in range(self.sequence_length):

            frame_num = frame - self.sequence_length + i

            image = self._load_frame(video_name, frame_num)

            crop = image[int(y1):int(y2), int(x1):int(x2)]

            crop = self.transform(crop)

            sequence.append(crop)

        sequence = torch.stack(sequence)

        return sequence, torch.tensor(label)

    def __del__(self):
        for cap in self.video_cache.values():
            cap.release()
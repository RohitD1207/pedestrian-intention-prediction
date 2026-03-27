import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class PIEDataset(Dataset):

    def __init__(self, annotation_file, video_dir, sequence_length=16):

        self.annotations = pd.read_csv(annotation_file).to_dict("records")
        self.video_dir = video_dir
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.annotations)


    def _load_sequence(self, video_path, start_frame):

        cap = cv2.VideoCapture(video_path)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []

        for _ in range(self.sequence_length):

            ret, frame = cap.read()

            if not ret:
                raise RuntimeError("Failed to read frame")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

        cap.release()

        return frames


    def __getitem__(self, idx):

        row = self.annotations[idx]

        video_name = row["video"]
        frame = int(row["frame"])
        label = int(row["label"])
        pid = int(row["pedestrian_id"])

        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])

        video_path = os.path.join(self.video_dir, video_name + ".mp4")

        start_frame = frame - self.sequence_length

        frames = self._load_sequence(video_path, start_frame)

        sequence = []

        for image in frames:

            crop = image[y1:y2, x1:x2]

            crop = TF.resize(
                torch.from_numpy(crop).permute(2,0,1).float()/255.0,
                (224,224)
            )

            sequence.append(crop)

        sequence = torch.stack(sequence)

        return sequence, torch.tensor(label), pid
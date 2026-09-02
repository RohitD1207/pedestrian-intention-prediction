import torch
from torch.utils.data import Dataset

from datasets.data_loader import PIEDataset
from models.pose_extracter import PoseExtractor
from models.graph_builder import build_graph


class GraphDataset(Dataset):

    def __init__(
        self,
        annotation_file,
        crop_dir,
        sequence_length=16
    ):
        self.dataset = PIEDataset(
            annotation_file=annotation_file,
            crop_dir=crop_dir,
            sequence_length=sequence_length
        )

        self.pose_model = PoseExtractor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        frames, label, pid = self.dataset[index]

        keypoints = self.pose_model.extract(frames)

        node_features, edge_index = build_graph(
            keypoints
        )

        return (
            node_features,
            edge_index,
            label,
            pid
        )
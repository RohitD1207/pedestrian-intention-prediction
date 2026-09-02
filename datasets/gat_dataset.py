import torch
from pathlib import Path
from torch.utils.data import Dataset
from torch_geometric.data import Data


class CachedGraphDataset(Dataset):

    def __init__(self, cache_dir):

        self.cache_dir = Path(cache_dir)

        self.files = sorted(
            self.cache_dir.glob("*.pt")
        )

        if len(self.files) == 0:
            raise RuntimeError(
                f"No graph files found in {self.cache_dir}"
            )

    def __len__(self):

        return len(self.files)

    def __getitem__(self, index):

        sample = torch.load(
            self.files[index],
            map_location="cpu",
            weights_only=True
        )

        x = sample["node_features"].float()

        edge_index = sample["edge_index"].long()

        y = sample["label"].float()

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y
        )

        data.pid = sample["pid"]

        return data
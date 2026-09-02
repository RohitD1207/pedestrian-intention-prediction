import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATModel(nn.Module):

    def __init__(
        self,
        in_channels=3,
        hidden_channels=64,
        heads=4,
        dropout=0.3
    ):
        super().__init__()

        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout
        )

        self.gat2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, x, edge_index=None):

        if edge_index is None:
            edge_index = x.edge_index
            x = x.x

        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = self.dropout(x)

        x = self.gat2(x, edge_index)
        x = F.elu(x)

        x = x.mean(dim=0, keepdim=True)

        x = self.classifier(x)

        return x
import torch
import torch.nn as nn

class LSTMModel(nn.Module):

    def __init__(self, feature_dim, units=128, dropout=0.2):
        super().__init__()

        self.lstm1 = nn.LSTM(feature_dim, units, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(units, units//2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(units//2, 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()

    def forward(self, x):

        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.dropout2(out)

        out = self.relu(self.fc1(out))
        out = self.fc2(out)

        return out
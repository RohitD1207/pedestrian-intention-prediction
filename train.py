import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

from datasets.data_loader import PIEDataset
from models.resnet_encoder import ResNetEncoder
from models.lstm import LSTMModel


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True


    # ------------------------
    # Dataset
    # ------------------------

    dataset = PIEDataset(
        annotation_file="datasets/pie_annotations_clean.csv",
        video_dir="data/PIE_clips/set01"
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("Dataset loaded")


    # ------------------------
    # Models
    # ------------------------

    resnet = ResNetEncoder().to(device)
    resnet.eval()   # we only extract features

    lstm = LSTMModel(feature_dim=512).to(device)


    # ------------------------
    # FEATURE EXTRACTION
    # ------------------------

    if os.path.exists("pie_features.pt"):

        print("Loading cached features...")

        features_tensor = torch.load("pie_features.pt")
        labels_tensor = torch.load("pie_labels.pt")

    else:

        print("Extracting ResNet features...")

        all_features = []
        all_labels = []

        with torch.no_grad():

            for seq, label in tqdm(loader, desc="Extracting features"):

                seq = seq.to(device)

                features = resnet(seq)

                all_features.append(features.cpu())
                all_labels.append(label)

        features_tensor = torch.cat(all_features)
        labels_tensor = torch.cat(all_labels)

        torch.save(features_tensor, "pie_features.pt")
        torch.save(labels_tensor, "pie_labels.pt")

    print("Features ready:", features_tensor.shape)


    # ------------------------
    # NEW DATASET (features)
    # ------------------------

    feature_dataset = TensorDataset(features_tensor, labels_tensor)

    feature_loader = DataLoader(
        feature_dataset,
        batch_size=32,
        shuffle=True
    )


    # ------------------------
    # Loss + Optimizer
    # ------------------------

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        lstm.parameters(),
        lr=0.0005
    )


    # ------------------------
    # TRAIN LSTM ONLY
    # ------------------------

    epochs = 30

    print("Training LSTM")

    for epoch in range(epochs):

        lstm.train()
        running_loss = 0

        for features, label in feature_loader:

            features = features.to(device)
            label = label.float().unsqueeze(1).to(device)

            logits = lstm(features)

            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(feature_loader):.4f}")


if __name__ == "__main__":
    main()
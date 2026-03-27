import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import numpy as np  # Added missing import
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from datasets.data_loader import PIEDataset
from models.resnet_encoder import ResNetEncoder
from models.lstm import LSTMModel
from models.mc_dropout import monte_carlo_dropout


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    MODEL_PATH = "lstm_weights.pth"

    # ------------------------
    # Dataset
    # ------------------------
    dataset = PIEDataset(
        annotation_file="pie_annotations_clean.csv",
        video_dir="data/PIE_clips/set03"
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
    resnet.eval()

    lstm = LSTMModel(feature_dim=512).to(device)

    # ------------------------
    # FEATURE EXTRACTION
    # ------------------------
    if os.path.exists("pie_features.pt"):
        print("Loading cached features...")
        features_tensor = torch.load("pie_features.pt", weights_only=True)
        labels_tensor = torch.load("pie_labels.pt", weights_only=True)
        ids = torch.load("pie_ids.pt") 
    else:
        print("Extracting ResNet features...")
        all_features = []
        all_labels = []
        all_ids = []

        with torch.no_grad():
            for seq, label, pid in tqdm(loader, desc="Extracting features"):
                seq = seq.to(device)
                features = resnet(seq)

                all_features.append(features.cpu())
                all_labels.append(label)

                # pid can be string → store as list
                if isinstance(pid, list):
                    all_ids.extend(pid)
                else:
                    all_ids.append(pid)

        features_tensor = torch.cat(all_features)
        labels_tensor = torch.cat(all_labels)
        ids = np.array(all_ids)

        torch.save(features_tensor, "pie_features.pt")
        torch.save(labels_tensor, "pie_labels.pt")
        torch.save(ids, "pie_ids.pt")

    print("Features ready:", features_tensor.shape)

    # ------------------------
    # SPLIT BY PEDESTRIAN ID
    # ------------------------
    X = features_tensor
    y = labels_tensor
    ids = np.array(ids)

    # Unique IDs
    unique_ids = np.unique(ids)

    # Shuffle IDs
    np.random.seed(42)
    np.random.shuffle(unique_ids)

    # Split
    split_idx = int(0.8 * len(unique_ids))
    train_ids = set(unique_ids[:split_idx])
    val_ids = set(unique_ids[split_idx:])

    # Create masks
    train_mask = np.array([pid in train_ids for pid in ids])
    val_mask = np.array([pid in val_ids for pid in ids])

    # Apply masks
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    # ------------------------
    # SANITY CHECK
    # ------------------------
    print("Train samples:", len(X_train))
    print("Val samples:", len(X_val))
    print("Unique Train IDs:", len(train_ids))
    print("Unique Val IDs:", len(val_ids))
    print("ID Overlap:", len(train_ids.intersection(val_ids))) 

    # ------------------------
    # DATA LOADERS
    # ------------------------
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False 
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
    # LOAD / TRAIN LOGIC
    # ------------------------
    use_pretrained = False

    if os.path.exists(MODEL_PATH):
        print("\nSaved LSTM weights found!")
        choice = input(
            "Choose an option:\n"
            "1 → Load weights (skip training)\n"
            "2 → Retrain from scratch\n"
            "3 → Resume training\n"
            "Enter choice (1/2/3): "
        )

        if choice == "1":
            lstm.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
            lstm.eval()
            print("Loaded pretrained weights. Skipping training.")
            use_pretrained = True
        elif choice == "3":
            lstm.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
            print("Resuming training from saved weights.")
        else:
            print("Training from scratch.")

    # ------------------------
    # TRAIN LSTM
    # ------------------------
    if not use_pretrained:
        epochs = 30
        best_f1 = 0
        print("Training LSTM")

        for epoch in range(epochs):
            lstm.train()
            running_loss = 0
            all_preds = []
            all_labels_epoch = []

            for features, label in train_loader:
                features = features.to(device)
                label = label.float().unsqueeze(1).to(device)

                logits = lstm(features)
                loss = criterion(logits, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                all_preds.extend(preds.cpu().numpy())
                all_labels_epoch.extend(label.cpu().numpy())

            # Flatten lists for metrics
            all_preds = [p[0] for p in all_preds]
            all_labels_epoch = [l[0] for l in all_labels_epoch]

            acc = accuracy_score(all_labels_epoch, all_preds)
            prec = precision_score(all_labels_epoch, all_preds)
            rec = recall_score(all_labels_epoch, all_preds)
            f1 = f1_score(all_labels_epoch, all_preds)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {running_loss/len(train_loader):.4f} | "
                f"Acc: {acc:.4f} | "
                f"Precision: {prec:.4f} | "
                f"Recall: {rec:.4f} | "
                f"F1: {f1:.4f}"
            )

            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                torch.save(lstm.state_dict(), MODEL_PATH)
                print("🔥 Saved BEST model")

        print(f"Training complete. Best F1: {best_f1:.4f}")

    # ------------------------
    # MC DROPOUT
    # ------------------------
    print("\nRunning MC Dropout...")
    mean_preds, uncertainties, labels, brier = monte_carlo_dropout(
        lstm,
        val_loader,
        device,
        T=30
    )
    print(f"Brier Score: {brier:.4f}")


if __name__ == "__main__":
    main()
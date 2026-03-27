import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import numpy as np 
import gc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datasets.data_loader import PIEDataset
from models.resnet_encoder import ResNetEncoder
from models.lstm import LSTMModel
from models.mc_dropout import monte_carlo_dropout

# Define Constants
MODEL_PATH = "lstm_model.pt"

def main():
    # 1. Clear memory and define device
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------
    # Dataset & Loader
    # ------------------------
    dataset = PIEDataset(
        annotation_file="datasets/pie_annotations_set03.csv",
        video_dir="data/PIE_clips/set03"
    )

    # Lowering num_workers to 2 to prevent CPU choking
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2, 
        pin_memory=True 
    )

    # ------------------------
    # Models
    # ------------------------
    resnet = ResNetEncoder().to(device)
    resnet.eval()
    
    # Initialize LSTM here
    lstm = LSTMModel(feature_dim=512).to(device)

    # ------------------------
    # FEATURE EXTRACTION
    # ------------------------
    if os.path.exists("pie_features.pt"):
        print("Loading cached features...")
        features_tensor = torch.load("pie_features.pt", map_location='cpu', weights_only=True)
        labels_tensor = torch.load("pie_labels.pt", map_location='cpu', weights_only=True)
        ids = torch.load("pie_ids.pt") 
    else:
        print("Extracting ResNet features...")
        all_features = []
        all_labels = []
        all_ids = []

        with torch.no_grad():
            for seq, label, pid in tqdm(loader, desc="Extracting"):
                seq = seq.to(device, non_blocking=True)
                features = resnet(seq)

                # Move to CPU immediately to keep 4GB VRAM free
                all_features.append(features.cpu())
                all_labels.append(label.cpu())

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

    print(f"Features ready: {features_tensor.shape}")

    # ------------------------
    # SPLIT BY PEDESTRIAN ID
    # ------------------------
    unique_ids = np.unique(ids)
    np.random.seed(42)
    np.random.shuffle(unique_ids)

    split_idx = int(0.8 * len(unique_ids))
    train_ids = set(unique_ids[:split_idx])

    # Masking
    train_mask = np.array([pid in train_ids for pid in ids])
    val_mask = ~train_mask

    # Create datasets using views/indexing
    train_dataset = TensorDataset(features_tensor[train_mask], labels_tensor[train_mask])
    val_dataset = TensorDataset(features_tensor[val_mask], labels_tensor[val_mask])

    # CRITICAL: Delete the massive original tensors to drop RAM from 95%
    del features_tensor
    del labels_tensor
    gc.collect()

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # ------------------------
    # Loss + Optimizer
    # ------------------------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.0005)

    # ------------------------
    # LOAD / TRAIN LOGIC
    # ------------------------
    use_pretrained = False
    if os.path.exists(MODEL_PATH):
        print("\nSaved LSTM weights found!")
        choice = input("1: Load | 2: Retrain | 3: Resume -> ")
        if choice == "1":
            lstm.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            lstm.eval()
            use_pretrained = True
        elif choice == "3":
            lstm.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # ------------------------
    # TRAIN LSTM
    # ------------------------
    if not use_pretrained:
        epochs = 30
        best_f1 = 0
        for epoch in range(epochs):
            lstm.train()
            running_loss = 0
            all_preds, all_labels_epoch = [], []

            for f_batch, l_batch in train_loader:
                # Optimized GPU transfer
                f_batch = f_batch.to(device, non_blocking=True)
                l_batch = l_batch.float().unsqueeze(1).to(device, non_blocking=True)

                logits = lstm(f_batch)
                loss = criterion(logits, l_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels_epoch.extend(l_batch.cpu().numpy())

            # Metrics Calculation
            y_true = np.array(all_labels_epoch).flatten()
            y_pred = np.array(all_preds).flatten()
            
            f1 = f1_score(y_true, y_pred, zero_division=0)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | F1: {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                torch.save(lstm.state_dict(), MODEL_PATH)
                print("⭐ Saved Best Model")

    # ------------------------
    # MC DROPOUT
    # ------------------------
    print("\nRunning MC Dropout...")
    # Ensure your monte_carlo_dropout function handles .to(device) internally
    mean_preds, uncertainties, labels, brier = monte_carlo_dropout(
        lstm, val_loader, device, T=30
    )
    print(f"Brier Score: {brier:.4f}")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import numpy as np 
import gc
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from datasets.data_loader import PIEDataset
from models.resnet_encoder import ResNetEncoder
from models.lstm import LSTMModel
from models.mc_dropout import monte_carlo_dropout

os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "lstm_weights.pth"

def get_train_statistics(train_features_tensor):
    
    features = train_features_tensor.mean(dim=1).numpy()
    mean = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False)
    
    cov += np.eye(cov.shape[0]) * 1e-6 
    inv_cov = np.linalg.inv(cov)
    return mean, inv_cov

def main():
    # 1. Clear memory and define device
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

   
    dataset = PIEDataset(
        annotation_file=PROJECT_ROOT / "datasets" / "pie_annotations_set01.csv",
        crop_dir=PROJECT_ROOT / "data" / "PIE_crops"
    )

    
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda"
    )

   
    resnet = ResNetEncoder().to(device)
    resnet.eval()
    
    
    lstm = LSTMModel(feature_dim=512).to(device)

    
    feature_path = PROJECT_ROOT / "pie_features.pt"
    label_path = PROJECT_ROOT / "pie_labels.pt"
    ids_path = PROJECT_ROOT / "pie_ids.pt"
    if feature_path.exists() and label_path.exists():
        print("Loading cached features...")
        features_tensor = torch.load(feature_path, map_location='cpu', weights_only=True)
        labels_tensor = torch.load(label_path, map_location='cpu', weights_only=True)
        if ids_path.exists():
            ids = np.array(torch.load(ids_path, map_location='cpu', weights_only=True))
        else:
            ids = np.array([str(row["pedestrian_id"]) for row in dataset.annotations])
            if len(ids) != len(features_tensor):
                raise RuntimeError("Cached features do not match the annotation file.")
            torch.save(ids.tolist(), ids_path)
    else:
        print("Extracting ResNet features from crops...")
        all_features, all_labels, all_ids = [], [], []

        with torch.no_grad():
            for seq, label, pid in tqdm(loader, desc="Extracting"):
                b, s, c, h, w = seq.shape
                
                seq_reshaped = seq.view(-1, c, h, w).to(device, non_blocking=True)
                
                features = resnet(seq_reshaped)
                
                features = features.view(b, s, -1)

                all_features.append(features.detach().cpu())
                all_labels.append(label.cpu())
                
                if torch.is_tensor(pid):
                    all_ids.extend(pid.tolist())
                elif isinstance(pid, (list, np.ndarray)):
                    all_ids.extend(pid)
                else:
                    all_ids.append(pid)

        features_tensor = torch.cat(all_features)
        labels_tensor = torch.cat(all_labels)
        ids = np.array(all_ids)

        torch.save(features_tensor, feature_path)
        torch.save(labels_tensor, label_path)
        torch.save(ids.tolist(), ids_path)

    if len(features_tensor) == 0:
        raise RuntimeError("No feature sequences were found in the dataset.")

    unique_ids = np.unique(ids)
    np.random.seed(42)
    np.random.shuffle(unique_ids)

    train_end = int(0.7 * len(unique_ids))
    val_end = int(0.85 * len(unique_ids))

    train_ids = set(unique_ids[:train_end])
    val_ids = set(unique_ids[train_end:val_end])
    test_ids = set(unique_ids[val_end:])

    train_mask = np.array([pid in train_ids for pid in ids])
    val_mask = np.array([pid in val_ids for pid in ids])
    test_mask = np.array([pid in test_ids for pid in ids])

    train_dataset = TensorDataset(features_tensor[train_mask], labels_tensor[train_mask])
    val_dataset = TensorDataset(features_tensor[val_mask], labels_tensor[val_mask])
    test_dataset = TensorDataset(features_tensor[test_mask], labels_tensor[test_mask])
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError("The train/validation/test ID split contains an empty dataset.")
    print("\nGenerating Training Statistics for Mahalanobis...")
    train_feat_raw = features_tensor[train_mask] 
    t_mean, t_inv_cov = get_train_statistics(train_feat_raw)
    del features_tensor
    del labels_tensor
    gc.collect()

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ------------------------
    # Loss + Optimizer
    # ------------------------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.0005)

    
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
            print("Resuming training from saved weights...")
            use_pretrained = False



    # ------------------------
    # TRAIN LSTM
    # ------------------------
    if not use_pretrained:
        epochs = 30
        best_val_f1 = 0
        
        for epoch in range(epochs):
            # --- TRAINING PHASE ---
            lstm.train()
            train_loss = 0
            for f_batch, l_batch in train_loader:
                f_batch = f_batch.to(device, non_blocking=True)
                l_batch = l_batch.float().unsqueeze(1).to(device, non_blocking=True)

                logits = lstm(f_batch)
                loss = criterion(logits, l_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            lstm.eval()
            val_loss = 0
            val_preds, val_labels = [], []
            
            with torch.no_grad():
                for f_batch, l_batch in val_loader:
                    f_batch = f_batch.to(device, non_blocking=True)
                    l_batch = l_batch.float().unsqueeze(1).to(device, non_blocking=True)

                    logits = lstm(f_batch)
                    loss = criterion(logits, l_batch)
                    val_loss += loss.item()

                    preds = (torch.sigmoid(logits) > 0.5).float()
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(l_batch.cpu().numpy())
            y_true_val = np.array(val_labels).flatten()
            y_pred_val = np.array(val_preds).flatten()
            val_f1 = f1_score(y_true_val, y_pred_val, zero_division=0)
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")

            # Save model if Validation F1 improves
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(lstm.state_dict(), MODEL_PATH)
                print(f"⭐ New Best Val F1: {val_f1:.4f} - Model Saved")

    # ------------------------
    # MC DROPOUT + UNCERTAINTY
    # ------------------------
    print("\nRunning MC Dropout & Uncertainty Analysis...")
    
    mean_preds, kl_divs, md_dist, test_labels, brier = monte_carlo_dropout(
        lstm, 
        test_loader, 
        device, 
        T=30,
        train_mean=t_mean,
        train_inv_cov=t_inv_cov
    )
    
    print(f"Brier Score: {float(brier):.4f}")
    print(f"Avg KL Divergence (Epistemic): {np.mean(kl_divs):.4f}")
    if md_dist is not None:
        print(f"Avg Mahalanobis Distance (OOD): {np.mean(md_dist):.4f}")
    results_path = PROJECT_ROOT / "final_results.npz"
    np.savez(results_path, probs=mean_preds, kl=kl_divs, md=md_dist, labels=test_labels)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
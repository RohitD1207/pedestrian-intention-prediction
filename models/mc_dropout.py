import torch
import numpy as np
from sklearn.metrics import brier_score_loss
from scipy.stats import entropy


def monte_carlo_dropout(model, dataloader, device, T=30, train_mean=None, train_inv_cov=None):
    model.train() # Enable Dropout
    # Keep Batchnorm in eval mode if you have it
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.eval()

    all_mean_preds, all_kl_div, all_mahalanobis, all_labels = [], [], [], []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            
            t_probs = []
            # T forward passes
            for _ in range(T):
                logits = model(features)
                t_probs.append(torch.sigmoid(logits).cpu().numpy())
            
            t_probs = np.array(t_probs) # Shape: (T, Batch, 1)
            
            # --- 1. Mean Prediction ---
            mean_pred = np.mean(t_probs, axis=0) # (Batch, 1)
            
            # --- 2. KL Divergence (Epistemic Uncertainty) ---
            # Measure how much T runs disagree with the mean
            # Using: KL(Mean || Run_i)
            kl_runs = []
            for t in range(T):
                p = t_probs[t]
                q = mean_pred
                # Binary KL Divergence formula
                kl = p * np.log((p + 1e-10) / (q + 1e-10)) + \
                     (1 - p) * np.log((1 - p + 1e-10) / (1 - q + 1e-10))
                kl_runs.append(kl)
            
            avg_kl = np.mean(kl_runs, axis=0)
            
            # --- 3. Mahalanobis Distance (OOD Detection) ---
            # If you passed train_mean and train_inv_cov, calculate it here
            if train_mean is not None and train_inv_cov is not None:
                # We apply this to the input features (the ResNet output)
                # Or better: the LSTM's final hidden state
                for i in range(features.shape[0]):
                    # Flattening sequence to a single vector for distance
                    feat_vec = features[i].mean(dim=0).cpu().numpy() 
                    diff = feat_vec - train_mean
                    md = np.sqrt(diff.dot(train_inv_cov).dot(diff))
                    all_mahalanobis.append(md)

            all_mean_preds.append(mean_pred.flatten())
            all_kl_div.append(avg_kl.flatten())
            all_labels.append(labels.numpy().flatten())

    # 1. Final Concatenation
    final_means = np.concatenate(all_mean_preds)
    final_labels = np.concatenate(all_labels)
    final_kl = np.concatenate(all_kl_div)
    final_md = np.array(all_mahalanobis) if all_mahalanobis else None

    # 2. Calculate Brier Score (Calibration Metric)
    # This measures how 'accurate' your probabilities are
    brier = brier_score_loss(final_labels, final_means)

    # 3. Return all 5 items
    return final_means, final_kl, final_md, final_labels, brier
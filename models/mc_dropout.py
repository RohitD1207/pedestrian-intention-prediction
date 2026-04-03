import torch
import numpy as np
from sklearn.metrics import brier_score_loss


def enable_mc_dropout(model):
    
    model.train()

    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


def monte_carlo_dropout(model, dataloader, device, T=30):
    enable_mc_dropout(model)

    all_mean_preds = []
    all_uncertainty = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device).float()

            predictions = []

            for _ in range(T):
                logits = model(features)
                probs = torch.sigmoid(logits)
                predictions.append(probs.cpu().numpy())

            predictions = np.array(predictions)  # (T, batch, 1 or batch)

            mean_pred = np.mean(predictions, axis=0)
            uncertainty = np.var(predictions, axis=0)

            # safer accumulation
            all_mean_preds.append(mean_pred)
            all_uncertainty.append(uncertainty)
            all_labels.append(labels.cpu().numpy())

    # concatenate properly
    all_mean_preds = np.concatenate(all_mean_preds, axis=0).reshape(-1)
    all_uncertainty = np.concatenate(all_uncertainty, axis=0).reshape(-1)
    all_labels = np.concatenate(all_labels, axis=0).reshape(-1)

    # Brier score (calibration metric)
    brier = brier_score_loss(all_labels, all_mean_preds)

    return all_mean_preds, all_uncertainty, all_labels, brier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc)
from sklearn.calibration import calibration_curve

# Load the data
data = np.load("final_results.npz")

# Extract the variables
y_probs = data['probs']
kl_scores = data['kl']
md_scores = data['md']
y_true = data['labels']

def generate_visual_report(y_true, y_probs, kl_scores, md_scores):
    # 1. Standard Metrics
    y_pred = (y_probs > 0.5).astype(int)
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred)
    }
    
    print("--- Standard Performance Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 2. Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix: Crossing Intent")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig("results/confusion_matrix.png")

    # 3. Uncertainty Distribution Plot (KL vs Prediction)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_probs, kl_scores, c=y_true, cmap='coolwarm', alpha=0.5)
    plt.axvline(0.5, color='black', linestyle='--')
    plt.title("Epistemic Uncertainty (KL) vs. Prediction Confidence")
    plt.xlabel("Predicted Probability (0=Wait, 1=Cross)")
    plt.ylabel("KL Divergence (Model Confusion)")
    plt.colorbar(label="Actual Label")
    plt.savefig("results/kl_vs_confidence.png")

    # 4. Mahalanobis 'OOD' Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(md_scores, kde=True, color="purple")
    plt.title("Distribution of Mahalanobis Distance (OOD Scores)")
    plt.xlabel("Distance from 'Normal' Pedestrian Training Distribution")
    plt.savefig("results/mahalanobis_distribution.png")

    # 5. Reliability Diagram (Calibration Plot)
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid()
    plt.savefig("results/reliability_diagram.png")

    plt.show()

generate_visual_report(y_true, y_probs, kl_scores, md_scores)

def plot_filtered_reliability(y_true, y_probs, md_scores, threshold=50):
    # 1. Create a mask for 'Trusted' predictions (Low Mahalanobis)
    trusted_mask = md_scores < threshold
    
    # 2. Filter the data
    y_true_filtered = y_true[trusted_mask]
    y_probs_filtered = y_probs[trusted_mask]
    
    # 3. Plotting logic
    plt.figure(figsize=(10, 6))
    
    # Original (from your current graph)
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label="Original Model")
    
    # Filtered (The 'Fixed' version)
    prob_true_filtered, prob_pred_filtered = calibration_curve(y_true_filtered, y_probs_filtered, n_bins=10)
    plt.plot(prob_pred_filtered, prob_true_filtered, marker='o', label=f"Filtered (MD < {threshold})")
    
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.title("Impact of Mahalanobis Filtering on Model Reliability")
    plt.legend()
    plt.savefig("results/reliability_improvement.png")
    plt.show()

plot_filtered_reliability(y_true, y_probs, md_scores, threshold=30)

def calculate_ece(y_true, y_probs, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        # Find samples in this specific confidence bin
        bin_mask = (y_probs > bin_boundaries[i]) & (y_probs <= bin_boundaries[i+1])
        if np.sum(bin_mask) > 0:
            # Accuracy in this bin
            bin_acc = np.mean(y_true[bin_mask])
            # Average confidence in this bin
            bin_conf = np.mean(y_probs[bin_mask])
            # Weighted difference
            ece += np.abs(bin_acc - bin_conf) * (np.sum(bin_mask) / len(y_true))
    return ece

threshold = 30
mask = data['md'] < threshold
y_true_filtered = data['labels'][mask]
y_probs_filtered = data['probs'][mask]

# Now run it on your data
original_ece = calculate_ece(y_true, y_probs)
filtered_ece = calculate_ece(y_true_filtered, y_probs_filtered)

print(f"Original Model ECE: {original_ece:.4f}")
print(f"Filtered (MD < 30) ECE: {filtered_ece:.4f}")
print(f"Improvement: {((original_ece - filtered_ece) / original_ece) * 100:.2f}%")


# 1. Define masks (Same as before)
y_pred = (y_probs > 0.5).astype(int)
fn_mask = (y_pred == 0) & (y_true == 1)
tn_mask = (y_pred == 0) & (y_true == 0)

# 2. Extract Data
fn_conf = 1 - y_probs[fn_mask] # Confidence in 'Not Crossing'
tn_conf = 1 - y_probs[tn_mask]
fn_md = md_scores[fn_mask] # Mahalanobis Distance
tn_md = md_scores[tn_mask]

print(f"--- Safety Analysis ---")
print(f"Avg Confidence for Correct 'Stay' (TN): {np.mean(tn_conf):.4f}")
print(f"Avg Confidence for Dangerous Mistake (FN): {np.mean(fn_conf):.4f}")
# Check if the Mahalanobis Distance is higher for False Negatives than True Negatives
avg_md_fn = np.mean(md_scores[fn_mask])
avg_md_tn = np.mean(md_scores[tn_mask])

print(f"Avg MD for Correct 'Stay' (TN): {avg_md_tn:.4f}")
print(f"Avg MD for Dangerous Mistake (FN): {avg_md_fn:.4f}")

# --- START PLOTTING ---
# Create a figure with two subplots side-by-side
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
plt.subplots_adjust(wspace=0.3) # Space between plots

# A. SUBPLOT 1: CONFIDENCE (THE FAILED SAFETY METRIC)
sns.kdeplot(ax=axes[0], data=tn_conf, label="True Negatives (Safe)", fill=True, color="#1f77b4", alpha=0.6)
sns.kdeplot(ax=axes[0], data=fn_conf, label="False Negatives (Hazardous)", fill=True, color="#ff7f0e", alpha=0.6)
axes[0].set_title("Probability is a Deceptive Safety Metric", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Confidence in 'Not Crossing'", fontsize=12)
axes[0].set_ylabel("Density (Sample Count)", fontsize=12)
# Since they overlap at 0.94 and 0.99, let's zoom in on that zone
axes[0].set_xlim(0.8, 1.0) 

# B. SUBPLOT 2: MAHALANOBIS (THE SUCCESSFUL SAFETY METRIC)
# Use a slight hue shift to make the orange 'jump' (hazardous)
sns.kdeplot(ax=axes[1], data=tn_md, label="True Negatives (Safe)", fill=True, color="#1f77b4", alpha=0.6)
sns.kdeplot(ax=axes[1], data=fn_md, label="False Negatives (Hazardous)", fill=True, color="#d62728", alpha=0.6)
axes[1].axvline(avg_md_tn, color='#1f77b4', linestyle='--')
axes[1].axvline(avg_md_fn, color='#d62728', linestyle='--')
axes[1].set_title("Mahalanobis Distance Identifies the Hazards", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Mahalanobis Distance (OOD Score)", fontsize=12)
# Distance has no limit, but most are under 60
axes[1].set_xlim(10, 60) 

# --- FINAL POLISHING ---
# Add one unified legend to the bottom of the whole figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.05))

# Use tight_layout to make sure text doesn't overlap
plt.tight_layout()
plt.savefig("results/integrated_safety_analysis.png", bbox_inches='tight', dpi=300)
plt.show()

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def calculate_tradeoff(y_true, y_probs, md_scores, threshold=30):
    # Original Predictions (Full Dataset)
    y_pred_orig = (y_probs > 0.5).astype(int)
    
    # Filtered Predictions (Only 'Trusted' Samples)
    mask = md_scores < threshold
    y_true_filt = y_true[mask]
    y_probs_filt = y_probs[mask]
    y_pred_filt = (y_probs_filt > 0.5).astype(int)
    
    # Calculate % of data kept
    coverage = (np.sum(mask) / len(y_true)) * 100
    
    print(f"--- Trade-off Analysis (Threshold MD < {threshold}) ---")
    print(f"Data Coverage: {coverage:.2f}% (Discarded {100-coverage:.2f}% as 'Too Risky')")
    print("-" * 30)
    print(f"Metric    | Original | Filtered")
    print(f"Accuracy  | {accuracy_score(y_true, y_pred_orig):.4f}   | {accuracy_score(y_true_filt, y_pred_filt):.4f}")
    print(f"F1-Score  | {f1_score(y_true, y_pred_orig):.4f}   | {f1_score(y_true_filt, y_pred_filt):.4f}")
    print(f"Precision | {precision_score(y_true, y_pred_orig):.4f}   | {precision_score(y_true_filt, y_pred_filt):.4f}")
    print(f"Recall    | {recall_score(y_true, y_pred_orig):.4f}   | {recall_score(y_true_filt, y_pred_filt):.4f}")

calculate_tradeoff(y_true, y_probs, md_scores, threshold=30)
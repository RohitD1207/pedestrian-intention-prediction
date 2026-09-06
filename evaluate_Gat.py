import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
from sklearn.calibration import calibration_curve

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.resnet_encoder import ResNetEncoder
from datasets.data_loader import PIEDataset
from models.pose_extracter import PoseExtractor
from models.graph_builder import build_graph
from models.Gat import GATModel


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "gat_best.pth"

RESULTS_DIR = PROJECT_ROOT / "results" / "gat"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 16
IMAGE_SIZE = 320
MC_SAMPLES = 30

parser = argparse.ArgumentParser(description="Evaluate the pedestrian-intention GAT model.")
parser.add_argument("--max-samples", type=int, default=0)
parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
parser.add_argument("--mc-samples", type=int, default=MC_SAMPLES)
args = parser.parse_args()

SEQUENCE_LENGTH = args.sequence_length
IMAGE_SIZE = args.imgsz
MC_SAMPLES = args.mc_samples


def extract_gat_features(model, x, edge_index):
    x = model.gat1(x, edge_index)
    x = F.elu(x)

    x = model.dropout(x)

    x = model.gat2(x, edge_index)
    x = F.elu(x)

    x = x.mean(dim=0, keepdim=True)

    return x


def prepare_graph(
    sequence,
    pose_extractor,
    resnet,
    resnet_projection
):
    sequence = sequence.to(DEVICE)

    pose = pose_extractor.extract(sequence)

    with torch.no_grad():
        visual_features = resnet(sequence)

    visual_features = visual_features.squeeze(-1).squeeze(-1)

    visual_features = resnet_projection(
        visual_features
    )

    visual_features = visual_features.unsqueeze(1).expand(
        -1,
        17,
        -1
    )

    fused_features = torch.cat(
        [
            pose.to(DEVICE),
            visual_features
        ],
        dim=-1
    )

    node_features, edge_index = build_graph(
        fused_features
    )

    node_features = node_features.to(DEVICE)
    edge_index = edge_index.to(DEVICE)

    return node_features, edge_index


def get_predictions(
    model,
    resnet,
    resnet_projection,
    pose_extractor,
    loader
):
    model.eval()

    labels = []
    probabilities = []
    features = []

    with torch.no_grad():

        for sequence, target, _ in loader:

            label = target.long().to(DEVICE)

            node_features, edge_index = prepare_graph(
                sequence[0],
                pose_extractor,
                resnet,
                resnet_projection
            )

            output = model(
                node_features,
                edge_index
            )

            probability = torch.softmax(
                output,
                dim=1
            )[:, 1]

            embedding = extract_gat_features(
                model,
                node_features,
                edge_index
            )

            labels.append(
                label.cpu().numpy()
            )

            probabilities.append(
                probability.cpu().numpy()
            )

            features.append(
                embedding.cpu().numpy()
            )

    labels = np.concatenate(labels)
    probabilities = np.concatenate(probabilities)
    features = np.concatenate(features)

    return labels, probabilities, features


def get_mc_predictions(
    model,
    resnet,
    resnet_projection,
    pose_extractor,
    loader,
    samples=30
):
    all_predictions = []

    model.train()

    for _ in range(samples):

        current_predictions = []

        with torch.no_grad():

            for sequence, _, _ in loader:

                node_features, edge_index = prepare_graph(
                    sequence[0],
                    pose_extractor,
                    resnet,
                    resnet_projection
                )

                output = model(
                    node_features,
                    edge_index
                )

                probability = torch.softmax(
                    output,
                    dim=1
                )[:, 1]

                current_predictions.append(
                    probability.cpu().numpy()
                )

        current_predictions = np.concatenate(
            current_predictions
        )

        all_predictions.append(
            current_predictions
        )

    all_predictions = np.asarray(
        all_predictions
    )

    mean_predictions = np.mean(
        all_predictions,
        axis=0
    )

    return all_predictions, mean_predictions


def calculate_kl_uncertainty(mc_predictions):

    eps = 1e-8

    p = np.clip(
        mc_predictions,
        eps,
        1 - eps
    )

    mean_p = np.mean(
        p,
        axis=0
    )

    mean_p = np.clip(
        mean_p,
        eps,
        1 - eps
    )

    kl_values = []

    for i in range(p.shape[0]):

        p_i = p[i]

        kl = (
            p_i * np.log(p_i / mean_p)
            +
            (1 - p_i)
            *
            np.log(
                (1 - p_i)
                /
                (1 - mean_p)
            )
        )

        kl_values.append(kl)

    kl_values = np.asarray(
        kl_values
    )

    return np.mean(
        kl_values,
        axis=0
    )


def calculate_mahalanobis(
    train_features,
    test_features
):

    mean_vector = np.mean(
        train_features,
        axis=0
    )

    feature_count = train_features.shape[1]
    if len(train_features) < 2:
        covariance = np.eye(feature_count)
    else:
        covariance = np.atleast_2d(
            np.cov(train_features, rowvar=False)
        )
        covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
        covariance += np.eye(feature_count) * 1e-6

    inverse_covariance = np.linalg.pinv(
        covariance
    )

    differences = (
        test_features
        -
        mean_vector
    )

    distances = np.sqrt(
        np.maximum(
            0,
            np.einsum(
                "ij,jk,ik->i",
                differences,
                inverse_covariance,
                differences
            )
        )
    )

    return distances


def save_confusion_matrix(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(
        figsize=(8, 6)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(
        "GAT Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "gat_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def save_kl_plot(
    probabilities,
    kl_scores,
    y_true
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.scatter(
        probabilities,
        kl_scores,
        c=y_true,
        cmap="coolwarm",
        alpha=0.5
    )

    plt.axvline(
        0.5,
        color="black",
        linestyle="--"
    )

    plt.xlabel(
        "Predicted Probability of Crossing"
    )

    plt.ylabel(
        "KL Divergence"
    )

    plt.title(
        "GAT Epistemic Uncertainty vs Prediction Confidence"
    )

    plt.colorbar(
        label="Actual Label"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "gat_kl_vs_confidence.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def save_mahalanobis_plot(
    md_scores
):

    plt.figure(
        figsize=(10, 6)
    )

    sns.histplot(
        md_scores,
        kde=True
    )

    plt.xlabel(
        "Mahalanobis Distance"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "GAT Mahalanobis Distance Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "gat_mahalanobis_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def save_reliability_plot(
    y_true,
    probabilities
):

    plt.figure(
        figsize=(8, 6)
    )

    if len(np.unique(y_true)) >= 2:
        prob_true, prob_pred = calibration_curve(
            y_true,
            probabilities,
            n_bins=10
        )
        plt.plot(prob_pred, prob_true, marker="o", label="GAT")

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration"
    )

    plt.xlabel(
        "Predicted Probability"
    )

    plt.ylabel(
        "Actual Accuracy"
    )

    plt.title(
        "GAT Reliability Diagram"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "gat_reliability_diagram.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def calculate_ece(
    y_true,
    probabilities,
    n_bins=10
):

    boundaries = np.linspace(
        0,
        1,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        if i == 0:
            mask = (
                probabilities >= boundaries[i]
            ) & (
                probabilities <= boundaries[i + 1]
            )
        else:
            mask = (
                probabilities > boundaries[i]
            ) & (
                probabilities <= boundaries[i + 1]
            )

        if np.sum(mask) == 0:
            continue

        bin_accuracy = np.mean(
            y_true[mask]
        )

        bin_confidence = np.mean(
            probabilities[mask]
        )

        bin_weight = (
            np.sum(mask)
            /
            len(y_true)
        )

        ece += (
            abs(
                bin_accuracy
                -
                bin_confidence
            )
            *
            bin_weight
        )

    return ece


def save_filtered_reliability(
    y_true,
    probabilities,
    md_scores,
    threshold
):

    mask = md_scores < threshold

    filtered_true = y_true[mask]
    filtered_probabilities = probabilities[mask]

    plt.figure(
        figsize=(10, 6)
    )

    if len(np.unique(y_true)) >= 2:
        prob_true, prob_pred = calibration_curve(
            y_true,
            probabilities,
            n_bins=10
        )
        plt.plot(prob_pred, prob_true, marker="o", label="Original GAT")

    if len(np.unique(filtered_true)) >= 2:

        prob_true_filtered, prob_pred_filtered = calibration_curve(
            filtered_true,
            filtered_probabilities,
            n_bins=10
        )

        plt.plot(
            prob_pred_filtered,
            prob_true_filtered,
            marker="o",
            label=f"Filtered MD < {threshold:.2f}"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Perfect Calibration"
    )

    plt.xlabel(
        "Predicted Probability"
    )

    plt.ylabel(
        "Actual Accuracy"
    )

    plt.title(
        "Impact of Mahalanobis Filtering on GAT Reliability"
    )

    plt.legend()

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "gat_reliability_improvement.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def save_safety_analysis(
    y_true,
    probabilities,
    md_scores
):

    y_pred = (
        probabilities >= 0.5
    ).astype(int)

    fn_mask = (
        (y_pred == 0)
        &
        (y_true == 1)
    )

    tn_mask = (
        (y_pred == 0)
        &
        (y_true == 0)
    )

    fn_conf = (
        1
        -
        probabilities[fn_mask]
    )

    tn_conf = (
        1
        -
        probabilities[tn_mask]
    )

    fn_md = md_scores[fn_mask]
    tn_md = md_scores[tn_mask]

    print("\n--- Safety Analysis ---")

    if len(tn_conf) > 0:
        print(
            f"TN confidence: "
            f"{np.mean(tn_conf):.4f}"
        )

    if len(fn_conf) > 0:
        print(
            f"FN confidence: "
            f"{np.mean(fn_conf):.4f}"
        )

    if len(tn_md) > 0:
        print(
            f"TN Mahalanobis: "
            f"{np.mean(tn_md):.4f}"
        )

    if len(fn_md) > 0:
        print(
            f"FN Mahalanobis: "
            f"{np.mean(fn_md):.4f}"
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6)
    )

    if len(tn_conf) > 1:
        sns.kdeplot(
            ax=axes[0],
            data=tn_conf,
            fill=True,
            label="True Negatives"
        )

    if len(fn_conf) > 1:
        sns.kdeplot(
            ax=axes[0],
            data=fn_conf,
            fill=True,
            label="False Negatives"
        )

    axes[0].set_title(
        "Confidence Distribution"
    )

    axes[0].set_xlabel(
        "Confidence in Not Crossing"
    )

    axes[0].set_xlim(
        0.8,
        1.0
    )

    axes[0].legend()

    if len(tn_md) > 1:
        sns.kdeplot(
            ax=axes[1],
            data=tn_md,
            fill=True,
            label="True Negatives"
        )

    if len(fn_md) > 1:
        sns.kdeplot(
            ax=axes[1],
            data=fn_md,
            fill=True,
            label="False Negatives"
        )

    axes[1].set_title(
        "Mahalanobis Distance Distribution"
    )

    axes[1].set_xlabel(
        "Mahalanobis Distance"
    )

    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "gat_integrated_safety_analysis.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def save_tradeoff(
    y_true,
    probabilities,
    md_scores,
    threshold
):

    mask = md_scores < threshold

    if np.sum(mask) == 0:
        return

    filtered_true = y_true[mask]
    filtered_probabilities = probabilities[mask]

    original_pred = (
        probabilities >= 0.5
    ).astype(int)

    filtered_pred = (
        filtered_probabilities >= 0.5
    ).astype(int)

    original_accuracy = accuracy_score(
        y_true,
        original_pred
    )

    filtered_accuracy = accuracy_score(
        filtered_true,
        filtered_pred
    )

    original_f1 = f1_score(
        y_true,
        original_pred,
        zero_division=0
    )

    filtered_f1 = f1_score(
        filtered_true,
        filtered_pred,
        zero_division=0
    )

    coverage = (
        np.sum(mask)
        /
        len(y_true)
    )

    print("\n--- GAT Trade-off Analysis ---")

    print(
        f"MD threshold: {threshold:.4f}"
    )

    print(
        f"Coverage: {coverage * 100:.2f}%"
    )

    print(
        f"Original Accuracy: "
        f"{original_accuracy:.4f}"
    )

    print(
        f"Filtered Accuracy: "
        f"{filtered_accuracy:.4f}"
    )

    print(
        f"Original F1: "
        f"{original_f1:.4f}"
    )

    print(
        f"Filtered F1: "
        f"{filtered_f1:.4f}"
    )


def main():

    print(
        f"Device: {DEVICE}"
    )

    print(
        "Loading GAT checkpoint..."
    )

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"GAT checkpoint not found at {CHECKPOINT_PATH}. "
            "Train first with: python models/train_gat.py"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    if (
        not isinstance(checkpoint, dict)
        or "gat" not in checkpoint
        or "resnet_projection" not in checkpoint
    ):
        raise RuntimeError(
            "Incompatible legacy checkpoint. Retrain with: "
            "python models/train_gat.py"
        )

    resnet = ResNetEncoder().to(
        DEVICE
    )

    resnet.eval()

    resnet_projection = torch.nn.Linear(
        512,
        64
    ).to(DEVICE)

    resnet_projection.load_state_dict(
        checkpoint["resnet_projection"]
    )

    model = GATModel(
        in_channels=67
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["gat"]
    )

    pose_extractor = PoseExtractor(
        model_name=str(
            PROJECT_ROOT /
            "yolo11n-pose.pt"
        ),
        device=DEVICE,
        image_size=IMAGE_SIZE
    )

    test_dataset = PIEDataset(
        annotation_file=(
            PROJECT_ROOT /
            "datasets" /
            "pie_annotations_set03.csv"
        ),
        crop_dir=(
            PROJECT_ROOT /
            "data" /
            "PIE_crops"
        ),
        sequence_length=SEQUENCE_LENGTH
    )

    test_dataset_for_eval = test_dataset
    if args.max_samples > 0:
        test_dataset_for_eval = torch.utils.data.Subset(
            test_dataset,
            range(min(args.max_samples, len(test_dataset)))
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset_for_eval,
        batch_size=1,
        shuffle=False
    )

    print(
        f"Evaluation samples: "
        f"{len(test_dataset_for_eval)}"
    )

    print(
        "\nGenerating standard predictions..."
    )

    y_true, y_probs, test_features = get_predictions(
        model,
        resnet,
        resnet_projection,
        pose_extractor,
        test_loader
    )

    y_pred = (
        y_probs >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    brier = np.mean(
        (
            y_probs
            -
            y_true
        ) ** 2
    )

    print("\n--- GAT Standard Metrics ---")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-Score : {f1:.4f}"
    )

    print(
        f"Brier    : {brier:.4f}"
    )

    save_confusion_matrix(
        y_true,
        y_pred
    )

    save_reliability_plot(
        y_true,
        y_probs
    )

    print(
        "\nRunning MC Dropout..."
    )

    mc_predictions, mc_mean = get_mc_predictions(
        model,
        resnet,
        resnet_projection,
        pose_extractor,
        test_loader,
        samples=MC_SAMPLES
    )

    kl_scores = calculate_kl_uncertainty(
        mc_predictions
    )

    kl_auc = (
        roc_auc_score(y_true, kl_scores)
        if len(np.unique(y_true)) >= 2
        else float("nan")
    )

    print(
        f"KL AUROC: {kl_auc:.4f}"
    )

    save_kl_plot(
        mc_mean,
        kl_scores,
        y_true
    )

    print(
        "\nGenerating training feature distribution..."
    )

    train_dataset = PIEDataset(
        annotation_file=(
            PROJECT_ROOT /
            "datasets" /
            "pie_annotations_set01.csv"
        ),
        crop_dir=(
            PROJECT_ROOT /
            "data" /
            "PIE_crops"
        ),
        sequence_length=SEQUENCE_LENGTH
    )

    train_dataset_for_eval = train_dataset
    if args.max_samples > 0:
        train_dataset_for_eval = torch.utils.data.Subset(
            train_dataset,
            range(min(args.max_samples, len(train_dataset)))
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset_for_eval,
        batch_size=1,
        shuffle=False
    )

    _, _, train_features = get_predictions(
        model,
        resnet,
        resnet_projection,
        pose_extractor,
        train_loader
    )

    print(
        "\nCalculating Mahalanobis distance..."
    )

    md_scores = calculate_mahalanobis(
        train_features,
        test_features
    )

    save_mahalanobis_plot(
        md_scores
    )

    md_threshold = np.percentile(
        md_scores,
        80
    )

    print(
        f"MD threshold (80th percentile): "
        f"{md_threshold:.4f}"
    )

    save_filtered_reliability(
        y_true,
        y_probs,
        md_scores,
        md_threshold
    )

    original_ece = calculate_ece(
        y_true,
        y_probs
    )

    trusted_mask = (
        md_scores
        <
        md_threshold
    )

    filtered_ece = calculate_ece(
        y_true[trusted_mask],
        y_probs[trusted_mask]
    )

    print(
        f"\nOriginal ECE : "
        f"{original_ece:.4f}"
    )

    print(
        f"Filtered ECE: "
        f"{filtered_ece:.4f}"
    )

    if original_ece > 0:

        improvement = (
            (
                original_ece
                -
                filtered_ece
            )
            /
            original_ece
        ) * 100

        print(
            f"ECE Improvement: "
            f"{improvement:.2f}%"
        )

    save_safety_analysis(
        y_true,
        y_probs,
        md_scores
    )

    save_tradeoff(
        y_true,
        y_probs,
        md_scores,
        md_threshold
    )

    np.savez(
        RESULTS_DIR / "gat_results.npz",
        labels=y_true,
        probs=y_probs,
        kl=kl_scores,
        md=md_scores,
        features=test_features,
        brier=brier,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        kl_auc=kl_auc,
        md_threshold=md_threshold
    )

    print(
        "\n================================"
    )

    print(
        "GAT evaluation completed."
    )

    print(
        f"Results saved to:"
    )

    print(
        RESULTS_DIR
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()
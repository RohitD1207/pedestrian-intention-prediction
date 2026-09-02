import torch
import numpy as np
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from datasets.data_loader import PIEDataset
from models.pose_extracter import PoseExtractor
from models.graph_builder import build_graph
from models.Gat import GATModel


BATCH_SIZE = 1
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "gat_best.pth"

parser = argparse.ArgumentParser(description="Evaluate the pedestrian-intention GAT model.")
parser.add_argument("--max-samples", type=int, default=0)
parser.add_argument("--sequence-length", type=int, default=16)
parser.add_argument("--imgsz", type=int, default=320)
args = parser.parse_args()

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"GAT checkpoint not found at {MODEL_PATH}. "
        "Train the model first with: python models/train_gat.py"
    )

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


test_dataset = PIEDataset(
    annotation_file=PROJECT_ROOT / "datasets" / "pie_annotations_set03.csv",
    crop_dir=PROJECT_ROOT / "data" / "PIE_crops",
    sequence_length=args.sequence_length
)

test_loader = DataLoader(
    test_dataset if args.max_samples <= 0 else torch.utils.data.Subset(
        test_dataset,
        range(min(args.max_samples, len(test_dataset)))
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)

pose_extractor = PoseExtractor(
    model_name=str(PROJECT_ROOT / "yolo11n-pose.pt"),
    device=DEVICE,
    image_size=args.imgsz
)

model = GATModel()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


all_labels = []
all_predictions = []


with torch.no_grad():

    for frames, labels, _ in test_loader:

        labels = labels.to(DEVICE)

        pose = pose_extractor.extract(frames[0])

        node_features, edge_index = build_graph(pose)
        node_features = node_features.to(DEVICE)
        edge_index = edge_index.to(DEVICE)

        output = model(node_features, edge_index)

        predictions = torch.argmax(
            output,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="binary",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="binary",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="binary",
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)


print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        zero_division=0
    )
)


results = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "confusion_matrix": cm.tolist()
}

np.save(
    "gat_results.npy",
    np.array(results, dtype=object),
    allow_pickle=True
)
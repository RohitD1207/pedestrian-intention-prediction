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
from models.resnet_encoder import ResNetEncoder


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

resnet = ResNetEncoder().to(DEVICE)
resnet.eval()
resnet_projection = torch.nn.Linear(512, 64).to(DEVICE)

model = GATModel(in_channels=67).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

gat_state = checkpoint.get("gat", checkpoint) if isinstance(checkpoint, dict) else checkpoint
projection_state = checkpoint.get("resnet_projection") if isinstance(checkpoint, dict) else None

try:
    model.load_state_dict(gat_state)
    if projection_state is None:
        raise RuntimeError(
            "The checkpoint does not contain resnet_projection weights. "
            "Retrain the GAT model with: python models/train_gat.py"
        )
    resnet_projection.load_state_dict(projection_state)
except RuntimeError as exc:
    raise RuntimeError(
        f"Incompatible GAT checkpoint at {MODEL_PATH}. "
        "Retrain the model with: python models/train_gat.py"
    ) from exc

model.eval()
resnet_projection.eval()


all_labels = []
all_predictions = []


with torch.no_grad():

    for frames, labels, _ in test_loader:

        labels = labels.to(DEVICE)

        pose = pose_extractor.extract(frames[0])

        with torch.no_grad():
            visual_features = resnet(frames[0].to(DEVICE))
            visual_features = visual_features.squeeze(-1).squeeze(-1)
            visual_features = resnet_projection(visual_features)
            visual_features = visual_features.unsqueeze(1).expand(-1, 17, -1)

        fused_features = torch.cat(
            [pose.to(DEVICE), visual_features],
            dim=-1
        )

        node_features, edge_index = build_graph(fused_features)
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
    PROJECT_ROOT / "gat_results.npy",
    np.array(results, dtype=object),
    allow_pickle=True
)
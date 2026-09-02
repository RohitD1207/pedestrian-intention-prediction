import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets.data_loader import PIEDataset
from models.pose_extracter import PoseExtractor
from models.graph_builder import build_graph
from models.Gat import GATModel


BATCH_SIZE = 1
DEFAULT_EPOCHS = 30
LEARNING_RATE = 0.001

parser = argparse.ArgumentParser(description="Train the pedestrian-intention GAT model.")
parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
parser.add_argument("--max-samples", type=int, default=0)
parser.add_argument("--sequence-length", type=int, default=16)
parser.add_argument("--imgsz", type=int, default=320)
args = parser.parse_args()
EPOCHS = args.epochs

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = PROJECT_ROOT / "checkpoints" / "gat_best.pth"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


train_dataset = PIEDataset(
    annotation_file=PROJECT_ROOT / "datasets" / "pie_annotations_set01.csv",
    crop_dir=PROJECT_ROOT / "data" / "PIE_crops",
    sequence_length=args.sequence_length
)

val_dataset = PIEDataset(
    annotation_file=PROJECT_ROOT / "datasets" / "pie_annotations_set03.csv",
    crop_dir=PROJECT_ROOT / "data" / "PIE_crops",
    sequence_length=args.sequence_length
)

if args.max_samples > 0:
    train_dataset = torch.utils.data.Subset(train_dataset, range(min(args.max_samples, len(train_dataset))))
    val_dataset = torch.utils.data.Subset(val_dataset, range(min(args.max_samples, len(val_dataset))))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


pose_extractor = PoseExtractor(
    model_name=str(PROJECT_ROOT / "yolo11n-pose.pt"),
    device=DEVICE,
    image_size=args.imgsz
)


model = GATModel().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


best_val_accuracy = -1.0


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for sequence, labels, _ in train_loader:

        label = labels.long().to(DEVICE)

        pose = pose_extractor.extract(sequence[0])

        node_features, edge_index = build_graph(pose)

        node_features = node_features.to(DEVICE)
        edge_index = edge_index.to(DEVICE)

        output = model(
            node_features,
            edge_index
        )

        loss = criterion(
            output,
            label
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        prediction = torch.argmax(output, dim=1)

        correct += (
            prediction == label
        ).sum().item()

        total += label.numel()


    train_accuracy = correct / total


    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for sequence, labels, _ in val_loader:

            label = labels.long().to(DEVICE)

            pose = pose_extractor.extract(sequence[0])

            node_features, edge_index = build_graph(pose)

            node_features = node_features.to(DEVICE)
            edge_index = edge_index.to(DEVICE)

            output = model(
                node_features,
                edge_index
            )

            prediction = torch.argmax(output, dim=1)

            val_correct += (
                prediction == label
            ).sum().item()

            val_total += label.numel()


    val_accuracy = val_correct / val_total

    average_loss = total_loss / len(train_loader)


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )


    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


print("\nTraining completed.")
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)
import os
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


# ============================================================
# 1. CONFIGURATION
# ============================================================

BATCH_SIZE = 1
EPOCHS = 30
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = PROJECT_ROOT / "checkpoints" / "gat_best.pth"

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DATASET
# ============================================================

train_dataset = PIEDataset(
    annotation_file=PROJECT_ROOT / "datasets" / "pie_annotations_set01.csv",
    crop_dir=PROJECT_ROOT / "data" / "PIE_crops"
)

val_dataset = PIEDataset(
    annotation_file=PROJECT_ROOT / "datasets" / "pie_annotations_set03.csv",
    crop_dir=PROJECT_ROOT / "data" / "PIE_crops"
)


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


# ============================================================
# 3. POSE EXTRACTOR
# ============================================================

pose_extractor = PoseExtractor(
    model_name=str(PROJECT_ROOT / "yolo11n-pose.pt")
)


# ============================================================
# 4. GAT MODEL
# ============================================================

model = GATModel()

model = model.to(DEVICE)


# ============================================================
# 5. LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 6. TRAINING
# ============================================================

best_val_accuracy = -1.0


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for sequence, labels, _ in train_loader:

        # ----------------------------------------------------
        # Get data
        # ----------------------------------------------------

        labels = labels.to(DEVICE)


        # ----------------------------------------------------
        # Pose extraction
        # ----------------------------------------------------

        pose = pose_extractor.extract(sequence[0])


        # ----------------------------------------------------
        # Build graph
        # ----------------------------------------------------

        node_features, edge_index = build_graph(pose)


        # ----------------------------------------------------
        # Move graph to GPU
        # ----------------------------------------------------

        node_features = node_features.to(DEVICE)
        edge_index = edge_index.to(DEVICE)


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        output = model(node_features, edge_index)


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(output, labels)


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += loss.item()

        predictions = torch.argmax(output, dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)


    train_accuracy = correct / total


    # ========================================================
    # 7. VALIDATION
    # ========================================================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for sequence, labels, _ in val_loader:

            labels = labels.to(DEVICE)

            pose = pose_extractor.extract(sequence[0])

            node_features, edge_index = build_graph(pose)

            node_features = node_features.to(DEVICE)
            edge_index = edge_index.to(DEVICE)

            output = model(node_features, edge_index)

            predictions = torch.argmax(
                output,
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)


    val_accuracy = val_correct / val_total


    # ========================================================
    # 8. PRINT RESULTS
    # ========================================================

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )


    # ========================================================
    # 9. SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"Best GAT model saved → {MODEL_PATH}"
        )


print("\nTraining completed.")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)
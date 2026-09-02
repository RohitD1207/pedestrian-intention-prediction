import os
import gc
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from datasets.data_loader import PIEDataset
from models.pose_extracter import PoseExtractor
from models.graph_builder import build_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANNOTATION_FILE = PROJECT_ROOT / "datasets" / "pie_annotations_set01.csv"
CROP_DIR = PROJECT_ROOT / "data" / "PIE_crops"

CACHE_DIR = PROJECT_ROOT / "graph_cache"

SEQUENCE_LENGTH = 16


def create_split(dataset):

    ids = np.array([
        str(row["pedestrian_id"])
        for row in dataset.annotations
    ])

    unique_ids = np.unique(ids)

    np.random.seed(42)
    np.random.shuffle(unique_ids)

    train_end = int(0.7 * len(unique_ids))
    val_end = int(0.85 * len(unique_ids))

    train_ids = set(unique_ids[:train_end])
    val_ids = set(unique_ids[train_end:val_end])
    test_ids = set(unique_ids[val_end:])

    train_indices = []
    val_indices = []
    test_indices = []

    for i, pid in enumerate(ids):

        if pid in train_ids:
            train_indices.append(i)

        elif pid in val_ids:
            val_indices.append(i)

        elif pid in test_ids:
            test_indices.append(i)

    return train_indices, val_indices, test_indices


def save_graphs(dataset, indices, split_name, pose_model):

    split_dir = CACHE_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCreating {split_name} graphs...")
    print(f"Samples: {len(indices)}")

    for count, index in enumerate(tqdm(indices)):

        output_path = split_dir / f"graph_{count:06d}.pt"

        if output_path.exists():
            continue

        frames, label, pid = dataset[index]

        with torch.no_grad():

            keypoints = pose_model.extract(frames)

            node_features, edge_index = build_graph(
                keypoints
            )

        sample = {
            "node_features": node_features.cpu(),
            "edge_index": edge_index.cpu(),
            "label": torch.tensor(label).long(),
            "pid": str(pid)
        }

        torch.save(sample, output_path)

        del frames
        del keypoints
        del node_features
        del edge_index

        if count % 100 == 0:
            gc.collect()

    print(f"{split_name} cache completed.")


def main():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    dataset = PIEDataset(
        annotation_file=ANNOTATION_FILE,
        crop_dir=CROP_DIR,
        sequence_length=SEQUENCE_LENGTH
    )

    print(f"Total samples: {len(dataset)}")

    train_indices, val_indices, test_indices = create_split(
        dataset
    )

    print("\nSplit:")
    print(f"Train: {len(train_indices)}")
    print(f"Validation: {len(val_indices)}")
    print(f"Test: {len(test_indices)}")

    pose_model = PoseExtractor(
        model_name=str(PROJECT_ROOT / "yolo11n-pose.pt"),
        device=device
    )

    save_graphs(
        dataset,
        train_indices,
        "train",
        pose_model
    )

    save_graphs(
        dataset,
        val_indices,
        "val",
        pose_model
    )

    save_graphs(
        dataset,
        test_indices,
        "test",
        pose_model
    )

    print(f"\nSaved inside:")
    print(CACHE_DIR)


if __name__ == "__main__":
    main()
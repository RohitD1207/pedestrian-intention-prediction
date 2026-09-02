from datasets.data_loader import PIEDataset
from pathlib import Path

from models.pose_extracter import PoseExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    dataset = PIEDataset(
        annotation_file=PROJECT_ROOT / "datasets" / "pie_annotations_set01.csv",
        crop_dir=PROJECT_ROOT / "data" / "PIE_crops",
        sequence_length=16
    )
    frames, label, pid = dataset[0]
    print("Frames shape:", frames.shape)
    print("Label:", label)
    print("Pedestrian ID:", pid)
    keypoints = PoseExtractor().extract(frames)
    print("Keypoints shape:", keypoints.shape)
    print(keypoints)


if __name__ == "__main__":
    main()
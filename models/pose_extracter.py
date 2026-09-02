import torch
from ultralytics import YOLO


class PoseExtractor:

    def __init__(self, model_name="yolo11n-pose.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = YOLO(model_name)

    def extract(self, frames):
        """
        frames:
            Tensor [T, 3, H, W]

        returns:
            keypoints: [T, 17, 3]
                       x, y, confidence
        """

        results = self.model(
            frames,
            device=self.device,
            verbose=False
        )

        sequence_keypoints = []

        for result in results:

            if result.keypoints is None:
                sequence_keypoints.append(
                    torch.zeros(17, 3)
                )
                continue

            keypoints = result.keypoints.data

            if keypoints.shape[0] == 0:
                sequence_keypoints.append(
                    torch.zeros(17, 3)
                )
                continue

            # Take the detected person with the highest confidence
            person = keypoints[0]

            # x, y, confidence
            person = person[:, :3]

            sequence_keypoints.append(
                person.cpu()
            )

        return torch.stack(sequence_keypoints)
from datasets.data_loader import PIEDataset
from torch.utils.data import DataLoader
from models.resnet_encoder import ResNetEncoder


dataset = PIEDataset(
    annotation_file="pie_annotations_clean.csv",
    video_dir="data/PIE_clips/set01"
)

# loader = DataLoader(dataset, batch_size=2, shuffle=True)
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

for seq, label in loader:

    print(seq.shape)
    print(label)

    break

model = ResNetEncoder()

for seq, label in loader:

    features = model(seq)

    print(features.shape)

    break
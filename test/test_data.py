from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATASET_DIR = "dataset"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_ds = datasets.ImageFolder(
    root=f"{DATASET_DIR}/train",
    transform=transform
)

val_ds = datasets.ImageFolder(
    root=f"{DATASET_DIR}/val",
    transform=transform
)

print("Classes:", train_ds.classes)
print("Train size:", len(train_ds))
print("Val size:", len(val_ds))

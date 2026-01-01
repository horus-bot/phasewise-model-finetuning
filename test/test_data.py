import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ===================== PATHS =====================
BASE_DIR = r"C:\Users\lenovo\Desktop\image classicification"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# ===================== CHECK PATHS =====================
print("Checking dataset paths...\n")
print("Train exists:", os.path.exists(TRAIN_DIR))
print("Val exists:", os.path.exists(VAL_DIR))
print("Test exists:", os.path.exists(TEST_DIR))

if not all([
    os.path.exists(TRAIN_DIR),
    os.path.exists(VAL_DIR),
    os.path.exists(TEST_DIR)
]):
    raise FileNotFoundError("❌ Dataset folders not found. Check paths.")

# ===================== TRANSFORMS =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===================== LOAD DATASETS =====================
train_ds = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=transform
)

val_ds = datasets.ImageFolder(
    root=VAL_DIR,
    transform=transform
)

test_ds = datasets.ImageFolder(
    root=TEST_DIR,
    transform=transform
)

# ===================== BASIC INFO =====================
print("\n✅ DATASET LOADED SUCCESSFULLY")
print("Classes:", train_ds.classes)
print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))
print("Test samples:", len(test_ds))

# ===================== CLASS DISTRIBUTION =====================
from collections import Counter

train_labels = [label for _, label in train_ds.samples]
class_counts = Counter(train_labels)

print("\nTrain class distribution:")
for cls, count in sorted(class_counts.items()):
    print(f"Class {cls}: {count} images")

# ===================== VISUAL SANITY CHECK =====================
loader = DataLoader(train_ds, batch_size=8, shuffle=True)

images, labels = next(iter(loader))

plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i].permute(1, 2, 0))
    plt.title(f"Label: {labels[i].item()}")
    plt.axis("off")

plt.suptitle("Random Training Samples", fontsize=14)
plt.tight_layout()
plt.show()

print("\n🎉 Dataset verification complete. Ready for training.")

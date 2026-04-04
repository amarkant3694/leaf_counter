import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import LeafDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# PATHS
# ======================
CSV_PATH = "leaf_estimation_dataset/train.csv"
ROOT_DIR = "leaf_estimation_dataset/train"

# ======================
# TRANSFORMS
# ======================
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.ColorJitter(p=0.3),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

# ======================
# DATASET
# ======================
from torch.utils.data import Subset

# full dataset (no transform)
full_dataset = LeafDataset(CSV_PATH, ROOT_DIR, transform=None)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

indices = torch.randperm(len(full_dataset)).tolist()

train_indices = indices[:train_size]
val_indices = indices[train_size:]

# ✅ Separate datasets with different transforms
train_dataset = LeafDataset(CSV_PATH, ROOT_DIR, transform=train_transform)
val_dataset = LeafDataset(CSV_PATH, ROOT_DIR, transform=val_transform)

train_dataset = Subset(train_dataset, train_indices)
val_dataset = Subset(val_dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

# ======================
# MODEL
# ======================
class LeafModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "convnext_base",
            pretrained=True,
            num_classes=0
        )

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            in_features = features.shape[1]

        self.regressor = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        return x

model = LeafModel().to(device)

# ======================
# LOSS + OPTIMIZER
# ======================
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ======================
# TRAIN FUNCTION
# ======================
def train_epoch():
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().to(device)

        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# ======================
# VALIDATION FUNCTION
# ======================
def validate():
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item()

    return total_loss / len(val_loader)

# ======================
# TRAIN LOOP
# ======================
EPOCHS = 5
best_loss = float("inf")

for epoch in range(EPOCHS):
    train_loss = train_epoch()
    val_loss = validate()

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Save best model
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Model saved!")
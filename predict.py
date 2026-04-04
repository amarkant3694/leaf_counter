import torch
import torch.nn as nn
import pandas as pd
import os
import cv2
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# PATHS
# ======================
TEST_CSV = "leaf_estimation_dataset/test.csv"
TEST_DIR = "leaf_estimation_dataset/test/images"

# ======================
# TRANSFORM (NO AUGMENTATION)
# ======================
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

# ======================
# MODEL (same as training)
# ======================
class LeafModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "convnext_base",
            pretrained=False,
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
# LOAD TRAINED WEIGHTS
# ======================
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ======================
# LOAD TEST CSV
# ======================
df = pd.read_csv(TEST_CSV)

predictions = []

# ======================
# PREDICTION LOOP
# ======================
for i in range(len(df)):
    filename = df.iloc[i]['filename']

    # remove "test/images/" prefix
    filename = filename.replace("test/images/", "")

    img_path = os.path.join(TEST_DIR, filename)

    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: missing image {img_path}")
        predictions.append(0)
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = transform(image=image)['image']
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image).item()
        print("Raw output:", output)

    # round to nearest integer
    pred = int(output)
    pred = max(0, pred)
    predictions.append(pred)
    print("Loading:", img_path)

# ======================
# SAVE CSV
# ======================
df['leaf_count'] = predictions
df.to_csv("submission.csv", index=False)

print("Submission file saved: submission.csv")

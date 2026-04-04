import streamlit as st
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Leaf AI 🌿", layout="centered")

# ======================
# CUSTOM STYLE
# ======================
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 AI Leaf Counter")
st.write("Upload or capture a plant image to count leaves")

# ======================
# DEVICE
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# MODEL
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

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    model = LeafModel().to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# ======================
# TRANSFORM
# ======================
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

# ======================
# INPUT OPTIONS
# ======================
option = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"])

image_np = None

# ----------------------
# UPLOAD OPTION
# ----------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

# ----------------------
# CAMERA OPTION
# ----------------------
if option == "Use Camera":
    camera_image = st.camera_input("📷 Capture Image")

    if camera_image is not None:
        image = Image.open(camera_image)
        image_np = np.array(image)
        st.image(image, caption="Captured Image", use_column_width=True)

# ======================
# PREDICTION BUTTON
# ======================
if image_np is not None:
    if st.button("🔍 Predict Leaf Count"):

        img = transform(image=image_np)['image']
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img).item()

        pred = int(output)
        pred = max(0, pred)

        # RESULT
        st.success(f"🌿 Predicted Leaves: {pred}")

        # EXTRA UI
        st.balloons()

        confidence = min(100, int(abs(output) * 5))
        st.info(f"Confidence Score: {confidence}%")


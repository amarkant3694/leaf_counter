# 🌿 Leaf Count Prediction using Deep Learning

## 🚀 Overview

This project is an AI-powered application developed to predict the number of leaves in a plant image using deep learning. It uses computer vision techniques and a pretrained neural network to provide accurate and real-time predictions through an interactive web interface.

---

## 🎯 Features

* 📤 Upload plant images
* 📷 Capture images using camera
* 🤖 AI-based leaf count prediction
* ⚡ Interactive and user-friendly UI
* 🌐 Deployed web application

---

## 🧠 Model Details

* **Architecture:** ConvNeXt (Pretrained)
* **Framework:** PyTorch
* **Task:** Regression (Leaf Count Estimation)
* **Input Size:** 224 × 224
* **Loss Function:** L1 Loss (MAE)

---

## 🗂️ Project Structure

```
LeafCount/
│
├── app.py              # Streamlit GUI application
├── train.py            # Model training script
├── predict.py          # Prediction script
├── dataset.py          # Dataset handling
├── requirements.txt    # Dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/amarkant3694/leaf-counter.git
cd leaf-counter
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App Locally

```bash
streamlit run app.py
```

---

## 🌐 Live Demo

👉 https://leafcounter-forgrowingplants.streamlit.app/

---

## 📊 Dataset

* The dataset consists of plant images with labeled leaf counts.
* Due to large size, the dataset is not included in this repository.

---

## 🏗️ Tech Stack

* Python
* PyTorch
* OpenCV
* Albumentations
* Streamlit

---

## 👨‍💻 Developed By

**Amarkant Raj**

---

## 💡 Future Improvements

* Improve accuracy using advanced models and ensemble techniques
* Add plant disease detection
* Optimize model for mobile deployment
* Enhance UI/UX further

---

## 📌 Conclusion

This project demonstrates the practical application of deep learning in agriculture, enabling automated and efficient plant analysis.

---

## ⭐ Acknowledgements

* PyTorch
* Streamlit
* Open-source libraries and datasets

---

⭐ If you like this project, consider giving it a star!

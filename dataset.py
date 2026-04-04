import os
import pandas as pd
import cv2
from torch.utils.data import Dataset

class LeafDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        while True:
            filename = self.df.iloc[idx]['filename']

            # remove 'train/' if present
            if filename.startswith("train/"):
                filename = filename.replace("train/", "")

            img_path = os.path.join(self.root_dir, filename)

            image = cv2.imread(img_path)

            # ✅ skip missing images
            if image is None:
                idx = (idx + 1) % len(self.df)
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            label = self.df.iloc[idx]['leaf_count']

            if self.transform:
                image = self.transform(image=image)['image']

            return image, label
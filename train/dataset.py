import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os

class BloodArtDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.normal_dir = os.path.join(root_dir, r"C:\Users\THIYA\OneDrive\Desktop\blood_art_world_new\dataset\normal")
        self.blood_art_dir = os.path.join(root_dir,r"C:\Users\THIYA\OneDrive\Desktop\blood_art_world_new\dataset\blood_art" )
        self.image_files = [f for f in os.listdir(self.normal_dir) if os.path.isfile(os.path.join(self.normal_dir, f))]
        self.transform = transform

    def __getitem__(self, idx):
        normal_path = os.path.join(self.normal_dir, self.image_files[idx])
        blood_art_path = os.path.join(self.blood_art_dir, self.image_files[idx])
        normal_img = Image.open(normal_path).convert('RGB')
        blood_art_img = Image.open(blood_art_path).convert('RGB')
        if self.transform:
            normal_img = self.transform(normal_img)
            blood_art_img = self.transform(blood_art_img)
        return normal_img, blood_art_img

    def __len__(self):
        return len(self.image_files)
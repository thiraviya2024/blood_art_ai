import os
import cv2
import numpy as np
from PIL import Image  # Ensure this line is present
import torch


def preprocess_pil(img):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    img_np = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img_np)

def postprocess_output(tensor):
    img = tensor_to_pil(tensor)
    img_np = np.array(img)
    img_np[:] = [255, 0, 0]  # Blue background
    flower = cv2.imread("utils/flower.png", cv2.IMREAD_UNCHANGED)
    if flower is not None:
        flower = cv2.resize(flower, (100, 100))
        x, y = 50, 50
        for c in range(0, 3):
            img_np[y:y+flower.shape[0], x:x+flower.shape[1], c] = flower[:, :, c] * (flower[:, :, 3] / 255.0) + img_np[y:y+flower.shape[0], x:x+flower.shape[1], c] * (1.0 - flower[:, :, 3] / 255.0)
    img_np = cv2.bilateralFilter(img_np, 9, 75, 75)  # Watercolor effect
    return Image.fromarray(img_np)

def tensor_to_pil(tensor):
    pil_img = torch.clamp(tensor * 0.5 + 0.5, 0, 1).permute(1, 2, 0).cpu().numpy() * 255
    return Image.fromarray(pil_img.astype(np.uint8))

def blood_art_filter(img, size=512):
    return img.resize((size, size), Image.LANCZOS)
import os
def generate_sketches():
    input_dir = r"C:\Users\THIYA\OneDrive\Desktop\blood_art_world_new\dataset\blood_art"
    output_dir = r"C:\Users\THIYA\OneDrive\Desktop\blood_art_world_new\dataset\normal"
    print(f"Creating directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing files from {input_dir}")
    for filename in os.listdir(input_dir):
        print(f"Processing {filename}")
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {filename}, skipping")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        cv2.imwrite(os.path.join(output_dir, filename), edges)
        print(f"Saved sketch: {filename}")

if __name__ == "__main__":
    generate_sketches()
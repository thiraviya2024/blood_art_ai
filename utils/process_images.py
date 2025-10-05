import os
import cv2
import numpy as np
from PIL import Image

# Paths
input_dir = r"C:\Users\THIYA\OneDrive\Desktop\blood_art_world_new\utils\archive\Images"
output_dir = r"C:\Users\THIYA\OneDrive\Desktop\blood_art_world_new\dataset\blood_art"
lotus_path = r"C:\Users\THIYA\OneDrive\Desktop\blood_art_world_new\utils\lotus.png"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory created or exists: {output_dir}")

# Load lotus image
lotus = cv2.imread(lotus_path, cv2.IMREAD_UNCHANGED)
if lotus is None:
    raise FileNotFoundError("Lotus image not found at " + lotus_path)
lotus = cv2.resize(lotus, (100, 100))  # Resize lotus to 100x100 pixels
lotus = lotus.astype(np.float32) / 255.0  # Normalize lotus to [0, 1] for blending
print(f"Lotus image loaded and resized: {lotus.shape}")

# Process images
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
print(f"Found {len(image_files)} image files in {input_dir}")
for i, filename in enumerate(image_files[:50]):  # Process first 50 images
    input_path = os.path.join(input_dir, filename)
    print(f"Processing {filename}")
    img = cv2.imread(input_path)
    if img is None:
        print(f"Skipping {filename}: Unable to load")
        continue

    # Convert to RGB and uint8
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

    # Add blue background
    mask = np.ones_like(img, dtype=np.uint8) * [0, 0, 255]  # Blue background
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask[thresh == 255] = img[thresh == 255]  # Keep original portrait
    img = mask

    # Overlay light pink lotus (3 positions) with float32 for precision
    img_float = img.astype(np.float32) / 255.0  # Convert to [0, 1] for blending
    height, width = img.shape[:2]
    positions = [(50, 50), (width-150, 50), (width//2, height-150)]
    for x, y in positions:
        if x + 100 <= width and y + 100 <= height:
            for c in range(0, 3):
                img_float[y:y+100, x:x+100, c] = (lotus[:, :, c] * (lotus[:, :, 3] / 255.0) + 
                                                img_float[y:y+100, x:x+100, c] * (1.0 - lotus[:, :, 3] / 255.0))

    # Apply watercolor-like effect on float32, then convert back
    img_filtered = cv2.bilateralFilter(img_float * 255.0, 9, 75, 75)  # Scale back to [0, 255]
    img = np.clip(img_filtered, 0, 255).astype(np.uint8)

    # Save with new name
    output_filename = f"image{i+1}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Saved {output_filename}")

print("Image processing complete!")
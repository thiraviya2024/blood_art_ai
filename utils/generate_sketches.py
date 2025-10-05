import cv2
import os

input_dir = r"C:\Users\THIYA\OneDrive\Desktop\blood_art_world_new\dataset\blood_art"
output_dir = r"C:\Users\THIYA\OneDrive\Desktop\blood_art_world_new\dataset\normal"
os.makedirs(output_dir, exist_ok=True)
print(f"Processing sketches from {input_dir} to {output_dir}")

for filename in os.listdir(input_dir):
    print(f"Processing {filename}")
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skipping {filename}: Unable to load")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cv2.imwrite(os.path.join(output_dir, filename), edges)
    print(f"Saved sketch: {filename}")

print("Sketches generated!")
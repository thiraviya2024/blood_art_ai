from PIL import Image, ImageFilter
import os

input_dir = r"C:\Users\THIYA\OneDrive\Desktop\blood_art_world_new\dataset\blood_art"
output_dir = r"C:\Users\THIYA\OneDrive\Desktop\blood_art_world_new\dataset\normal"
os.makedirs(output_dir, exist_ok=True)
print("hello")
print(f"Starting sketch generation. Input directory: {input_dir}")
print(f"Output directory: {output_dir}")
print(f"Checking files in {input_dir}...")

file_count = 0
files_found = os.listdir(input_dir)
print("hello again")
print(f"Found {len(files_found)} files: {files_found[:5]}...")
for filename in files_found:
    file_count += 1
    print(f"hello loop {file_count}")
    print(f"Attempting to process file {file_count}: {filename}")
    img_path = os.path.join(input_dir, filename)
    try:
        with Image.open(img_path) as img:
            print(f"hello loaded {filename}")
            print(f"Successfully loaded {filename}. Converting to sketch...")
            img_gray = img.convert("L")
            img_sketch = img_gray.filter(ImageFilter.FIND_EDGES)
            img_sketch = img_sketch.point(lambda x: 0 if x < 128 else 255)
            output_path = os.path.join(output_dir, filename)
            img_sketch.save(output_path, "JPEG")
            print(f"hello saved {filename}")
            print(f"Successfully saved sketch: {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}. Skipping.")

print(f"hello done")
print(f"Total files processed: {file_count}")
print("Sketches generated!")
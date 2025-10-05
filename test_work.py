import sys
import os
import torch
from PIL import Image
from torchvision import transforms

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your generator and discriminator from the models folder
from models.generator import GeneratorUNet
from models.discriminator import Discriminator

# Device
device = torch.device("cpu")

# Initialize generator
generator = GeneratorUNet().to(device)

# Load trained model weights (use raw string for Windows path)
generator_path = r"C:\Users\THIYA\OneDrive\Desktop\blood_art_world_new\models\generator_epoch_10.pth"
generator.load_state_dict(torch.load(generator_path, map_location=device))
generator.eval()

# Load test image
test_image_path = r"C:\Users\THIYA\OneDrive\Desktop\blood_art_world_new\dataset\test_image.jpeg"
img = Image.open(test_image_path).convert("RGB")

# Transform image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
input_tensor = transform(img).unsqueeze(0).to(device)

# Generate output
with torch.no_grad():
    output = generator(input_tensor)

# Convert output to image
output_img = transforms.ToPILImage()(output.squeeze(0))
output_img.save("output.jpg")

print("✅ Output saved as output.jpg in the backend folder")

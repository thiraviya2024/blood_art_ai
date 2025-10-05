# backend/app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from models.generator import GeneratorUNet
from models.discriminator import Discriminator  # if needed later
import torch
from torchvision import transforms
from PIL import Image
import io
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))

from generator import GeneratorUNet
from discriminator import Discriminator


app = FastAPI(title="Blood Art Generator API")

# -------------------------------
# Load the trained model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = GeneratorUNet()
generator.load_state_dict(torch.load(r"C:\Users\THIYA\OneDrive\Desktop\blood_art_world_new\models\generator_epoch_10.pth", map_location=device))
generator.to(device)
generator.eval()

# -------------------------------
# Image transformation
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# -------------------------------
# Root endpoint
# -------------------------------
@app.get("/")
async def root():
    return {"message": "Blood Art Generator API is running!"}

# -------------------------------
# Image generation endpoint
# -------------------------------
@app.post("/generate")
async def generate_blood_art(file: UploadFile = File(...)):
    try:
        # Read image
        image = Image.open(file.file).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Generate image
        with torch.no_grad():
            output_tensor = generator(input_tensor)
        
        # Convert tensor to PIL image
        output_tensor = (output_tensor.squeeze(0).cpu() * 0.5 + 0.5)  # Denormalize
        output_image = transforms.ToPILImage()(output_tensor)

        # Convert to bytes
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

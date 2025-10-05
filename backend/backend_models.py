# backend/app.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
from torchvision import transforms

# Import Generator and Discriminator from the root models folder
from models.generator import GeneratorUNet
from models.discriminator import Discriminator  # if you plan to use it

app = FastAPI(title="Blood Art Generator API")

# Allow CORS (so your frontend can call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Generator model
generator = GeneratorUNet()
generator.load_state_dict(torch.load(r"C:\Users\THIYA\OneDrive\Desktop\blood_art_world_new\models\generator_epoch_10.pth", map_location=device))
generator.to(device)
generator.eval()

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

@app.post("/generate")
async def generate_blood_art(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Generate blood art
        with torch.no_grad():
            output_tensor = generator(img_tensor)
        
        # Convert output to PIL image
        output_tensor = (output_tensor.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
        output_image = transforms.ToPILImage()(output_tensor)

        # Save output to bytes
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return JSONResponse(content={"message": "Success"}, media_type="application/json")
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "Blood Art Generator API is running!"}

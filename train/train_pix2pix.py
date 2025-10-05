import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.generator import GeneratorUNet
from models.discriminator import Discriminator
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from train.dataset import BloodArtDataset  # Make sure your dataset.py is in train/

from train.losses import generator_loss, discriminator_loss

# ------------------------------
# Config
# ------------------------------
ROOT_DIR = r"C:\Users\THIYA\OneDrive\Desktop\blood_art_world_new\dataset"
SAVE_DIR = r"C:\Users\THIYA\OneDrive\Desktop\blood_art_world_new\models"
EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------
# Dataset
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = BloodArtDataset(ROOT_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ------------------------------
# Models
# ------------------------------
generator = GeneratorUNet().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

optimizer_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

# ------------------------------
# Training Loop
# ------------------------------
for epoch in range(1, EPOCHS + 1):
    for i, (normal_img, blood_art_img) in enumerate(dataloader):
        normal_img, blood_art_img = normal_img.to(DEVICE), blood_art_img.to(DEVICE)

        # ------------------------------
        # Train Generator
        # ------------------------------
        optimizer_g.zero_grad()
        fake_img = generator(normal_img)
        g_loss = generator_loss(discriminator, fake_img, blood_art_img)
        g_loss.backward()
        optimizer_g.step()

        # ------------------------------
        # Train Discriminator
        # ------------------------------
        optimizer_d.zero_grad()
        d_loss = discriminator_loss(discriminator, fake_img.detach(), blood_art_img)
        d_loss.backward()
        optimizer_d.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Batch [{i}/{len(dataloader)}] "
                  f"G Loss: {g_loss.item():.4f} D Loss: {d_loss.item():.4f}")

    # Save checkpoint every epoch
    torch.save(generator.state_dict(), os.path.join(SAVE_DIR, f"generator_epoch_{epoch}.pth"))
    print(f"Saved checkpoint: generator_epoch_{epoch}.pth")

print("Training finished!")

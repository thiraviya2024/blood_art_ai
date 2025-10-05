import torch
import torch.nn as nn

def generator_loss(discriminator, fake_img, target_img):
    gan_loss = nn.BCEWithLogitsLoss()(discriminator(fake_img), torch.ones_like(discriminator(fake_img)))
    l1_loss = nn.L1Loss()(fake_img, target_img)
    return gan_loss + 150 * l1_loss

def discriminator_loss(discriminator, fake_img, target_img):
    real_loss = nn.BCEWithLogitsLoss()(discriminator(target_img), torch.ones_like(discriminator(target_img)))
    fake_loss = nn.BCEWithLogitsLoss()(discriminator(fake_img.detach()), torch.zeros_like(discriminator(fake_img)))
    return (real_loss + fake_loss) / 2
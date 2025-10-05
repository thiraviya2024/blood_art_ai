import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_act=True):
        super().__init__()
        if down:
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                      nn.BatchNorm2d(out_ch)]
            if use_act:
                layers.append(nn.LeakyReLU(0.2))
        else:
            layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                      nn.BatchNorm2d(out_ch),
                      nn.ReLU()]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Encoder
        self.down1 = UNetBlock(in_channels, 64, down=True, use_act=False)
        self.down2 = UNetBlock(64, 128)
        self.down3 = UNetBlock(128, 256)
        self.down4 = UNetBlock(256, 512)
        # Decoder
        self.up1 = UNetBlock(512, 256, down=False)
        self.up2 = UNetBlock(256*2, 128, down=False)
        self.up3 = UNetBlock(128*2, 64, down=False)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64*2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        return self.final(torch.cat([u3, d1], dim=1))

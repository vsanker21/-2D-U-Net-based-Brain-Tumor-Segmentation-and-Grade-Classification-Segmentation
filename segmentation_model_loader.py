"""
Load 2D U-Net segmentation model for manuscript figure generation.
Architecture must match the notebook (Project1) UNet2D: in_channels=1, num_classes=4.
"""
import torch
import torch.nn as nn
from pathlib import Path

class UNet2D(nn.Module):
    """2D U-Net for slice-based brain tumor segmentation (matches notebook)."""
    def __init__(self, in_channels=1, num_classes=4, base_features=32):
        super(UNet2D, self).__init__()
        self.enc1 = self._conv_block(in_channels, base_features)
        self.enc2 = self._conv_block(base_features, base_features * 2)
        self.enc3 = self._conv_block(base_features * 2, base_features * 4)
        self.enc4 = self._conv_block(base_features * 4, base_features * 8)
        self.bottleneck = self._conv_block(base_features * 8, base_features * 16)
        self.up4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, 2, 2)
        self.dec4 = self._conv_block(base_features * 16, base_features * 8)
        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, 2, 2)
        self.dec3 = self._conv_block(base_features * 8, base_features * 4)
        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, 2, 2)
        self.dec2 = self._conv_block(base_features * 4, base_features * 2)
        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, 2, 2)
        self.dec1 = self._conv_block(base_features * 2, base_features)
        self.final = nn.Conv2d(base_features, num_classes, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.final(d1)


def load_segmentation_model(path, device=None):
    """Load UNet2D state dict from path. Returns model in eval mode."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet2D(in_channels=1, num_classes=4, base_features=32).to(device)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()
    return model

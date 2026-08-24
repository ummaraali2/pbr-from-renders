"""PBRNet: six views -> basecolor / roughness / metallic maps.

This is a PLAIN torch.nn.Module, deliberately NOT a Tesseract. Tesseract's own
learned-closure demo keeps the network in-process and containerizes only the
solver; containerizing the network doubles HTTP round-trips and breaks
`optimizer = Adam(model.parameters())`, since a Tesseract object has no
`.parameters()`.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class PBRNet(nn.Module):
    """Input  [B, 6, H, W, 3] -> reshaped to [B, 18, H, W]
    Output basecolor [B,3,H,W], roughness [B,1,H,W], metallic [B,1,H,W]

    Stacking all six views as 18 channels lets the first convolution compare
    how a surface point's brightness changes across viewpoints -- the signal
    that separates roughness from basecolor.

    H and W must be divisible by 8 (three 2x pools).
    """

    def __init__(self, base_ch=32):
        super().__init__()
        self.enc1 = DoubleConv(18, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        # Sigmoid makes physically invalid values unrepresentable rather than
        # merely penalised. All three PBR params are bounded to [0,1].
        self.head_bc = nn.Sequential(nn.Conv2d(base_ch, 3, 1), nn.Sigmoid())
        self.head_ro = nn.Sequential(nn.Conv2d(base_ch, 1, 1), nn.Sigmoid())
        self.head_me = nn.Sequential(nn.Conv2d(base_ch, 1, 1), nn.Sigmoid())

    def forward(self, views):
        b, n, h, w, c = views.shape
        assert h % 8 == 0 and w % 8 == 0, f"H,W must be divisible by 8, got {h}x{w}"
        x = views.permute(0, 1, 4, 2, 3).reshape(b, n * c, h, w)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bott = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(bott), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return {
            "basecolor": self.head_bc(d1),
            "roughness": self.head_ro(d1),
            "metallic": self.head_me(d1),
        }


def to_renderer_layout(maps):
    """[B,C,H,W] -> [H,W,C], dropping the batch dim. Renderer wants HWC."""
    return {
        "basecolor": maps["basecolor"][0].permute(1, 2, 0),
        "roughness": maps["roughness"][0].permute(1, 2, 0),
        "metallic": maps["metallic"][0].permute(1, 2, 0),
    }

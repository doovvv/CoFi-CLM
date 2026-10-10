"""Fuse direct coarse CSI and the CSI generated from fine-token predictions."""

import torch
import torch.nn as nn


class FusionResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = next(g for g in (8, 4, 2, 1) if channels % g == 0)
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class CoarseLMResidualFusion(nn.Module):
    def __init__(self, channels=2, hidden_channels=64, residual_blocks=2, csi_center=0.5):
        super().__init__()
        self.csi_center = csi_center
        self.input_projection = nn.Conv2d(channels * 3, hidden_channels, 3, padding=1)
        self.blocks = nn.Sequential(
            *(FusionResidualBlock(hidden_channels) for _ in range(residual_blocks))
        )
        self.gate_head = nn.Conv2d(hidden_channels, channels, 1)
        self.feature_bias = nn.Parameter(torch.zeros(hidden_channels))
        self.gate_bias = nn.Parameter(torch.zeros(channels))
        self.correction_head = nn.Conv2d(hidden_channels, channels, 3, padding=1)
        nn.init.kaiming_normal_(self.input_projection.weight, nonlinearity="linear")
        nn.init.zeros_(self.input_projection.bias)
        for module in (self.gate_head, self.correction_head):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, coarse, generated):
        coarse_centered, generated_centered = coarse - self.csi_center, generated - self.csi_center
        features = torch.cat(
            (coarse_centered, generated_centered, generated_centered - coarse_centered), dim=1
        )
        projected = self.input_projection(features)
        feature_bias = self.feature_bias.to(projected.dtype)[None, :, None, None]
        hidden = self.blocks(projected + feature_bias)
        gate_features = self.gate_head(hidden)
        gate_bias = self.gate_bias.to(gate_features.dtype)[None, :, None, None]
        gate = 1.0 + torch.tanh(gate_features + gate_bias)
        correction = self.correction_head(hidden)
        fused = coarse_centered + gate * (generated_centered - coarse_centered) + correction
        reconstruction = self.csi_center + fused
        return reconstruction

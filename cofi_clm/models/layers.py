"""Shared convolutional and spatial-position layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def bilinear_interpolate_2d(inputs, size=None, scale_factor=None):
    """Run bilinear interpolation safely on runtimes without BF16 support."""
    output_dtype = inputs.dtype
    interpolation_input = inputs.float() if output_dtype == torch.bfloat16 else inputs
    output = F.interpolate(
        interpolation_input,
        size=size,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=False,
    )
    return output.to(dtype=output_dtype)


class BFloat16SafeBilinearUpsample(nn.Module):
    """Bilinear upsampling with a local FP32 fallback for BF16 tensors."""

    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, inputs):
        return bilinear_interpolate_2d(
            inputs,
            scale_factor=self.scale_factor,
        )


def _group_count(channels, max_groups=8):
    groups = min(int(max_groups), int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class ResidualConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = F.gelu(self.norm1(x))
        residual = self.conv1(residual)
        residual = F.gelu(self.norm2(residual))
        residual = self.conv2(residual)
        return x + residual


class Learned2DPositionEncoding(nn.Module):
    def __init__(self, height, width, dim):
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.row = nn.Embedding(self.height, dim)
        self.column = nn.Embedding(self.width, dim)
        nn.init.normal_(self.row.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.column.weight, mean=0.0, std=0.02)

    def forward(self, batch_size):
        positions = self.row.weight[:, None, :] + self.column.weight[None, :, :]
        return positions.reshape(1, self.height * self.width, -1).expand(
            batch_size,
            -1,
            -1,
        )

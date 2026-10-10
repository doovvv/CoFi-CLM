"""Convolutional CSI tokenizer with K-means initialization and EMA vector quantization."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cofi_clm.models.layers import (
    BFloat16SafeBilinearUpsample,
    ResidualConvBlock,
    _group_count,
)


class LearnedCSITokenizer(nn.Module):
    def __init__(
        self,
        in_channels=2,
        patch_size=4,
        codebook_size=256,
        embedding_dim=64,
        hidden_channels=64,
        commitment_weight=0.25,
        ema_decay=0.99,
        ema_eps=1e-5,
        dead_code_threshold=1.0,
        kmeans_iters=5,
        kmeans_samples=4096,
        diversity_temperature=0.1,
        diversity_target_fraction=0.25,
        latent_std_target=0.1,
        residual_blocks=2,
        csi_center=0.5,
    ):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if patch_size & (patch_size - 1):
            raise ValueError("patch_size must be a power of two.")
        if codebook_size <= 1:
            raise ValueError("codebook_size must be greater than 1.")
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.codebook_size = int(codebook_size)
        self.embedding_dim = int(embedding_dim)
        self.commitment_weight = float(commitment_weight)
        self.ema_decay = float(ema_decay)
        self.ema_eps = float(ema_eps)
        self.dead_code_threshold = float(dead_code_threshold)
        self.kmeans_iters = int(kmeans_iters)
        self.kmeans_samples = int(kmeans_samples)
        self.diversity_temperature = float(diversity_temperature)
        self.diversity_target_fraction = float(diversity_target_fraction)
        self.latent_std_target = float(latent_std_target)
        self.residual_blocks = int(residual_blocks)
        self.csi_center = float(csi_center)
        self.ema_updates_enabled = True
        self.quantizer_enabled = True
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1).")
        if self.dead_code_threshold <= 0:
            raise ValueError("dead_code_threshold must be positive.")
        if self.kmeans_iters < 0 or self.kmeans_samples <= 0:
            raise ValueError("kmeans_iters must be non-negative and kmeans_samples positive.")
        if self.diversity_temperature <= 0:
            raise ValueError("diversity_temperature must be positive.")
        if not 0.0 < self.diversity_target_fraction <= 1.0:
            raise ValueError("diversity_target_fraction must be in (0, 1].")
        if self.latent_std_target < 0:
            raise ValueError("latent_std_target must be non-negative.")
        if self.residual_blocks <= 0:
            raise ValueError("residual_blocks must be positive.")

        downsample_stages = int(math.log2(patch_size))
        encoder_layers = [nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)]
        for _ in range(downsample_stages):
            encoder_layers.extend(
                ResidualConvBlock(hidden_channels) for _ in range(self.residual_blocks)
            )
            encoder_layers.append(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
        encoder_layers.extend(
            ResidualConvBlock(hidden_channels) for _ in range(self.residual_blocks)
        )
        encoder_layers.append(nn.Conv2d(hidden_channels, embedding_dim, kernel_size=1))
        self.encoder = nn.Sequential(*encoder_layers)
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        self.codebook.weight.requires_grad_(False)
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed_sum", torch.zeros(codebook_size, embedding_dim))
        self.register_buffer("codebook_initialized", torch.tensor(False, dtype=torch.bool))
        decoder_layers = [nn.Conv2d(embedding_dim, hidden_channels, kernel_size=3, padding=1)]
        for _ in range(downsample_stages):
            decoder_layers.extend(
                ResidualConvBlock(hidden_channels) for _ in range(self.residual_blocks)
            )
            decoder_layers.extend(
                [
                    BFloat16SafeBilinearUpsample(scale_factor=2),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                ]
            )
        decoder_layers.extend(
            ResidualConvBlock(hidden_channels) for _ in range(self.residual_blocks)
        )
        decoder_layers.extend(
            [
                nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
                nn.GELU(),
                nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
            ]
        )
        self.decoder = nn.Sequential(*decoder_layers)
        self.last_vq_loss = None
        self.last_commitment_loss = None
        self.last_diversity_loss = None
        self.last_latent_variance_loss = None
        self._reset_parameters()

    @staticmethod
    def _distances(tokens, codebook):
        return (
            torch.sum(tokens**2, dim=-1, keepdim=True)
            - 2.0 * torch.matmul(tokens, codebook.t())
            + torch.sum(codebook**2, dim=-1).reshape(1, -1)
        )

    def set_ema_updates_enabled(self, enabled):
        self.ema_updates_enabled = bool(enabled)

    def set_quantizer_enabled(self, enabled):
        self.quantizer_enabled = bool(enabled)

    def _latent_variance_loss(self, z_tokens):
        flat_tokens = z_tokens.reshape(-1, self.embedding_dim).float()
        latent_std = torch.sqrt(torch.var(flat_tokens, dim=0, unbiased=False) + 1e-4)
        return torch.mean(F.relu(self.latent_std_target - latent_std))

    def _anti_collapse_losses(self, z_tokens, indices, distances):
        soft_assignments = F.softmax(
            -distances / self.diversity_temperature,
            dim=-1,
        )
        hard_assignments = F.one_hot(indices, num_classes=self.codebook_size).to(
            dtype=soft_assignments.dtype
        )
        straight_through_assignments = (
            hard_assignments + soft_assignments - soft_assignments.detach()
        )
        average_probabilities = straight_through_assignments.mean(dim=(0, 1))
        assignment_entropy = -torch.sum(
            average_probabilities * torch.log(average_probabilities.clamp_min(1e-8))
        )
        target_perplexity = max(2.0, self.codebook_size * self.diversity_target_fraction)
        target_entropy = math.log(target_perplexity)
        diversity_loss = F.relu(assignment_entropy.new_tensor(target_entropy) - assignment_entropy)

        latent_variance_loss = self._latent_variance_loss(z_tokens)
        return diversity_loss, latent_variance_loss

    @torch.no_grad()
    def _initialize_codebook(self, z_tokens):
        flat = z_tokens.detach().reshape(-1, self.embedding_dim).float()
        sample_count = min(flat.shape[0], self.kmeans_samples)
        samples = flat[torch.randperm(flat.shape[0], device=flat.device)[:sample_count]]
        if sample_count < self.codebook_size:
            repeats = int(math.ceil(self.codebook_size / float(sample_count)))
            samples = samples.repeat(repeats, 1)
            sample_count = samples.shape[0]

        centroid_indices = torch.randperm(sample_count, device=flat.device)[: self.codebook_size]
        centroids = samples[centroid_indices].clone()
        for _ in range(self.kmeans_iters):
            assignments = torch.argmin(self._distances(samples, centroids), dim=-1)
            counts = torch.bincount(assignments, minlength=self.codebook_size).float()
            sums = torch.zeros_like(centroids)
            sums.index_add_(0, assignments, samples)
            nonempty = counts > 0
            centroids[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(-1)
            empty = ~nonempty
            if torch.any(empty):
                replacements = samples[
                    torch.randint(sample_count, (int(empty.sum().item()),), device=flat.device)
                ]
                centroids[empty] = replacements

        assignments = torch.argmin(self._distances(flat, centroids), dim=-1)
        counts = torch.bincount(assignments, minlength=self.codebook_size).float()
        initial_counts = counts.clamp_min(self.dead_code_threshold)
        self.codebook.weight.copy_(centroids)
        self.ema_cluster_size.copy_(initial_counts)
        self.ema_embed_sum.copy_(centroids * initial_counts.unsqueeze(-1))
        self.codebook_initialized.fill_(True)

    @torch.no_grad()
    def _ema_update_codebook(self, z_tokens, indices):
        flat = z_tokens.detach().reshape(-1, self.embedding_dim).to(self.ema_embed_sum.dtype)
        flat_indices = indices.detach().reshape(-1)
        counts = torch.bincount(flat_indices, minlength=self.codebook_size).float()
        embed_sum = torch.zeros_like(self.ema_embed_sum)
        embed_sum.index_add_(0, flat_indices, flat)

        self.ema_cluster_size.mul_(self.ema_decay).add_(
            counts,
            alpha=1.0 - self.ema_decay,
        )
        self.ema_embed_sum.mul_(self.ema_decay).add_(
            embed_sum,
            alpha=1.0 - self.ema_decay,
        )

        total_count = self.ema_cluster_size.sum()
        smoothed_counts = (
            (self.ema_cluster_size + self.ema_eps)
            / (total_count + self.codebook_size * self.ema_eps)
            * total_count
        )
        updated_codebook = self.ema_embed_sum / smoothed_counts.unsqueeze(-1).clamp_min(
            self.ema_eps
        )
        self.codebook.weight.copy_(updated_codebook)

        dead_codes = self.ema_cluster_size < self.dead_code_threshold
        if torch.any(dead_codes):
            dead_count = int(dead_codes.sum().item())
            replacements = flat[torch.randint(flat.shape[0], (dead_count,), device=flat.device)]
            self.codebook.weight[dead_codes] = replacements
            self.ema_cluster_size[dead_codes] = self.dead_code_threshold
            self.ema_embed_sum[dead_codes] = replacements * self.dead_code_threshold

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, ResidualConvBlock):
                nn.init.zeros_(module.conv2.weight)
                if module.conv2.bias is not None:
                    nn.init.zeros_(module.conv2.bias)
        output_layer = self.decoder[-1]
        nn.init.zeros_(output_layer.weight)
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)
        nn.init.uniform_(self.codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def encode(self, x):
        z = self.encoder(x - self.csi_center)
        batch_size, dim, grid_h, grid_w = z.shape
        z_tokens = z.permute(0, 2, 3, 1).reshape(batch_size, grid_h * grid_w, dim)

        if not self.quantizer_enabled:
            zero = z_tokens.sum() * 0.0
            indices = torch.zeros(
                batch_size,
                grid_h * grid_w,
                dtype=torch.long,
                device=z_tokens.device,
            )
            self.last_commitment_loss = zero
            self.last_vq_loss = zero
            self.last_diversity_loss = zero
            self.last_latent_variance_loss = self._latent_variance_loss(z_tokens)
            return z_tokens, indices, (grid_h, grid_w)

        if self.training and self.ema_updates_enabled and not self.codebook_initialized.item():
            self._initialize_codebook(z_tokens)

        distance_tokens = z_tokens.float()
        codebook_snapshot = self.codebook.weight.detach().float().clone()
        distances = self._distances(distance_tokens, codebook_snapshot)
        indices = torch.argmin(distances, dim=-1)
        codebook_tokens = F.embedding(indices, codebook_snapshot)
        quantized = z_tokens + (codebook_tokens.to(z_tokens.dtype) - z_tokens).detach()

        commitment_loss = F.mse_loss(
            distance_tokens,
            codebook_tokens.detach(),
        )
        diversity_loss, latent_variance_loss = self._anti_collapse_losses(
            distance_tokens,
            indices,
            distances,
        )
        self.last_commitment_loss = commitment_loss
        self.last_vq_loss = self.commitment_weight * commitment_loss
        self.last_diversity_loss = diversity_loss
        self.last_latent_variance_loss = latent_variance_loss
        if self.training and self.ema_updates_enabled:
            self._ema_update_codebook(z_tokens, indices)
        return quantized, indices, (grid_h, grid_w)

    def decode(self, token_embeddings, grid_shape):
        batch_size, num_tokens, dim = token_embeddings.shape
        grid_h, grid_w = grid_shape
        if num_tokens != grid_h * grid_w:
            raise ValueError("token count does not match grid_shape.")
        z = token_embeddings.reshape(batch_size, grid_h, grid_w, dim).permute(0, 3, 1, 2)
        return self.csi_center + self.decoder(z)

    def forward(self, x):
        quantized, indices, grid_shape = self.encode(x)
        reconstruction = self.decode(quantized, grid_shape)
        return reconstruction, indices, quantized

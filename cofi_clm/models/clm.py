"""Paper CoFi-CLM with separate UE feedback and BS reconstruction interfaces."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cofi_clm.models.fusion import CoarseLMResidualFusion
from cofi_clm.models.language_model import CoarseToFineChannelLM
from cofi_clm.models.tokenizer import LearnedCSITokenizer


class CoFiCLM(nn.Module):
    def __init__(self, config, fine_tokenizer):
        super().__init__()
        if not fine_tokenizer.codebook_initialized.item():
            raise ValueError("Pretrain and initialize the absolute-CSI fine tokenizer first.")
        self.config = config
        self.feedback_bits = config["feedback_bits"]
        self.selected_tokens = config["rate"]["selected_tokens"]
        self.importance_decay = config["coarse"]["importance_decay"]
        self.coarse_grid = (32 // config["coarse"]["patch_size"],) * 2
        self.fine_grid = (32 // fine_tokenizer.patch_size,) * 2
        self.num_coarse_tokens = self.coarse_grid[0] * self.coarse_grid[1]
        if not 1 <= self.selected_tokens <= self.num_coarse_tokens:
            raise ValueError("The coarse grid must contain all selected tokens.")
        self.coarse_missing_embedding = nn.Parameter(
            torch.zeros(1, 1, config["tokenizer"]["embedding_dim"])
        )
        self.register_buffer("coarse_importance_ema", torch.zeros(self.num_coarse_tokens))
        self.fine_tokenizer = fine_tokenizer.requires_grad_(False)
        self.fine_tokenizer.set_ema_updates_enabled(False)
        self.fine_tokenizer.set_quantizer_enabled(True)
        self.fine_tokenizer.eval()
        self.coarse_tokenizer = LearnedCSITokenizer(
            patch_size=config["coarse"]["patch_size"],
            codebook_size=config["rate"]["codebook_size"],
            **config["tokenizer"],
        )
        self.channel_lm = CoarseToFineChannelLM(
            self.coarse_grid,
            self.fine_grid,
            self.coarse_tokenizer.embedding_dim,
            fine_tokenizer.embedding_dim,
            fine_tokenizer.codebook_size,
            **config["transformer"],
        )
        self.residual_fusion = CoarseLMResidualFusion(
            csi_center=config["tokenizer"]["csi_center"], **config["fusion"]
        )

    def train(self, mode=True):
        super().train(mode)
        self.fine_tokenizer.eval()
        return self

    def feedback_positions(self, batch_size):
        if self.selected_tokens == self.num_coarse_tokens:
            positions = torch.arange(
                self.num_coarse_tokens, device=self.coarse_importance_ema.device
            )
        else:
            positions = torch.topk(
                self.coarse_importance_ema, self.selected_tokens, sorted=True
            ).indices
        return positions.expand(batch_size, -1)

    def _transmitted_grid(self, embeddings, positions):
        selected = embeddings.gather(
            1, positions.unsqueeze(-1).expand(-1, -1, embeddings.shape[-1])
        )
        grid = (
            self.coarse_missing_embedding.to(embeddings.dtype)
            .expand(embeddings.shape[0], self.num_coarse_tokens, -1)
            .clone()
        )
        return grid.scatter(1, positions.unsqueeze(-1).expand_as(selected), selected)

    def _decode_grid(self, coarse_embeddings):
        coarse = self.coarse_tokenizer.decode(coarse_embeddings, self.coarse_grid)
        fine_embeddings, logits = self.channel_lm(
            coarse_embeddings, self.fine_tokenizer.codebook.weight
        )
        generated = self.fine_tokenizer.decode(fine_embeddings, self.fine_grid)
        return {
            "reconstruction": self.residual_fusion(coarse, generated),
            "coarse_reconstruction": coarse,
            "lm_reconstruction": generated,
            "token_logits": logits,
        }

    @torch.no_grad()
    def encode_feedback(self, csi):
        """UE: transmit only codebook indices at the checkpoint's fixed positions."""
        if self.training:
            raise RuntimeError("Call model.eval() before using the online feedback interface.")
        _, indices, _ = self.coarse_tokenizer.encode(csi)
        return indices.gather(1, self.feedback_positions(len(csi)))

    def reconstruct(self, indices):
        """BS: reconstruct CSI using indices and the shared model; no fine encoder."""
        if self.training:
            raise RuntimeError(
                "Call model.eval() before using the online reconstruction interface."
            )
        if indices.ndim != 2 or indices.shape[1] != self.selected_tokens:
            raise ValueError(f"Expected integer indices [batch, {self.selected_tokens}].")
        if indices.dtype != torch.long:
            raise TypeError("Feedback indices must have dtype torch.long.")
        selected = F.embedding(indices, self.coarse_tokenizer.codebook.weight)
        positions = self.feedback_positions(len(indices))
        grid = self.coarse_missing_embedding.expand(
            len(indices), self.num_coarse_tokens, -1
        ).clone()
        grid = grid.scatter(1, positions.unsqueeze(-1).expand_as(selected), selected)
        return self._decode_grid(grid)

    def forward(self, target):
        """Offline training: ground-truth fine indices supervise the token CE loss."""
        embeddings, indices, _ = self.coarse_tokenizer.encode(target)
        if self.training and self.selected_tokens < self.num_coarse_tokens:
            with torch.no_grad():
                distortion = ((embeddings - self.coarse_missing_embedding.detach()) ** 2).mean(-1)
                self.coarse_importance_ema.mul_(self.importance_decay).add_(
                    distortion.mean(0).float(), alpha=1 - self.importance_decay
                )
        positions = self.feedback_positions(len(target))
        coarse_grid = (
            embeddings
            if self.selected_tokens == self.num_coarse_tokens
            else self._transmitted_grid(embeddings, positions)
        )
        output = self._decode_grid(coarse_grid)
        with torch.no_grad():
            _, fine_indices, _ = self.fine_tokenizer.encode(target)
        output.update(
            target=target,
            fine_indices=fine_indices,
            feedback_indices=indices.gather(1, positions),
            vq=self.coarse_tokenizer.last_vq_loss,
            diversity=self.coarse_tokenizer.last_diversity_loss,
            variance=self.coarse_tokenizer.last_latent_variance_loss,
        )
        return output

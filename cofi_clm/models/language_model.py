"""Single-pass conditional prediction of the fine-codebook distribution."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from cofi_clm.models.layers import Learned2DPositionEncoding, bilinear_interpolate_2d


class CoarseToFineChannelLM(nn.Module):
    def __init__(
        self,
        coarse_grid_shape,
        fine_grid_shape,
        coarse_embedding_dim,
        fine_embedding_dim,
        fine_codebook_size,
        transformer_dim,
        transformer_layers,
        transformer_heads,
        dim_feedforward,
        dropout=0.05,
        gradient_checkpointing=True,
    ):
        super().__init__()
        if transformer_layers < 1 or transformer_dim % transformer_heads:
            raise ValueError("Use positive layers and a dimension divisible by the head count.")
        self.coarse_grid_shape = coarse_grid_shape
        self.fine_grid_shape = fine_grid_shape
        self.transformer_dim = transformer_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.coarse_projection = nn.Linear(coarse_embedding_dim, transformer_dim)
        self.coarse_position = Learned2DPositionEncoding(*coarse_grid_shape, transformer_dim)
        self.fine_position = Learned2DPositionEncoding(*fine_grid_shape, transformer_dim)
        self.fine_queries = nn.Parameter(
            torch.empty(1, fine_grid_shape[0] * fine_grid_shape[1], transformer_dim)
        )
        layer = nn.TransformerDecoderLayer(
            transformer_dim,
            transformer_heads,
            dim_feedforward,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerDecoder(
            layer, transformer_layers, norm=nn.LayerNorm(transformer_dim)
        )
        self.token_head = nn.Linear(transformer_dim, fine_codebook_size)
        self.fine_embedding_projection = nn.Linear(fine_embedding_dim, fine_embedding_dim)
        nn.init.normal_(self.fine_queries, std=0.02)
        nn.init.xavier_uniform_(self.coarse_projection.weight)
        nn.init.zeros_(self.coarse_projection.bias)
        nn.init.xavier_uniform_(self.token_head.weight)
        nn.init.zeros_(self.token_head.bias)
        nn.init.eye_(self.fine_embedding_projection.weight)
        nn.init.zeros_(self.fine_embedding_projection.bias)

    def forward(self, coarse_embeddings, fine_codebook):
        batch = coarse_embeddings.shape[0]
        coarse_hidden = self.coarse_projection(coarse_embeddings)
        memory = coarse_hidden + self.coarse_position(batch)
        spatial = coarse_hidden.reshape(
            batch, *self.coarse_grid_shape, self.transformer_dim
        ).permute(0, 3, 1, 2)
        aligned = bilinear_interpolate_2d(spatial, size=self.fine_grid_shape)
        aligned = aligned.permute(0, 2, 3, 1).reshape(batch, -1, self.transformer_dim)
        queries = self.fine_queries.expand(batch, -1, -1) + self.fine_position(batch) + aligned
        if self.gradient_checkpointing and self.training:
            hidden = queries
            for layer in self.transformer.layers:
                hidden = checkpoint(layer, hidden, memory, use_reentrant=False)
            hidden = self.transformer.norm(hidden)
        else:
            hidden = self.transformer(queries, memory)
        logits = self.token_head(hidden)
        embeddings = torch.matmul(F.softmax(logits, dim=-1), fine_codebook)
        return self.fine_embedding_projection(embeddings), logits

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torch.nn as nn


class PerceptionLoss(nn.Module):
    """
    MSE loss between:
      - projected generator hidden states (B, L, hidden_size)
      - frozen codebook embeddings indexed by input_ids (B, L, hidden_size)

    Notes:
      - This implementation assumes `input_ids` are image-code tokens offset by `token_offset`.
      - Set `seq_len` to match your image token count (default: 1024).
    """

    def __init__(
        self,
        embed_path: Union[str, Path] = "./src/PeSFT/perception_module.ckpt",
        *,
        vocab_size: int = 8192,
        hidden_size: int = 256,
        generator_hidden_size: int = 4096,
        seq_len: int = 1024,
        token_offset: int = 4,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.generator_hidden_size = int(generator_hidden_size)
        self.seq_len = int(seq_len)
        self.token_offset = int(token_offset)

        embed_path = Path(embed_path)
        embedding = torch.load(embed_path, map_location="cpu")

        if not torch.is_tensor(embedding):
            raise TypeError(f"Expected a tensor in embed_path, got {type(embedding)}.")
        if embedding.dim() != 2:
            raise ValueError(f"Expected embedding to be 2D (vocab, hidden), got {tuple(embedding.shape)}.")
        if embedding.shape[0] != self.vocab_size or embedding.shape[1] != self.hidden_size:
            raise ValueError(
                f"Embedding shape {tuple(embedding.shape)} does not match "
                f"(vocab_size={self.vocab_size}, hidden_size={self.hidden_size})."
            )

        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True).to(dtype=dtype)
        self.states_to_hidden = nn.Linear(self.generator_hidden_size, self.hidden_size, dtype=dtype)
        self.criterion = nn.MSELoss()

    def forward(self, input_ids: torch.Tensor, generated_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
          input_ids: (B, L) image-code token ids
          generated_hidden_states: (B, L, generator_hidden_size)

        Returns:
          scalar loss tensor
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D (B, L), got {tuple(input_ids.shape)}.")
        if generated_hidden_states.dim() != 3:
            raise ValueError(
                f"generated_hidden_states must be 3D (B, L, H), got {tuple(generated_hidden_states.shape)}."
            )

        bsz, seq_len = input_ids.shape
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}.")
        if generated_hidden_states.shape[0] != bsz or generated_hidden_states.shape[1] != seq_len:
            raise ValueError("generated_hidden_states must align with input_ids on (B, L).")
        if generated_hidden_states.shape[2] != self.generator_hidden_size:
            raise ValueError(
                f"Expected generator_hidden_size={self.generator_hidden_size}, got {generated_hidden_states.shape[2]}."
            )

        # Convert token ids to embedding indices.
        # Keep indices on the same device as embedding weights for efficient lookup.
        indices = (input_ids.to(torch.long) - self.token_offset).to(self.embedding.weight.device)

        if indices.min().item() < 0 or indices.max().item() >= self.vocab_size:
            raise ValueError(
                f"Embedding indices out of range after offset={self.token_offset}: "
                f"min={int(indices.min())}, max={int(indices.max())}, vocab_size={self.vocab_size}."
            )

        labels = self.embedding(indices)  # (B, L, hidden_size)

        # Project generator hidden states into embedding space.
        features = self.states_to_hidden(generated_hidden_states.to(self.states_to_hidden.weight.device))  # (B, L, hidden)

        # Ensure both tensors are on same device/dtype for MSE.
        labels = labels.to(device=features.device, dtype=features.dtype)
        return self.criterion(features, labels)

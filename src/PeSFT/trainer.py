from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
from transformers import Trainer

LOGGER = logging.getLogger(__name__)


def extract_image_segments(
    input_ids: torch.Tensor,
    last_hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    boi_token_id: int = 8197,
    eoi_token_id: int = 8196,
    image_token_count: int = 1024,
    ignore_label_id: int = -100,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extract fixed-length image token segments from a batch.

    Assumptions:
      - An image segment begins with a BOI token (boi_token_id),
        followed by exactly `image_token_count` image tokens.
      - A sample may contain multiple images.
      - Only images whose BOI position is NOT ignored by labels are used
        (labels[batch, boi_pos] != ignore_label_id).

    Args:
      input_ids:   (B, T)
      last_hidden: (B, T, H)  last-layer hidden states aligned with input_ids
      labels:      (B, T)

    Returns:
      image_ids:  (N, image_token_count) or None
      image_hids: (N, image_token_count, H) or None
    """
    if input_ids.dim() != 2 or labels.dim() != 2:
        raise ValueError("input_ids and labels must be 2D tensors with shape (B, T).")
    if last_hidden.dim() != 3:
        raise ValueError("last_hidden must be a 3D tensor with shape (B, T, H).")

    bsz, seq_len = input_ids.shape
    if last_hidden.shape[0] != bsz or last_hidden.shape[1] != seq_len:
        raise ValueError("last_hidden must align with input_ids on dimensions (B, T).")

    boi_positions = torch.nonzero(input_ids == boi_token_id, as_tuple=False)  # (K, 2)
    if boi_positions.numel() == 0:
        return None, None

    ids_list: List[torch.Tensor] = []
    hids_list: List[torch.Tensor] = []

    for batch_id, boi_pos in boi_positions.tolist():
        if int(labels[batch_id, boi_pos].item()) == ignore_label_id:
            continue

        start = boi_pos + 1
        end = start + image_token_count
        if end > seq_len:
            continue

        ids_list.append(input_ids[batch_id, start:end])
        hids_list.append(last_hidden[batch_id, start:end, :])

    if not ids_list:
        return None, None

    image_ids = torch.stack(ids_list, dim=0)            # (N, L)
    image_hidden = torch.stack(hids_list, dim=0)        # (N, L, H)
    return image_ids, image_hidden


class PerceptionTrainer(Trainer):
    """
    A Trainer that adds an auxiliary perceptual loss on extracted image token segments.

    The base model is expected to return:
      - outputs.loss (cross-entropy)
      - outputs.hidden_states (when output_hidden_states=True)

    The loss network is expected to accept:
      loss_net(input_ids=..., generated_hidden_states=...) -> scalar tensor
    """

    def __init__(
        self,
        loss_net,
        *args,
        boi_token_id: int = 8197,
        eoi_token_id: int = 8196,
        image_token_count: int = 1024,
        ignore_label_id: int = -100,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.loss_net = loss_net.to(self.args.device)

        self.boi_token_id = int(boi_token_id)
        self.eoi_token_id = int(eoi_token_id)  # kept for compatibility
        self.image_token_count = int(image_token_count)
        self.ignore_label_id = int(ignore_label_id)

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        outputs = model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
            attention_mask=inputs.get("attention_mask", None),
            output_hidden_states=True,
        )

        local_loss = outputs.loss
        last_hidden = outputs.hidden_states[-1]

        input_ids = inputs["input_ids"]
        labels = inputs["labels"]

        if last_hidden.shape[:2] != input_ids.shape:
            raise ValueError(
                f"Hidden states shape {tuple(last_hidden.shape[:2])} "
                f"does not match input_ids shape {tuple(input_ids.shape)}."
            )

        image_ids, image_hids = extract_image_segments(
            input_ids=input_ids,
            last_hidden=last_hidden,
            labels=labels,
            boi_token_id=self.boi_token_id,
            eoi_token_id=self.eoi_token_id,
            image_token_count=self.image_token_count,
            ignore_label_id=self.ignore_label_id,
        )

        if image_ids is None or image_hids is None:
            global_loss = torch.zeros_like(local_loss)
        else:
            image_ids = image_ids.to(device=input_ids.device, dtype=torch.long)
            image_hids = image_hids.to(device=last_hidden.device, dtype=last_hidden.dtype)

            global_loss = self.loss_net(
                input_ids=image_ids,
                generated_hidden_states=image_hids,
            )

            if not torch.is_tensor(global_loss):
                global_loss = torch.tensor(
                    float(global_loss), device=local_loss.device, dtype=local_loss.dtype
                )
            else:
                global_loss = global_loss.to(device=local_loss.device, dtype=local_loss.dtype)

            if global_loss.dim() != 0:
                global_loss = global_loss.mean()

        total_loss = local_loss + global_loss

        logging_steps = max(int(getattr(self.args, "logging_steps", 0) or 0), 0)
        if logging_steps > 0 and (self.state.global_step % logging_steps == 0):
            self.log(
                {
                    "loss_total": float(total_loss.detach().cpu()),
                    "loss_local": float(local_loss.detach().cpu()),
                    "loss_global": float(global_loss.detach().cpu()),
                }
            )

        return (total_loss, outputs) if return_outputs else total_loss

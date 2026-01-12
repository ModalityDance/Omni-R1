#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import random
import re
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    ChameleonForConditionalGeneration,
    ChameleonProcessor,
    TrainingArguments,
)

from perception import PerceptionLoss
from trainer import PerceptionTrainer

LOGGER = logging.getLogger("train.entrypoint")
IGNORE_LABEL_ID = -100

LEAD_PHRASES: List[str] = [
    "Let's think step-by-step.",
    "Let's reason step-by-step.",
    "Let's think this through.",
    "Let's reason this through.",
    "Let's work it out step-by-step.",
    "Let's break it down step-by-step.",
    "Step-by-step reasoning follows.",
    "Reasoning step-by-step begins.",
]


# ================================================================
#                           UTILITIES
# ================================================================
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed(local_rank: int) -> None:
    """Initialize torch.distributed if launched with DDP/DeepSpeed."""
    if local_rank == -1:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("local_rank is set but CUDA is not available.")
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is not available in this build.")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")


def write_deepspeed_config(ds_config_json: str) -> str:
    """
    Write DeepSpeed config JSON string into a temporary file and return its path.
    The file will be removed automatically at process exit.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        path = tmp.name
        json.dump(json.loads(ds_config_json), tmp)

    def _cleanup() -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    atexit.register(_cleanup)
    return path


def resolve_reserved_token_ids(tokenizer) -> Dict[str, int]:
    """
    Resolve reserved token IDs from tokenizer vocab to avoid hard-coding.
    Falls back to known IDs if the tokenizer doesn't know these tokens.
    """
    mapping = {
        "R_THOUGHT_OPEN": "<reserved12856>",
        "R_THOUGHT_CLOSE": "<reserved12857>",
        "R_ANS_OPEN": "<reserved12866>",
        "R_ANS_CLOSE": "<reserved12867>",
    }

    ids: Dict[str, int] = {}
    for key, tok in mapping.items():
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is None or tok_id == tokenizer.unk_token_id:
            # Fallback to the original hard-coded IDs (keeps backward behavior)
            fallback = {
                "R_THOUGHT_OPEN": 12860,
                "R_THOUGHT_CLOSE": 12861,
                "R_ANS_OPEN": 12870,
                "R_ANS_CLOSE": 12871,
            }
            tok_id = fallback[key]
        ids[key] = int(tok_id)
    return ids


# ================================================================
#                           DATASET
# ================================================================
@dataclass
class DatasetConfig:
    max_length: int = 8192
    mode: str = "templated"  # "templated" or "plain"
    include_rationale_in_plain: bool = False

    # Tokenization / formatting (templated mode)
    prefix_text: str = "You are a helpful assistant.\nUser: "
    suffix_text: str = (
        " Think with images first. The image reasoning process and answer are enclosed within "
        "<reserved12856> <reserved12857> and <reserved12866> <reserved12867> XML tags, respectively.\n"
        "Assistant:"
    )

    # Image placeholder syntax: <image_start>[key]<image_end>
    image_pattern: str = r"<image_start>\[(.*?)\]<image_end>"


class JsonlSFTDataset(Dataset):
    """
    Two modes:

    1) templated:
       BOS + prefix + question(with images) + suffix +
       lead_phrase +
       R_THOUGHT_OPEN + rationale(with images) + R_THOUGHT_CLOSE +
       R_ANS_OPEN + answer + R_ANS_CLOSE + EOS

    2) plain:
       BOS + question(with images) + response + EOS
       where response is either:
         - answer only
         - (rationale + "\\n" + answer) if include_rationale_in_plain=True

    Notes:
      - Image tokens are read from the JSON object by key referenced in the placeholder.
      - Prompt tokens are masked in labels with IGNORE_LABEL_ID.
    """

    def __init__(self, jsonl_dir: str, tokenizer, cfg: DatasetConfig, seed: int = 42):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.data: List[Dict[str, torch.Tensor]] = []

        mode = str(cfg.mode).strip().lower()
        if mode not in {"templated", "plain"}:
            raise ValueError(f"Unsupported mode: {cfg.mode}")
        self.mode = mode

        self.image_re = re.compile(cfg.image_pattern)

        self.bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

        self._rng = random.Random(seed)

        files = sorted(
            [os.path.join(jsonl_dir, f) for f in os.listdir(jsonl_dir) if f.endswith(".jsonl")]
        )
        for path in files:
            self._load_one_file(path)

        self._rng.shuffle(self.data)
        LOGGER.info("Loaded %d samples from %d jsonl files.", len(self.data), len(files))

    def _encode_text(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _encode_with_images(self, sample: dict, text: str) -> Optional[List[int]]:
        """
        Replace <image_start>[key]<image_end> with sample[key] tokens.
        Returns None if any referenced key is missing.
        """
        out: List[int] = []
        last_end = 0

        for m in self.image_re.finditer(text):
            start, end = m.span()
            if start > last_end:
                out.extend(self._encode_text(text[last_end:start]))

            key = m.group(1)
            image_tokens = sample.get(key)
            if image_tokens is None:
                return None
            out.extend(image_tokens)
            last_end = end

        if last_end < len(text):
            out.extend(self._encode_text(text[last_end:]))

        return out

    def _load_one_file(self, path: str) -> None:
        reserved_ids = None  # lazily resolved only if needed
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                sample = json.loads(line)

                question = (sample.get("Question") or "").strip()
                rationale = (sample.get("Text Reasoning Trace") or "").strip()
                answer = (sample.get("Final Answer") or "").strip()

                if not question or not answer:
                    continue
                if self.mode == "templated" and not rationale:
                    continue

                q_ids = self._encode_with_images(sample, question)
                if q_ids is None:
                    continue

                a_ids = self._encode_text(answer)

                # --------------------
                # plain mode
                # --------------------
                if self.mode == "plain":
                    if self.cfg.include_rationale_in_plain and rationale:
                        resp_text = f"{rationale}\n{answer}"
                        r_ids = self._encode_with_images(sample, resp_text)
                        if r_ids is None:
                            continue
                    else:
                        r_ids = a_ids

                    input_ids = [self.bos_id] + q_ids + r_ids + [self.eos_id]
                    if len(input_ids) > self.cfg.max_length:
                        continue

                    labels = [IGNORE_LABEL_ID] * (1 + len(q_ids)) + r_ids + [self.eos_id]
                    self.data.append(
                        {
                            "input_ids": torch.tensor(input_ids, dtype=torch.long),
                            "labels": torch.tensor(labels, dtype=torch.long),
                        }
                    )
                    continue

                # --------------------
                # templated mode
                # --------------------
                r_ids = self._encode_with_images(sample, rationale)
                if r_ids is None:
                    continue

                if reserved_ids is None:
                    reserved_ids = resolve_reserved_token_ids(self.tokenizer)

                prefix_ids = self._encode_text(self.cfg.prefix_text)
                suffix_ids = self._encode_text(self.cfg.suffix_text)

                prompt_ids = [self.bos_id] + prefix_ids + q_ids + suffix_ids
                lead_ids = self._encode_text(self._rng.choice(LEAD_PHRASES))

                completion_ids = (
                    lead_ids
                    + [reserved_ids["R_THOUGHT_OPEN"]]
                    + r_ids
                    + [reserved_ids["R_THOUGHT_CLOSE"]]
                    + [reserved_ids["R_ANS_OPEN"]]
                    + a_ids
                    + [reserved_ids["R_ANS_CLOSE"], self.eos_id]
                )

                input_ids = prompt_ids + completion_ids
                if len(input_ids) > self.cfg.max_length:
                    continue

                labels = [IGNORE_LABEL_ID] * len(prompt_ids) + completion_ids
                self.data.append(
                    {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }
                )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]


# ================================================================
#                           COLLATOR
# ================================================================
def build_collator(pad_id: int, label_pad_id: int = IGNORE_LABEL_ID) -> Callable:
    def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]

        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=label_pad_id)

        attention_mask = (input_ids_padded != pad_id).long()
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
        }

    return collate


# ================================================================
#                           ARGS
# ================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT training entrypoint (Chameleon + PerceptionLoss).")

    p.add_argument("--model_path", type=str, required=True, help="Base model path or HF repo id.")
    p.add_argument("--output_path", type=str, required=True, help="Directory to save checkpoints.")
    p.add_argument("--json_dir", type=str, required=True, help="Directory containing *.jsonl files.")
    p.add_argument(
        "--deepspeed_config_json",
        type=str,
        required=True,
        help="DeepSpeed config as a JSON string (will be written to a temp file).",
    )

    p.add_argument("--learning_rate", type=float, required=True)
    p.add_argument("--gradient_accumulation_steps", type=int, required=True)

    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--local_rank", type=int, default=-1)

    # dataset / formatting
    p.add_argument("--max_length", type=int, default=5100)
    p.add_argument("--mode", type=str, default="templated", choices=["templated", "plain"])
    p.add_argument(
        "--plain_include_rationale",
        action="store_true",
        help="Plain mode only: include rationale before answer (default: off).",
    )

    # (optional) override template text in templated mode
    p.add_argument("--prefix_text", type=str, default=None)
    p.add_argument("--suffix_text", type=str, default=None)

    # reproducibility / reporting
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "none"])

    return p.parse_args()


# ================================================================
#                           MAIN
# ================================================================
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()

    set_global_seed(int(args.seed))
    init_distributed(args.local_rank)

    ds_config_path = write_deepspeed_config(args.deepspeed_config_json)

    LOGGER.info("Loading model from: %s", args.model_path)
    model = ChameleonForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = ChameleonProcessor.from_pretrained(args.model_path)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
    collator = build_collator(pad_id=pad_id, label_pad_id=IGNORE_LABEL_ID)

    ds_cfg = DatasetConfig(
        max_length=int(args.max_length),
        mode=str(args.mode),
        include_rationale_in_plain=bool(args.plain_include_rationale),
    )
    if args.prefix_text is not None:
        ds_cfg.prefix_text = str(args.prefix_text)
    if args.suffix_text is not None:
        ds_cfg.suffix_text = str(args.suffix_text)

    train_dataset = JsonlSFTDataset(args.json_dir, tokenizer, ds_cfg, seed=int(args.seed))

    # Loss network (configure in your PerceptionLoss implementation)
    loss_net = PerceptionLoss()

    report_to = [] if args.report_to == "none" else [args.report_to]

    training_args = TrainingArguments(
        output_dir=args.output_path,
        learning_rate=float(args.learning_rate),
        num_train_epochs=int(args.num_train_epochs),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        bf16=True,
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        deepspeed=ds_config_path,
        report_to=report_to,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
        local_rank=int(args.local_rank),
    )

    trainer = PerceptionTrainer(
        loss_net=loss_net,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_path)

    if processor is not None:
        processor.save_pretrained(args.output_path)

    LOGGER.info("Model saved to %s", args.output_path)


if __name__ == "__main__":
    main()

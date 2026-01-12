#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import random
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from collections.abc import Mapping

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

IMG_TOKEN = "<image>"
SENTINEL_DONE = {"__type__": "DONE"}


# --------------------------- I/O utilities ----------------------------------

def collect_parquets(root: str) -> list[str]:
    """
    Collect parquet files.
    Supports:
      - a single parquet file path
      - a directory (recursive scan)
    """
    if not root:
        raise FileNotFoundError("Empty parquet path.")

    if os.path.isfile(root):
        if not root.lower().endswith(".parquet"):
            raise FileNotFoundError(f"Not a parquet file: {root}")
        logger.info("[Dataset] Using single parquet: %s", root)
        return [root]

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Parquet path not found: {root}")

    out: list[str] = []
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".parquet"):
                out.append(os.path.join(r, f))

    if not out:
        raise FileNotFoundError("No .parquet files found under the given directory.")

    out.sort()
    logger.info("[Dataset] Collected %d parquet files", len(out))
    return out


# --------------------------- JSON-safe conversion ----------------------------

def json_safe_value(v: Any) -> Any:
    """Convert common numpy/pandas/torch types into JSON-serializable Python primitives."""
    try:
        if v is None or isinstance(v, (str, int, float, bool)):
            return v

        if isinstance(v, (np.generic,)):
            return v.item()

        if isinstance(v, np.ndarray):
            return v.tolist()

        if torch.is_tensor(v):
            return v.detach().cpu().tolist()

        if isinstance(v, Mapping):
            return {k: json_safe_value(val) for k, val in v.items()}

        if isinstance(v, (list, tuple)):
            return [json_safe_value(x) for x in v]

        return str(v)
    except Exception:
        return str(v)


def build_raw_from_series(
    s: pd.Series,
    *,
    drop_keys: Optional[List[str]] = None,
) -> dict:
    """
    Build a JSON-safe dict from a pandas Series, dropping specified fields.
    Default drops 'image' (and common variants).
    """
    d = s.to_dict()

    drop = drop_keys or ["image", "images"]
    drop_set = {k.lower() for k in drop}

    for k in list(d.keys()):
        if str(k).lower() in drop_set:
            d.pop(k, None)

    return {k: json_safe_value(v) for k, v in d.items()}


# --------------------------- Base64 helpers ----------------------------------

_BASE64_CHARS_RE = None

def _looks_like_base64_str(s: str) -> bool:
    """
    Heuristic: sample chars are base64 alphabet, and length is "large enough" to be an image.
    We keep it permissive and rely on PIL open success as the true check.
    """
    global _BASE64_CHARS_RE
    if _BASE64_CHARS_RE is None:
        import re
        _BASE64_CHARS_RE = re.compile(r'^[A-Za-z0-9+/=\s]+$')

    if not isinstance(s, str):
        return False
    if len(s) < 256:  # images base64 are usually much longer than this
        return False
    sample = s[:4096]
    return bool(_BASE64_CHARS_RE.match(sample))


def _try_open_base64_image(s: str) -> Optional[Image.Image]:
    """
    Try:
      - data:image/...;base64,....
      - raw base64 string
    Return PIL.Image or None.
    """
    try:
        if not isinstance(s, str):
            return None

        if s.startswith("data:image/"):
            comma = s.find(",")
            if comma == -1:
                return None
            b64 = s[comma + 1 :]
            data = base64.b64decode(b64, validate=False)
            return Image.open(io.BytesIO(data)).convert("RGB")

        if _looks_like_base64_str(s):
            b64 = "".join(s.split())  # remove whitespace/newlines
            data = base64.b64decode(b64, validate=False)
            return Image.open(io.BytesIO(data)).convert("RGB")

        return None
    except Exception:
        return None


# --------------------------- Image normalization -----------------------------

def to_pil(img_like: Any, *, tlts_token: str = "<TLTS>") -> Optional[Image.Image]:
    """
    Best-effort conversion to PIL.Image.Image.
    Supports:
      - PIL.Image
      - HF datasets Image-like objects with .to_pil()
      - objects/dicts with {bytes,path}
      - bytes / bytearray / memoryview
      - numpy arrays
      - torch tensors
      - data URLs (data:image/...;base64,...)
      - pure base64 strings (no prefix)
      - file path strings
      - tlts_token (treated as missing image)
    """
    try:
        if img_like is None:
            return None

        # TLTS placeholder -> treat as no image
        if isinstance(img_like, str) and img_like == tlts_token:
            return None

        if isinstance(img_like, Image.Image):
            return img_like.convert("RGB") if img_like.mode != "RGB" else img_like

        if hasattr(img_like, "to_pil") and callable(img_like.to_pil):
            pil = img_like.to_pil()
            return pil.convert("RGB") if pil.mode != "RGB" else pil

        if hasattr(img_like, "image") and isinstance(getattr(img_like, "image"), Image.Image):
            pil = getattr(img_like, "image")
            return pil.convert("RGB") if pil.mode != "RGB" else pil

        # objects with bytes/path
        if hasattr(img_like, "bytes") or hasattr(img_like, "path"):
            b = getattr(img_like, "bytes", None)
            if b is not None:
                try:
                    return Image.open(io.BytesIO(bytes(b))).convert("RGB")
                except Exception:
                    pass

            p = getattr(img_like, "path", None)
            if p:
                try:
                    return Image.open(p).convert("RGB")
                except Exception:
                    pass

        # dict {"bytes":..., "path":...}
        if isinstance(img_like, Mapping):
            b = img_like.get("bytes")
            if b is not None:
                try:
                    return Image.open(io.BytesIO(bytes(b))).convert("RGB")
                except Exception:
                    pass

            p = img_like.get("path")
            if p:
                try:
                    return Image.open(p).convert("RGB")
                except Exception:
                    pass

        # string: data URL / pure base64 / file path
        if isinstance(img_like, str):
            # 1) dataURL or pure base64
            pil = _try_open_base64_image(img_like)
            if pil is not None:
                return pil

            # 2) file path
            try:
                return Image.open(img_like).convert("RGB")
            except Exception:
                return None

        # raw bytes
        if isinstance(img_like, (bytes, bytearray, memoryview)):
            return Image.open(io.BytesIO(bytes(img_like))).convert("RGB")

        # numpy
        if isinstance(img_like, np.ndarray):
            arr = img_like
            if arr.ndim == 2:
                return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
            if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                if arr.shape[2] == 1:
                    arr = np.repeat(arr, 3, axis=2)
                return Image.fromarray(arr[:, :, :3], mode="RGB")
            return None

        # torch tensor (H, W, C)
        if torch.is_tensor(img_like):
            ten = img_like.detach().cpu()
            if ten.ndim == 2:
                ten = ten.unsqueeze(-1).repeat(1, 1, 3)
            if ten.ndim == 3 and ten.shape[2] in (1, 3, 4):
                if ten.dtype != torch.uint8:
                    ten = ten.clamp(0, 255).to(torch.uint8)
                if ten.shape[2] == 1:
                    ten = ten.repeat(1, 1, 3)
                arr = ten.numpy()
                return Image.fromarray(arr[:, :, :3], mode="RGB")
            return None

        return None
    except Exception as e:
        logger.warning("Image normalization failed: %s", e)
        return None


def normalize_image_field(v: Any) -> Optional[list[Any]]:
    """Normalize a row's image field into a list-like container; return None if empty."""
    if v is None:
        return None
    if isinstance(v, (list, tuple, pd.Series)):
        lst = [x for x in v if x is not None]
        return lst if lst else None
    if isinstance(v, np.ndarray) and v.dtype != object:
        return [v]
    return [v]


# --------------------------- Placeholder alignment ----------------------------

def align_placeholders(
    text: str,
    images: Optional[List[Image.Image]],
    *,
    auto_insert: bool = True,
    auto_remove: bool = True,
) -> tuple[Optional[str], Optional[List[Image.Image]]]:
    """
    Ensure number of <image> placeholders matches number of images.
    """
    n_img = len(images) if images else 0
    n_ph = text.count(IMG_TOKEN)

    if n_img == 0 and n_ph == 0:
        return text, None

    if n_img == 0 and n_ph > 0:
        return "".join(text.split(IMG_TOKEN)), None

    if n_ph < n_img:
        if auto_insert:
            t = text.rstrip()
            if not t.endswith(" "):
                t += " "
            t += " ".join([IMG_TOKEN] * (n_img - n_ph))
            text = t
            n_ph = n_img
        else:
            images = images[:n_ph]
            n_img = len(images)

    if n_ph > n_img:
        if auto_remove:
            parts = text.split(IMG_TOKEN)
            kept = parts[0]
            for i in range(n_img):
                kept += IMG_TOKEN + parts[i + 1]
            kept += "".join(parts[n_img + 1 :])
            text = kept
        else:
            return None, None

    return text, images


# --------------------------- Dataset -----------------------------------------

@dataclass
class RowItem:
    global_index: int
    prompt: str
    images: Optional[List[Image.Image]]
    source: str
    dataset: str
    local_index: int
    raw: dict


class MMParquetDataset(Dataset):
    """
    Parquet -> vLLM-ready prompt + PIL images
    Designed for columns: image(base64) / question / answer
    """

    def __init__(
        self,
        parquet_path_or_dir: str,
        *,
        question_col: str = "question",
        image_col: str = "image",
        answer_col: str = "answer",
        tlts_token: str = "<TLTS>",
        skip_if_no_image: bool = False,  # True: 没有图（或 TLTS/坏图）就跳过
        use_template: bool = True,
        auto_insert_placeholders: bool = True,
        auto_remove_placeholders: bool = True,
        shuffle: bool = True,
        seed: int = 0,
    ):
        paths = collect_parquets(parquet_path_or_dir)

        self.prefix = "You are a helpful assistant.\nUser: "
        self.suffix = (
            " Think with images first. The image reasoning process and the final answer are enclosed "
            "within <reserved12856> <reserved12857> and <reserved12866> <reserved12867> XML tags, respectively.\nAssistant:"
        )

        rng = random.Random(seed)
        rows: list[RowItem] = []
        global_idx = 0

        for fp in paths:
            try:
                df = pd.read_parquet(fp)
            except Exception as e:
                logger.warning("Failed to read parquet: %s (%s)", fp, e)
                continue

            missing = [c for c in (question_col, image_col) if c not in df.columns]
            if missing:
                logger.warning("[Skip] %s: missing columns: %s", fp, missing)
                continue

            rel = os.path.relpath(fp, parquet_path_or_dir).replace("\\", "/") if os.path.isdir(parquet_path_or_dir) else os.path.basename(fp)
            fname = os.path.basename(fp)

            for local_idx in range(len(df)):
                # robust row access
                row_s = df.iloc[local_idx]

                q = row_s.get(question_col, None)
                if not isinstance(q, str) or not q.strip():
                    continue
                question = q.strip()

                # raw fields (keep GT answer for later evaluation)
                raw = build_raw_from_series(row_s, drop_keys=[image_col])
                if answer_col in df.columns:
                    raw["_gt_answer"] = json_safe_value(row_s.get(answer_col, None))

                # normalize images
                img_val = row_s.get(image_col, None)
                img_field = normalize_image_field(img_val)

                imgs_pil: Optional[List[Image.Image]] = None
                if img_field:
                    pil_list: List[Image.Image] = []
                    for obj in img_field:
                        pil = to_pil(obj, tlts_token=tlts_token)
                        if pil is not None:
                            pil_list.append(pil)
                    imgs_pil = pil_list if pil_list else None

                # if you require image but it is missing/un-decodable
                if skip_if_no_image and (imgs_pil is None or len(imgs_pil) == 0):
                    continue

                # Build prompt for VQA: ensure <image> at start if there is an image and question doesn't include it
                prompt = question
                if imgs_pil and IMG_TOKEN not in prompt:
                    prompt = IMG_TOKEN + prompt

                # Align placeholders count with images count
                prompt_aligned, imgs_aligned = align_placeholders(
                    prompt,
                    imgs_pil,
                    auto_insert=auto_insert_placeholders,
                    auto_remove=auto_remove_placeholders,
                )
                if prompt_aligned is None:
                    logger.warning("[Skip sample] Placeholder/image mismatch: %s | row=%d", rel, local_idx)
                    continue

                prompt_final = (self.prefix + prompt_aligned + self.suffix) if use_template else prompt_aligned

                rows.append(
                    RowItem(
                        global_index=global_idx,
                        prompt=prompt_final,
                        images=imgs_aligned,
                        source=rel,
                        dataset=fname,
                        local_index=local_idx,
                        raw=raw,
                    )
                )
                global_idx += 1

        if not rows:
            raise RuntimeError("No valid samples were built from parquet(s).")

        if shuffle:
            rng.shuffle(rows)

        self.data = rows
        logger.info("[Dataset] Prepared %d samples", len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict:
        r = self.data[i]
        return {
            "vllm_prompt": r.prompt,
            "vllm_images": r.images,  # None or List[PIL.Image]
            "index": r.global_index,
            "source": r.source,
            "dataset": r.dataset,
            "local_index": r.local_index,
            "raw": r.raw,
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "vllm_prompt": [b["vllm_prompt"] for b in batch],
        "vllm_images": [b["vllm_images"] for b in batch],
        "index": [b["index"] for b in batch],
        "source": [b["source"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
        "local_index": [b["local_index"] for b in batch],
        "raw": [b["raw"] for b in batch],
    }


# --------------------------- vLLM inference ----------------------------------

def generate_batch(
    llm: LLM,
    prompts: List[str],
    images_batch: List[Optional[List[Image.Image]]],
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> List[str]:
    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        skip_special_tokens=False,
    )

    reqs: list[Any] = []
    for p, imgs in zip(prompts, images_batch):
        if imgs and len(imgs) > 0:
            reqs.append({"prompt": p, "multi_modal_data": {"image": imgs}})
        else:
            reqs.append(p)

    outs = llm.generate(reqs, sampling_params=params)

    texts: List[str] = []
    for o in outs:
        if not o.outputs:
            texts.append("")
        else:
            texts.append(o.outputs[0].text.strip())
    return texts


def infer_streaming(
    llm: LLM,
    dl: DataLoader,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    micro_batch_size: Optional[int],
    emit: Callable[[dict], None],
):
    for batch in tqdm(dl, desc="inference", dynamic_ncols=True):
        prompts = batch["vllm_prompt"]
        images_list = batch["vllm_images"]

        step = micro_batch_size or len(prompts)
        for i in range(0, len(prompts), step):
            seg_prompts = prompts[i : i + step]
            seg_images = images_list[i : i + step]

            try:
                preds = generate_batch(
                    llm,
                    seg_prompts,
                    seg_images,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as e:
                preds = [f"<ERROR: {type(e).__name__}: {str(e)}>" for _ in seg_prompts]

            idxs = batch["index"][i : i + step]
            srcs = batch["source"][i : i + step]
            dsets = batch["dataset"][i : i + step]
            locs = batch["local_index"][i : i + step]
            raws = batch["raw"][i : i + step]

            for idx, src, dset, loc, pred, raw in zip(idxs, srcs, dsets, locs, preds, raws):
                emit(
                    {
                        "index": int(idx),
                        "source": src,
                        "dataset": dset,
                        "local_index": int(loc),
                        "raw": raw,  # includes _gt_answer if present
                        "prediction": pred,
                    }
                )


# --------------------------- Multiprocessing writer --------------------------

def writer_process(
    output_path: str,
    q,
    summary_q,
    expected_done: int,
    flush_every: int,
    fsync_every: int,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    wrote = 0
    dones = 0
    flushed = 0

    with open(output_path, "a", encoding="utf-8") as fout:
        while True:
            item = q.get()
            if item == SENTINEL_DONE:
                dones += 1
                if dones >= expected_done:
                    break
                continue

            fout.write(
                json.dumps(
                    {
                        "index": item["index"],
                        "source": item.get("source"),
                        "dataset": item.get("dataset"),
                        "local_index": item.get("local_index"),
                        "raw": item.get("raw"),
                        "prediction": item.get("prediction"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            wrote += 1

            if flush_every > 0 and (wrote % flush_every == 0):
                fout.flush()
                flushed += 1
                if fsync_every > 0 and (flushed % fsync_every == 0):
                    try:
                        os.fsync(fout.fileno())
                    except Exception:
                        pass

        try:
            fout.flush()
            if fsync_every == 0:
                os.fsync(fout.fileno())
        except Exception:
            pass

    summary_q.put({"wrote": wrote})


# --------------------------- Sharding & device selection ---------------------

def filter_shard(dataset: MMParquetDataset, shard_id: int, num_shards: int) -> MMParquetDataset:
    if num_shards <= 1:
        return dataset
    before = len(dataset.data)
    dataset.data = [s for s in dataset.data if (s.global_index % num_shards) == shard_id]
    logger.info("[Worker %d/%d] Kept %d / %d samples", shard_id, num_shards, len(dataset.data), before)
    return dataset


def detect_devices() -> list[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env and env.strip():
        ids = [x.strip() for x in env.split(",") if x.strip() != ""]
        return ids if ids else ["0"]

    try:
        n = torch.cuda.device_count()
        return [str(i) for i in range(n)] if n > 0 else ["0"]
    except Exception:
        return ["0"]


# --------------------------- Worker ------------------------------------------

def run_worker(
    args,
    *,
    shard_id: int,
    num_shards: int,
    device_id: Optional[str],
    q,
):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        logger.info("[Worker %d] Bound to CUDA_VISIBLE_DEVICES=%s", shard_id, os.environ["CUDA_VISIBLE_DEVICES"])

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        dtype=args.dtype,
        limit_mm_per_prompt={"image": args.mm_images_per_prompt},
    )

    dataset = MMParquetDataset(
        args.parquet_path,  # file or dir
        question_col=args.question_col,
        image_col=args.image_col,
        answer_col=args.answer_col,
        tlts_token=args.tlts_token,
        skip_if_no_image=args.skip_if_no_image,
        use_template=not args.plain_prompt,
        auto_insert_placeholders=not args.no_auto_insert,
        auto_remove_placeholders=not args.no_auto_remove,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    dataset = filter_shard(dataset, shard_id, num_shards)

    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    def emit(rec: dict) -> None:
        q.put(rec)

    infer_streaming(
        llm,
        dl,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        micro_batch_size=args.micro_batch_size,
        emit=emit,
    )

    q.put(SENTINEL_DONE)


# --------------------------- Main --------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Streaming multimodal inference with vLLM over Parquet VQA data (image=base64)."
    )

    ap.add_argument("--parquet_path", required=True, help="Parquet file path OR a directory containing parquet files.")
    ap.add_argument("--model_path", required=True, help="vLLM model path or identifier.")

    ap.add_argument("--output_dir", default="results", help="Output directory for JSONL results.")
    ap.add_argument("--outfile", default="predictions.jsonl", help="Output JSONL filename (under output_dir).")

    # Column mapping for your cleaned VQA parquet
    ap.add_argument("--question_col", default="question", help="Question column name (default: question).")
    ap.add_argument("--image_col", default="image", help="Image column name (default: image).")
    ap.add_argument("--answer_col", default="answer", help="GT answer column name to keep in raw (default: answer).")

    ap.add_argument("--tlts_token", default="<TLTS>", help="Placeholder token for removed images.")
    ap.add_argument("--skip_if_no_image", action="store_true",
                    help="Skip samples if image is missing/undecodable (recommended for real MM eval).")

    # Inference parameters
    ap.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum generated tokens per sample.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    ap.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size inside each worker.")

    # Multimodal settings
    ap.add_argument("--mm_images_per_prompt", type=int, default=1, help="Max images per prompt enforced by vLLM.")
    ap.add_argument("--plain_prompt", action="store_true", help="Do not add the system template prefix/suffix.")
    ap.add_argument("--no_auto_insert", action="store_true", help="Do not auto-append missing <image> placeholders.")
    ap.add_argument("--no_auto_remove", action="store_true", help="Do not auto-remove extra <image> placeholders.")

    # Batching controls
    ap.add_argument("--batch_size", type=int, default=8, help="DataLoader batch size (macro batch).")
    ap.add_argument("--micro_batch_size", type=int, default=None, help="Micro-batch size for vLLM requests.")

    # Dataset controls
    ap.add_argument("--shuffle", action="store_true", help="Shuffle samples globally after loading.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed used when --shuffle is enabled.")

    # Writer durability controls
    ap.add_argument("--flush_every", type=int, default=1, help="Flush after every N records.")
    ap.add_argument("--fsync_every", type=int, default=0, help="fsync after every N flushes; 0 means fsync once at end.")

    args = ap.parse_args()

    devices = detect_devices()
    num_workers = max(1, len(devices))

    if num_workers > 1 and args.tp_size != 1:
        raise SystemExit("When running multiple workers, set --tp_size=1 (one GPU per worker).")

    logger.info("Detected %d GPU(s): %s", num_workers, devices)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.outfile)

    ctx = get_context("spawn")
    q = ctx.Queue(maxsize=1024)
    summary_q = ctx.Queue()

    writer = ctx.Process(
        target=writer_process,
        args=(
            output_path,
            q,
            summary_q,
            num_workers,
            max(1, args.flush_every),
            max(0, args.fsync_every),
        ),
    )
    writer.start()

    procs = []
    for rank, dev in enumerate(devices):
        p = ctx.Process(
            target=run_worker,
            kwargs=dict(
                args=args,
                shard_id=rank,
                num_shards=num_workers,
                device_id=dev,
                q=q,
            ),
        )
        p.start()
        procs.append(p)

    exit_codes = []
    for p in procs:
        p.join()
        exit_codes.append(p.exitcode)

    summary = summary_q.get()
    writer.join()

    if any(code != 0 for code in exit_codes):
        bad = [i for i, c in enumerate(exit_codes) if c != 0]
        raise SystemExit(f"Some workers failed. Exit codes: {exit_codes}. Failed workers: {bad}")

    wrote = summary.get("wrote", 0)
    print(f"✓ Completed. Wrote {wrote} records.")
    print(f"↳ Results written to: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image, ImageFile

from transformers import (
    ChameleonProcessor,
    ChameleonForConditionalGeneration,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

LOGGER = logging.getLogger("chameleon_infer")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
# ----------------------------
# Stopping criteria
# ----------------------------
class StopOnToken(StoppingCriteria):
    """Stop generation once the last generated token equals `stop_token_id`."""

    def __init__(self, stop_token_id: int):
        self.stop_token_id = int(stop_token_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return bool((input_ids[0, -1] == self.stop_token_id).item())


# ----------------------------
# Helpers
# ----------------------------
def set_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no + 1}: {e}") from e


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_load_images(image_paths: Any) -> List[Image.Image]:
    """Best-effort image loader: returns a list of valid PIL images."""
    if not image_paths:
        return []
    if isinstance(image_paths, (str, Path)):
        image_paths = [str(image_paths)]
    imgs: List[Image.Image] = []
    for p in image_paths:
        try:
            if isinstance(p, str) and os.path.exists(p):
                imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            # Skip unreadable images.
            continue
    return imgs


def normalize_name(s: str) -> str:
    """Lowercase and normalize separators for stable substring matching."""
    s = s.lower()
    s = re.sub(r"[_\-\s]+", " ", s)
    return s.strip()


def get_dtype(dtype_str: str) -> torch.dtype:
    s = dtype_str.lower().strip()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def get_special_token_ids(model: ChameleonForConditionalGeneration) -> Tuple[int, int, int]:
    """
    Returns (boi, eoi, eos) token ids.
    Chameleon usually exposes these in config.
    """
    cfg = model.config
    boi = int(getattr(cfg, "boi_token_id"))
    eoi = int(getattr(cfg, "eoi_token_id"))
    eos = int(getattr(cfg, "eos_token_id"))
    return boi, eoi, eos


def get_image_placeholder_id(processor: ChameleonProcessor) -> int:
    """
    Try to find the placeholder token id used for images in the tokenized prompt.
    This assumes the placeholder text is "<image>" and it maps to a single token.
    """
    ids = processor.tokenizer.encode("<image>", add_special_tokens=False)
    if len(ids) != 1:
        # Fallback to a common id used in some Chameleon checkpoints.
        # If your checkpoint differs, you should override this via --image-placeholder-id.
        LOGGER.warning("'<image>' does not map to a single token; fallback placeholder id may be wrong.")
        return 8711
    return int(ids[0])


def replace_image_placeholders(
    input_ids: torch.LongTensor,
    image_tokens: torch.LongTensor,
    placeholder_id: int,
) -> torch.LongTensor:
    """
    Replace placeholder tokens in `input_ids` with actual image tokens.
    Requires that the number of placeholders equals the number of image tokens.
    """
    mask = (input_ids == placeholder_id)
    needed = int(mask.sum().item())
    provided = int(image_tokens.numel())
    if needed != provided:
        raise ValueError(f"Placeholder/token mismatch: placeholders={needed}, image_tokens={provided}")
    out = input_ids.clone()
    out[mask] = image_tokens.reshape(-1).to(dtype=torch.long, device=out.device)
    return out


def model_get_image_tokens(model: ChameleonForConditionalGeneration, pixel_values: torch.Tensor) -> torch.LongTensor:
    """
    Obtain image tokens from pixel_values.
    Some checkpoints expose get_image_tokens on the top-level model; others may expose it elsewhere.
    """
    if hasattr(model, "get_image_tokens"):
        return model.get_image_tokens(pixel_values)
    if hasattr(model, "model") and hasattr(model.model, "get_image_tokens"):
        return model.model.get_image_tokens(pixel_values)
    raise AttributeError("Cannot find get_image_tokens on the model; please check your checkpoint/transformers version.")


# ----------------------------
# Token segmentation + decoding
# ----------------------------
def split_interleaved_tokens(tokens: List[int], boi: int, eoi: int) -> List[Tuple[str, List[int]]]:
    """
    Split a full token sequence into alternating text/image segments.

    Returns:
      [("text", [...]), ("image", [...]), ...]
    Note: The returned image segment excludes <boi> and <eoi> tokens.
    """
    segments: List[Tuple[str, List[int]]] = []
    current: List[int] = []
    in_image = False

    for t in tokens:
        if t == boi:
            if current:
                segments.append(("text", current))
                current = []
            in_image = True
            continue
        if t == eoi and in_image:
            segments.append(("image", current))
            current = []
            in_image = False
            continue
        current.append(t)

    if current:
        segments.append(("image" if in_image else "text", current))

    return segments


def pixels_to_pil_via_processor(pixels: torch.Tensor, processor: ChameleonProcessor) -> Image.Image:
    """
    Convert model decoded pixels [B,3,H,W] into a PIL image using the processor postprocess.
    """
    ip = getattr(processor, "image_processor", None) or getattr(processor, "feature_extractor", None)
    if ip is None:
        raise RuntimeError("processor.image_processor is not available; please use ChameleonProcessor.")

    px_uint8 = ip.postprocess(
        pixels.float(),
        do_rescale=True,
        do_unnormalize=True,
    )
    arr = px_uint8[0].permute(1, 2, 0).detach().cpu().numpy()
    return Image.fromarray(arr)


def decode_interleaved_sample(
    tokens: List[int],
    processor: ChameleonProcessor,
    model: ChameleonForConditionalGeneration,
    out_dir: Path,
    sample_id: str,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Decode tokens into text with <image_k> placeholders + saved images on disk.

    strict=True:
      - Validate image token length and token membership when possible.
      - On invalid segments, store debugging artifacts and emit <invalid_image_k>.
    """
    device = next(model.parameters()).device
    boi, eoi, _ = get_special_token_ids(model)
    segments = split_interleaved_tokens(tokens, boi, eoi)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to derive expected image code length.
    expected_len = getattr(getattr(model, "model", None), "image_seq_length", None)
    if expected_len is None and hasattr(model, "config"):
        expected_len = getattr(model.config, "image_seq_length", None)

    # Optional membership checks if vocabulary_mapping exists.
    vocab_map = getattr(model, "vocabulary_mapping", None)
    img_bpe_ids = None
    bpe2img_map = None
    if vocab_map is not None:
        try:
            img_bpe_ids = torch.tensor(vocab_map.image_token_ids, device=device, dtype=torch.long)
            bpe2img_map = vocab_map.bpe2img_mapping_tensor.to(device)
        except Exception:
            img_bpe_ids = None
            bpe2img_map = None

    text_out = ""
    image_paths: List[Optional[str]] = []
    img_index = 0

    for kind, seg in segments:
        if kind == "text":
            seg_t = torch.tensor([seg], device=device, dtype=torch.long)
            text_out += processor.batch_decode(seg_t, skip_special_tokens=False)[0]
            continue

        # Image segment
        placeholder = f"<image_{img_index}>"
        seg_t = torch.tensor([seg], device=device, dtype=torch.long)

        # Strict validations (best effort)
        if strict and expected_len is not None and seg_t.shape[1] != int(expected_len):
            debug_path = out_dir / f"image_{img_index}_invalid_len.pt"
            torch.save(
                {"tokens": seg_t.detach().cpu(), "expected_len": int(expected_len)},
                str(debug_path),
            )
            text_out += f"<invalid_image_{img_index}>"
            image_paths.append(None)
            img_index += 1
            continue

        if strict and img_bpe_ids is not None:
            isin = torch.isin(seg_t, img_bpe_ids)
            if not bool(isin.all()):
                bad_pos = (~isin).nonzero(as_tuple=False)[:20].tolist()
                debug_path = out_dir / f"image_{img_index}_non_image_bpe.pt"
                torch.save(
                    {"tokens": seg_t.detach().cpu(), "bad_pos": bad_pos},
                    str(debug_path),
                )
                text_out += f"<invalid_image_{img_index}>"
                image_paths.append(None)
                img_index += 1
                continue

        # Decode
        try:
            pixels = model.decode_image_tokens(seg_t.to(device=device, dtype=torch.long))
            pil = pixels_to_pil_via_processor(pixels, processor)
            img_path = out_dir / f"image_{img_index}.png"
            pil.save(img_path)
            image_paths.append(str(img_path))
            text_out += placeholder
        except Exception as e:
            debug_path = out_dir / f"image_{img_index}_decode_error.pt"
            payload: Dict[str, Any] = {"tokens": seg_t.detach().cpu(), "error": str(e)}
            if bpe2img_map is not None:
                try:
                    img_ids = bpe2img_map[seg_t.to(device)]
                    payload["img_ids_min"] = int(img_ids.min().item())
                    payload["img_ids_max"] = int(img_ids.max().item())
                except Exception:
                    pass
            torch.save(payload, str(debug_path))
            text_out += f"<error_image_{img_index}>"
            image_paths.append(None)

        img_index += 1

    return {"id": sample_id, "text": text_out, "images": image_paths}


# ----------------------------
# Inference engine
# ----------------------------
@dataclass
class InferenceConfig:
    input_path: Path
    model_path: Path
    processor_path: Optional[Path]
    vq_ref_path: Optional[Path]
    output_dir: Path

    device: str
    dtype: str

    max_length: int
    temperature: float
    top_p: float
    do_sample: bool

    generation_mode: str 
    max_images: Optional[int]

    image_placeholder_id: Optional[int]
    strict_decode: bool

    resume: bool
    retry_errors: bool

    priority_terms: List[str]


class ChameleonInfer:
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg

        device = get_device(cfg.device)
        dtype = get_dtype(cfg.dtype)

        processor_path = cfg.processor_path or cfg.model_path
        LOGGER.info("Loading processor from: %s", processor_path)
        self.processor = ChameleonProcessor.from_pretrained(str(processor_path))

        LOGGER.info("Loading model from: %s", cfg.model_path)
        self.model = ChameleonForConditionalGeneration.from_pretrained(
            str(cfg.model_path),
            device_map="cuda" if device.type == "cuda" else None,
            torch_dtype=dtype,
        )
        self.model.eval()

        # Move to explicit device if not using device_map (CPU or single device).
        if device.type != "cuda":
            self.model.to(device)

        # Optional VQ replacement
        if cfg.vq_ref_path is not None:
            self._replace_vq(cfg.vq_ref_path, dtype)

        self.boi, self.eoi, self.eos = get_special_token_ids(self.model)

        if cfg.image_placeholder_id is not None:
            self.image_placeholder_id = int(cfg.image_placeholder_id)
        else:
            self.image_placeholder_id = get_image_placeholder_id(self.processor)

        LOGGER.info(
            "Token ids: boi=%d, eoi=%d, eos=%d, image_placeholder_id=%d",
            self.boi,
            self.eoi,
            self.eos,
            self.image_placeholder_id,
        )

    def _replace_vq(self, ref_path: Path, dtype: torch.dtype) -> None:
        """
        Replace the model VQ module weights with a reference checkpoint VQ weights, then freeze it.
        """
        LOGGER.info("Replacing VQ weights from ref model: %s", ref_path)
        ref = ChameleonForConditionalGeneration.from_pretrained(
            str(ref_path),
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=dtype,
        )
        ref.eval()

        self.model.model.vqmodel.load_state_dict(ref.model.vqmodel.state_dict(), strict=True)

        vq = self.model.model.vqmodel
        vq.eval()
        for p in vq.parameters():
            p.requires_grad = False

        del ref
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        LOGGER.info("VQ replacement done and frozen.")

    @torch.inference_mode()
    def prepare_prompt_tokens(self, prompt: str, image_paths: Any) -> List[int]:
        """
        Build input token ids for generation.
        If images are provided, replace placeholder tokens with image tokens.
        """
        pil_images = safe_load_images(image_paths)

        # Remove placeholder when no images are provided (prevents processor errors).
        if len(pil_images) == 0 and "<image>" in prompt:
            prompt = prompt.replace("<image>", " ").strip()

        if len(pil_images) == 0:
            inputs = self.processor(
                prompt,
                padding=False,
                return_for_text_completion=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                prompt,
                images=pil_images,
                padding=False,
                return_for_text_completion=True,
                return_tensors="pt",
            )

        # BatchFeature supports .to(...)
        inputs = inputs.to(next(self.model.parameters()).device)

        input_ids = inputs["input_ids"].to(dtype=torch.long)

        if len(pil_images) > 0:
            pixel_values = inputs["pixel_values"]
            image_tokens = model_get_image_tokens(self.model, pixel_values)
            input_ids = replace_image_placeholders(input_ids, image_tokens, self.image_placeholder_id)

        tokens = input_ids[0].tolist()
        
        return tokens

    @torch.inference_mode()
    def generate_once(self, prompt_tokens: List[int]) -> List[int]:
        """
        Single-call generation. Stops on EOS or max_length.
        """
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([prompt_tokens], device=device, dtype=torch.long)

        stop_eos = StopOnToken(self.eos)

        out = self.model.generate(
            input_ids=input_ids,
            max_length=int(self.cfg.max_length),
            temperature=float(self.cfg.temperature),
            top_p=float(self.cfg.top_p),
            do_sample=bool(self.cfg.do_sample),
            pad_token_id=1,
            multimodal_generation_mode=self.cfg.generation_mode,
            stopping_criteria=StoppingCriteriaList([stop_eos]),
        )
        full = out[0].tolist()

        # Optional post-truncation by number of images (count <boi> occurrences in the full sequence).
        if self.cfg.max_images is not None:
            full = truncate_by_max_images(
                full,
                boi_id=self.boi,
                eos_id=self.eos,
                max_images=int(self.cfg.max_images),
            )

        return full

    def run_jsonl(self, jsonl_path: Path, subset_name: str) -> None:
        """
        Run inference on a single JSONL file, writing per-sample outputs and an aggregated jsonl.
        """
        subset_out = self.cfg.output_dir / subset_name
        subset_out.mkdir(parents=True, exist_ok=True)

        aggregated: List[Dict[str, Any]] = []

        for idx, obj in enumerate(read_jsonl(jsonl_path)):
            sample_dir = subset_out / str(idx)
            result_path = sample_dir / "result.json"
            err_path = sample_dir / "error.txt"

            if self.cfg.resume:
                if result_path.exists():
                    continue
                if (not self.cfg.retry_errors) and err_path.exists():
                    continue

            sample_dir.mkdir(parents=True, exist_ok=True)

            sample_id = str(obj.get("id", idx))
            prompt = str(obj.get("prompt", ""))
            images = obj.get("images", [])

            try:
                prompt_tokens = self.prepare_prompt_tokens(prompt, images)
                tokens = self.generate_once(prompt_tokens)

                decoded = decode_interleaved_sample(
                    tokens=tokens,
                    processor=self.processor,
                    model=self.model,
                    out_dir=sample_dir,
                    sample_id=sample_id,
                    strict=self.cfg.strict_decode,
                )

                with result_path.open("w", encoding="utf-8") as f:
                    json.dump(decoded, f, ensure_ascii=False, indent=2)

                if err_path.exists():
                    try:
                        err_path.unlink()
                    except Exception:
                        pass

                aggregated.append(decoded)

            except Exception as e:
                LOGGER.exception("Failed sample idx=%d id=%s: %s", idx, sample_id, e)
                with err_path.open("w", encoding="utf-8") as f:
                    f.write(str(e))

            # Opportunistic cleanup for long runs.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        write_jsonl(subset_out / "decoded_results.jsonl", aggregated)
        LOGGER.info("Finished subset '%s': wrote %d decoded rows.", subset_name, len(aggregated))


def truncate_by_max_images(tokens: List[int], boi_id: int, eos_id: int, max_images: int) -> List[int]:
    """
    Keep at most `max_images` images, where an image is identified by <boi>.
    This counts <boi> occurrences in the full sequence.
    """
    if max_images <= 0:
        return [eos_id]

    count = 0
    cut_at: Optional[int] = None
    for i, t in enumerate(tokens):
        if t == boi_id:
            count += 1
            if count > max_images:
                cut_at = i
                break

    if cut_at is None:
        return tokens

    return tokens[:cut_at] + [eos_id]


# ----------------------------
# Directory runner with priority queue
# ----------------------------
def collect_jsonl_files(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix == ".jsonl":
        return [input_path]
    if input_path.is_dir():
        return sorted([p for p in input_path.iterdir() if p.suffix == ".jsonl"], key=lambda x: x.name)
    raise FileNotFoundError(f"Input path not found or unsupported: {input_path}")


def order_by_priority(files: List[Path], priority_terms: List[str]) -> List[Path]:
    norm_names = {p: normalize_name(p.stem) for p in files}
    used: set[Path] = set()
    ordered: List[Path] = []

    for term in priority_terms:
        t = normalize_name(term)
        matches = [p for p in files if p not in used and t in norm_names[p]]
        matches.sort(key=lambda x: x.name)
        ordered.extend(matches)
        used.update(matches)

    rest = [p for p in files if p not in used]
    rest.sort(key=lambda x: x.name)
    ordered.extend(rest)
    return ordered


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> InferenceConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="A .jsonl file or a directory containing .jsonl files.")
    ap.add_argument("--output-dir", type=str, required=True, help="Output directory.")

    ap.add_argument("--model-path", type=str, required=True, help="Path to the Chameleon model checkpoint.")
    ap.add_argument(
        "--processor-path",
        type=str,
        default=None,
        help="Path to the processor checkpoint (defaults to model-path).",
    )
    ap.add_argument("--vq-ref-path", type=str, default=None, help="Optional: reference model to copy VQ weights from.")

    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device type.")
    ap.add_argument("--dtype", type=str, default="bfloat16", help="One of: bfloat16, float16, float32.")

    ap.add_argument("--max-length", type=int, default=12000, help="Max total sequence length.")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--do-sample", action="store_true", help="Enable sampling; default is greedy if not set.")

    ap.add_argument(
        "--generation-mode",
        type=str,
        default="unrestricted",
        help="multimodal_generation_mode for transformers generate(). e.g. 'unrestricted' or 'interleaved-text-image'.",
    )
    ap.add_argument("--max-images", type=int, default=None, help="Maximum number of images to keep.")

    ap.add_argument("--image-placeholder-id", type=int, default=None, help="Override image placeholder token id.")
    ap.add_argument("--no-strict-decode", action="store_true", help="Disable strict image segment validation during decoding.")

    ap.add_argument("--resume", action="store_true", help="Skip samples that already have result.json.")
    ap.add_argument("--retry-errors", action="store_true", help="Retry samples that have error.txt when resuming.")

    ap.add_argument(
        "--priority-terms",
        type=str,
        nargs="*",
        default=["visual_search", "visual-search", "geometry", "visual_jigsaw", "visual-jigsaw"],
        help="File name terms to prioritize (in order).",
    )

    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()
    set_logging(args.verbose)

    def _norm(p: Optional[str]) -> Optional[Path]:
        if p is None:
            return None
        return Path(os.path.expanduser(p)).resolve()

    return InferenceConfig(
        input_path=_norm(args.input) or Path(args.input),
        model_path=_norm(args.model_path) or Path(args.model_path),
        processor_path=_norm(args.processor_path),
        vq_ref_path=_norm(args.vq_ref_path),
        output_dir=_norm(args.output_dir) or Path(args.output_dir),
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=bool(args.do_sample),
        generation_mode=args.generation_mode,
        max_images=args.max_images,
        image_placeholder_id=args.image_placeholder_id,
        strict_decode=not bool(args.no_strict_decode),
        resume=bool(args.resume),
        retry_errors=bool(args.retry_errors),
        priority_terms=list(args.priority_terms),
    )


def main() -> None:
    cfg = parse_args()

    runner = ChameleonInfer(cfg)

    input_files = collect_jsonl_files(cfg.input_path)
    ordered = order_by_priority(input_files, cfg.priority_terms)

    LOGGER.info("Ordered subsets: %s", [p.name for p in ordered])

    for p in ordered:
        subset_name = p.stem
        LOGGER.info("Running subset: %s", subset_name)
        runner.run_jsonl(p, subset_name)

    LOGGER.info("All done.")


if __name__ == "__main__":
    main()

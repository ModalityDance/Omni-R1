from __future__ import annotations

import functools
import json
import logging
import os
import re
import sys
import time
import uuid
from contextlib import contextmanager
from difflib import SequenceMatcher
from logging.handlers import RotatingFileHandler
from mathruler.grader import grade_answer
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================================================
# Defaults for a fixed quantized grid (e.g., 256x256 downsample by 8 -> 32x32)
# ================================================================
GRID_H_DEFAULT: int = 32
GRID_W_DEFAULT: int = 32

LOGGER_NAME = "splited.scores"
logger = logging.getLogger(LOGGER_NAME)
logger.propagate = False

# ----------------------------------------------------------------
# Environment-driven settings
# ----------------------------------------------------------------
ENV_LOG_LEVEL = os.getenv("SPLITED_LOG_LEVEL", "").upper()
ENV_LOG_FILE = os.getenv("SPLITED_LOG_FILE", "").strip()
ENV_LOG_MAX_BYTES = int(os.getenv("SPLITED_LOG_MAX_BYTES", "10485760"))  # 10MB
ENV_LOG_BACKUPS = int(os.getenv("SPLITED_LOG_BACKUPS", "3"))

SEG_LOG_LIMIT = int(os.getenv("SPLITED_SEG_LOG_LIMIT", "5"))
PREVIEW_N = int(os.getenv("SPLITED_PREVIEW_N", "64"))
DUMP_DIR = os.getenv("SPLITED_DUMP_DIR", "").strip()
TV2D_STRICT = int(os.getenv("SPLITED_TV2D_STRICT", "0"))

THRESH_SIM_DEFAULT = 0.77
THRESH_SIM_TEXT = float(os.getenv("SPLITED_THRESH_TEXT", str(THRESH_SIM_DEFAULT)))
THRESH_SIM_CHEM_NAME = float(os.getenv("SPLITED_THRESH_CHEM_NAME", "0.90"))
THRESH_JACCARD_CHEM = float(os.getenv("SPLITED_THRESH_JACCARD_CHEM", "0.85"))


def _setup_logger() -> None:
    if logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)

    if ENV_LOG_LEVEL in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        logger.setLevel(getattr(logging, ENV_LOG_LEVEL))
    else:
        logger.setLevel(logging.INFO)

    if ENV_LOG_FILE:
        try:
            fh = RotatingFileHandler(
                ENV_LOG_FILE,
                maxBytes=ENV_LOG_MAX_BYTES,
                backupCount=ENV_LOG_BACKUPS,
            )
            fh.setFormatter(handler.formatter)
            fh.setLevel(logger.level)
            logger.addHandler(fh)
            logger.info("File logging enabled -> %s", ENV_LOG_FILE)
        except Exception as e:
            logger.warning("Failed to enable file logging: %s", e)


_setup_logger()


# ================================================================
# Basic coercions / sanitizers
# ================================================================
def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().reshape(-1)[0].item()) if x.numel() else float(default)
        arr = np.asarray(x)
        if arr.size == 0:
            return float(default)
        return float(arr.reshape(-1)[0])
    except Exception:
        try:
            return float(x)
        except Exception:
            return float(default)


def _as_int(x: Any, default: int = 0) -> int:
    try:
        if isinstance(x, torch.Tensor):
            return int(x.detach().cpu().reshape(-1)[0].item()) if x.numel() else int(default)
        arr = np.asarray(x)
        if arr.size == 0:
            return int(default)
        return int(arr.reshape(-1)[0])
    except Exception:
        try:
            return int(x)
        except Exception:
            return int(default)


def _as_str(x: Any, default: str = "") -> str:
    try:
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return str(x.detach().cpu().item())
            return str(x.detach().cpu().tolist())
        arr = np.asarray(x)
        if arr.size == 0:
            return default
        if arr.size == 1:
            return str(arr.reshape(()).item())
        return str(arr.tolist())
    except Exception:
        try:
            return str(x)
        except Exception:
            return default


def _as_dict(x: Any, name: str = "reward_kwargs") -> Dict[str, Any]:
    if isinstance(x, dict):
        return x

    if isinstance(x, np.ndarray):
        try:
            if x.dtype == object and x.size == 1:
                item = x.item()
                if isinstance(item, dict):
                    return item
        except Exception:
            pass
        logger.warning("[sanitize] %s is ndarray(dtype=%s, size=%s) -> {}", name, getattr(x, "dtype", None), getattr(x, "size", None))
        return {}

    if isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], dict):
        return x[0]

    logger.warning("[sanitize] %s type=%s -> {}", name, type(x).__name__)
    return {}


# ================================================================
# Logging helpers
# ================================================================
def _short_shape(x: Any) -> Tuple[int, ...]:
    try:
        return tuple(x.shape)
    except Exception:
        try:
            return (len(x),)
        except Exception:
            return ()


def _dtype_of(x: Any) -> str:
    if isinstance(x, torch.Tensor):
        return str(x.dtype)
    if isinstance(x, np.ndarray):
        return str(x.dtype)
    return type(x).__name__


def _preview1d(x: Any, n: int = PREVIEW_N) -> str:
    try:
        if isinstance(x, torch.Tensor):
            arr = x.detach().cpu().view(-1).tolist()
        elif isinstance(x, np.ndarray):
            arr = x.reshape(-1).tolist()
        elif isinstance(x, (list, tuple)):
            arr = list(x)
        else:
            s = str(x)
            return s[:120] + ("..." if len(s) > 120 else "")
        return repr(arr[:n]) + ("..." if len(arr) > n else "")
    except Exception:
        return "<unpreviewable>"


def _maybe_dump(trace_id: str, name: str, arr: Any) -> None:
    if not DUMP_DIR:
        return
    try:
        os.makedirs(DUMP_DIR, exist_ok=True)
        path = os.path.join(DUMP_DIR, f"{name}.{trace_id}.npy")
        if isinstance(arr, torch.Tensor):
            np.save(path, arr.detach().cpu().numpy())
        else:
            np.save(path, np.asarray(arr))
        logger.info("[dump] %s -> %s", name, path)
    except Exception as e:
        logger.warning("[dump] %s failed: %s", name, e)


@contextmanager
def _log_scope(scope: str, **fields: Any):
    trace_id = fields.pop("trace_id", None) or uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    try:
        if logger.isEnabledFor(logging.INFO):
            logger.info("[%s] BEGIN trace=%s %s", scope, trace_id, json.dumps(fields, ensure_ascii=True))
        yield trace_id
    except Exception:
        logger.exception("[%s] EXCEPTION trace=%s", scope, trace_id)
        raise
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if logger.isEnabledFor(logging.INFO):
            logger.info("[%s] END   trace=%s elapsed_ms=%.2f", scope, trace_id, dt_ms)


# ================================================================
# Misc helpers
# ================================================================
def truncate(text: Optional[str], max_len: int = 3000) -> str:
    if not text:
        return ""
    return text if len(text) <= max_len else text[:max_len] + "..."


def _first_present(d: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _is_empty_seq(x: Any) -> bool:
    if x is None:
        return True
    try:
        return len(x) == 0
    except Exception:
        pass
    if hasattr(x, "numel"):
        try:
            return int(x.numel()) == 0
        except Exception:
            pass
    if hasattr(x, "size"):
        try:
            return int(x.size) == 0
        except Exception:
            pass
    return False


def _coerce_ids_tensor(x: Any, name: str = "responses") -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.to(torch.long)
    else:
        arr = x
        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
            arr = arr.item()
        try:
            arr = np.asarray(arr, dtype=np.int64)
            if arr.dtype == object:
                try:
                    arr = np.concatenate([np.asarray(a, dtype=np.int64).reshape(-1) for a in arr], axis=0)
                except Exception as e:
                    raise TypeError(
                        f"{name}: failed to concatenate object arrays into int64 vector (shape mismatch). Original: {e}"
                    )
        except Exception as e:
            raise TypeError(f"{name}: cannot convert to int64 array: {e}")
        t = torch.from_numpy(arr)

    if t.dim() == 2 and t.size(0) == 1:
        t = t[0]
    if t.dim() != 1:
        t = t.reshape(-1)
    t = t.to(torch.long).contiguous()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[coerce_ids] %s -> shape=%s dtype=%s preview=%s", name, _short_shape(t), _dtype_of(t), _preview1d(t))
    return t


def _coerce_hids_tensor(x: Any, name: str = "hidden_states") -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.to(torch.float32)
    else:
        arr = x
        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
            arr = arr.item()
        try:
            arr = np.asarray(arr)
            if arr.dtype == object:
                try:
                    arr = np.stack([np.asarray(a, dtype=np.float32) for a in arr], axis=0)  # (T, H)
                except Exception as e:
                    raise TypeError(f"{name}: failed to stack object arrays into (T,H). Original: {e}")
            else:
                arr = arr.astype(np.float32, copy=False)
        except Exception as e:
            raise TypeError(f"{name}: cannot convert to float32 matrix: {e}")
        t = torch.from_numpy(arr).to(torch.float32)

    if t.dim() == 3 and t.size(0) == 1:
        t = t[0]
    if t.dim() != 2:
        raise ValueError(f"{name}: expected shape (T,H) or (1,T,H), got {tuple(t.shape)}")

    t = t.contiguous()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[coerce_hids] %s -> shape=%s dtype=%s preview=%s", name, _short_shape(t), _dtype_of(t), _preview1d(t.view(-1)))
    return t


# ================================================================
# Answer formatting / parsing
# ================================================================
def extract_after_final_answer(text: str, key: str = "Final Answer:") -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = re.search(re.escape(key) + r"\s*(.*)", text, flags=re.DOTALL)
    return m.group(1).strip() if m else None


def format_reward(predict_str: str) -> float:
    if not isinstance(predict_str, str):
        return 0.0
    s = predict_str.strip()
    ok = ("THOUGHT" in s) and ("Final Answer:" in s)
    return 1.0 if ok else 0.0


_math_sign_re = re.compile(r"[=+*/^]|\\[a-zA-Z]+")
_digits_re = re.compile(r"^\d+$")


def _normalize_punct(s: str) -> str:
    t = (s or "").strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _strip_answer_prefix_and_trailing_punct(s: str) -> str:
    t = _normalize_punct(s)
    t = re.sub(r"^\s*(?:the\s+answer\s+is|answer\s+is)\s*[:\-]?\s*", "", t, flags=re.I)
    t = re.sub(r"[\s\.\?!;:]+$", "", t)
    return t.strip()


_ARTICLE_LEADING_RE = re.compile(r"^(?:\s*)(?:the|a|an)\s+", flags=re.I)


def _strip_leading_articles(s: str) -> str:
    return _ARTICLE_LEADING_RE.sub("", (s or "").strip())


_INT_TO_LETTER = {i: chr(ord("A") + i - 1) for i in range(1, 11)}
_ROMAN_MAP_ASCII = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
_ROMAN_FULLWIDTH_TO_ASCII = str.maketrans(
    {"Ⅰ": "I", "Ⅱ": "II", "Ⅲ": "III", "Ⅳ": "IV", "Ⅴ": "V", "Ⅵ": "VI", "Ⅶ": "VII", "Ⅷ": "VIII", "Ⅸ": "IX", "Ⅹ": "X"}
)


def _roman_to_int(tok: str) -> Optional[int]:
    t = (tok or "").translate(_ROMAN_FULLWIDTH_TO_ASCII).upper()
    return _ROMAN_MAP_ASCII.get(t)


def _normalize_label_token(tok: str) -> Optional[str]:
    if not isinstance(tok, str):
        return None
    t = _normalize_punct(tok).strip()

    if re.fullmatch(r"[A-Ja-j]", t):
        return t.upper()

    m = re.fullmatch(r"(?:\s*)(\d{1,2})\s*[\)\]\.:]?", t)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 10:
            return _INT_TO_LETTER[n]

    r = _roman_to_int(t)
    if r and 1 <= r <= 10:
        return _INT_TO_LETTER[r]

    return None


_LABEL_TOKEN = r"(?:[A-Ja-j]|10|[1-9]|i{1,3}|iv|v|vi{1,3}|ix|x|Ⅰ|Ⅱ|Ⅲ|Ⅳ|Ⅴ|Ⅵ|Ⅶ|Ⅷ|Ⅸ|Ⅹ)"
_LB = r"(?<![0-9A-Za-z])"
_RB = r"(?![0-9A-Za-z])"

ANCHOR_LABEL_RE = re.compile(
    rf"^\s*(?:option|answer)?\s*[:\-\)]?\s*[\(\[]?\s*"
    rf"({_LABEL_TOKEN})(?=$|\s|[\)\]\.:])"
    rf"\s*[\)\]\.:]?\s*(.*)$",
    flags=re.I,
)


def parse_choice_like(s: str) -> Tuple[Optional[str], str]:
    """
    Minimal, safer choice parsing:
      - Only accepts labels anchored at the beginning of the string.
      - Rejects labels if immediately followed by '-' or '(' (common in locant-like patterns).
      - Avoids treating 'a <word>' as a label.
    """
    if not isinstance(s, str):
        return (None, "")

    t = _strip_answer_prefix_and_trailing_punct(s)
    m = ANCHOR_LABEL_RE.match(t)
    if not m:
        return (None, t)

    raw_lab = m.group(1)
    j = m.end(1)

    while j < len(t) and t[j].isspace():
        j += 1
    if j < len(t) and t[j] in "-(":
        return (None, _strip_answer_prefix_and_trailing_punct(t))

    if len(raw_lab) == 1 and raw_lab.islower():
        k = j
        while k < len(t) and t[k].isspace():
            k += 1
        if k < len(t) and t[k].isalpha():
            return (None, _strip_answer_prefix_and_trailing_punct(t))

    lab = _normalize_label_token(raw_lab)
    if not lab:
        return (None, t)

    rest = (m.group(2) or "").strip()
    rest = re.sub(r"^[\-\.:)\]\s]+", "", rest)
    rest = _strip_answer_prefix_and_trailing_punct(rest)
    return (lab, rest)


def _text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


# ================================================================
# Chemistry heuristics (optional)
# ================================================================
_CHEM_FORMULA_RE = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")


def _parse_formula_counts(s: str) -> Optional[Dict[str, int]]:
    t = (s or "").replace(" ", "")
    if not _CHEM_FORMULA_RE.fullmatch(t):
        return None
    counts: Dict[str, int] = {}
    for elem, num in re.findall(r"([A-Z][a-z]?)(\d*)", t):
        counts[elem] = counts.get(elem, 0) + (int(num) if num else 1)
    return counts


def _formula_equivalent(a: str, b: str) -> Optional[bool]:
    ca = _parse_formula_counts(a)
    cb = _parse_formula_counts(b)
    if ca is None or cb is None:
        return None
    return ca == cb


_CHEM_MORPHEMES = (
    "benz",
    "phen",
    "naphth",
    "piperazin",
    "pyrid",
    "imidazol",
    "thiazol",
    "quinazolin",
    "methyl",
    "ethyl",
    "propyl",
    "butyl",
    "benzyl",
    "hydroxy",
    "amino",
    "oxo",
    "carbox",
    "acetyl",
    "methoxy",
    "chloro",
    "fluoro",
    "bromo",
    "iodo",
    "sulfon",
    "sulfone",
    "yl",
    "ylidene",
)


def _looks_chem_name(s: str) -> bool:
    if not isinstance(s, str):
        return False
    t = s.lower()
    has_locants = bool(re.search(r"\d+(?:,\d+)*\s*[-)]", s))
    has_paren_or_dash = any(ch in s for ch in "-()[]")
    has_morpheme = any(m in t for m in _CHEM_MORPHEMES)
    return has_morpheme and (has_locants or has_paren_or_dash)


def _normalize_chem_name_tokens(s: str) -> List[str]:
    t = re.sub(r"[\d,]+", " ", s or "")
    t = t.replace("-", " ").replace("–", " ")
    t = re.sub(r"[\(\)\[\]\{\}]", " ", t)
    t = t.lower()
    return re.findall(r"[a-z]+", t)


def _jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    A, B = set(a_tokens), set(b_tokens)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _chem_name_similarity(a: str, b: str) -> Tuple[float, float, float]:
    ta = _normalize_chem_name_tokens(a)
    tb = _normalize_chem_name_tokens(b)
    jacc = _jaccard(ta, tb)
    seq_sim = _text_similarity(" ".join(ta), " ".join(tb))
    fused = 0.5 * jacc + 0.5 * seq_sim
    return jacc, seq_sim, fused


# ================================================================
# Numeric-in-text handling to prevent false positives (e.g., 5 vs 9)
# ================================================================
_NUM_IN_TEXT_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])")


def _numbers_in_text(s: str) -> List[float]:
    if not isinstance(s, str):
        return []
    return [float(x) for x in _NUM_IN_TEXT_RE.findall(s or "")]


def _drop_numbers(s: str) -> str:
    return _NUM_IN_TEXT_RE.sub(" ", s or "")


def compare_text_like(ans_raw: str, gt_raw: str) -> float:
    def _prep_cmp(a_raw: str, g_raw: str) -> Tuple[str, str]:
        a0 = _strip_answer_prefix_and_trailing_punct(a_raw)
        g0 = _strip_answer_prefix_and_trailing_punct(g_raw)
        a_has_art = (_strip_leading_articles(a0) != a0)
        g_has_art = (_strip_leading_articles(g0) != g0)
        if a_has_art and g_has_art:
            return _strip_leading_articles(a0), _strip_leading_articles(g0)
        return a0, g0

    a_lab, a_txt = parse_choice_like(ans_raw)
    g_lab, g_txt = parse_choice_like(gt_raw)

    if g_lab:
        if not g_txt:
            if a_lab:
                result = 1.0 if a_lab == g_lab else 0.0
                logger.info("choice-only compare: %s (ans=%s gt=%s)", bool(result), a_lab, g_lab)
                return result
            logger.info("choice-only compare: False (ans has no label, gt=%s)", g_lab)
            return 0.0

        if a_lab and a_lab == g_lab:
            logger.info("choice compare: True (ans=%s gt=%s)", a_lab, g_lab)
            return 1.0

        a_cmp, g_cmp = _prep_cmp(a_txt or ans_raw, g_txt or gt_raw)
    else:
        a_cmp, g_cmp = _prep_cmp(a_txt or ans_raw, g_txt or gt_raw)

    eq_formula = _formula_equivalent(a_cmp.replace(" ", ""), g_cmp.replace(" ", ""))
    if eq_formula is not None:
        logger.info("formula compare: %s (ans='%s' gt='%s')", eq_formula, truncate(a_cmp, 80), truncate(g_cmp, 80))
        return 1.0 if eq_formula else 0.0

    if _looks_chem_name(a_cmp) and _looks_chem_name(g_cmp):
        jacc, seq_sim, fused = _chem_name_similarity(a_cmp, g_cmp)
        ok = (jacc >= THRESH_JACCARD_CHEM) and (seq_sim >= THRESH_SIM_CHEM_NAME)
        logger.info(
            "chem-name compare: jacc=%.3f seq=%.3f fused=%.3f thr=(%.2f, %.2f) -> %s (ans='%s' gt='%s')",
            jacc,
            seq_sim,
            fused,
            THRESH_JACCARD_CHEM,
            THRESH_SIM_CHEM_NAME,
            ok,
            truncate(a_cmp, 80),
            truncate(g_cmp, 80),
        )
        return 1.0 if ok else 0.0

    nums_a, nums_g = _numbers_in_text(a_cmp), _numbers_in_text(g_cmp)
    if nums_a and nums_g and not (_looks_chem_name(a_cmp) or _looks_chem_name(g_cmp)):
        if len(nums_a) == len(nums_g) and all(abs(x - y) < 1e-9 for x, y in zip(nums_a, nums_g)):
            logger.info("numbers match -> True nums=%s", nums_a)
            return 1.0
        ctx_sim_wo_nums = _text_similarity(_drop_numbers(a_cmp), _drop_numbers(g_cmp))
        if ctx_sim_wo_nums >= max(THRESH_SIM_TEXT, 0.90):
            logger.info("context matches but numbers differ -> False (sim_wo_nums=%.3f A=%s G=%s)", ctx_sim_wo_nums, nums_a, nums_g)
            return 0.0

    sim = _text_similarity(a_cmp, g_cmp)
    acc_val = sim if sim > THRESH_SIM_TEXT else 0.0
    logger.info(
        "text similarity: %.3f (thr=%.2f) acc=%.3f (ans='%s' gt='%s')",
        sim,
        THRESH_SIM_TEXT,
        acc_val,
        truncate(a_cmp, 80),
        truncate(g_cmp, 80),
    )
    return acc_val


# ================================================================
# Numeric / currency / percent parsing
# ================================================================
_CURRENCY_CHARS = r"$€£¥₹₩₽₫₴₪₦₱฿₭₮₸₺₼₿"


def _parse_numeric_like(s: str) -> Tuple[Optional[float], bool, bool]:
    if not isinstance(s, str):
        return (None, False, False)
    t = s.strip()
    is_paren_neg = bool(re.match(r"^\(\s*.*\s*\)$", t))
    if is_paren_neg:
        t = t[1:-1].strip()

    has_pct = t.endswith("%")
    if has_pct:
        t = t[:-1].strip()

    t = re.sub(rf"[{_CURRENCY_CHARS},\s]", "", t)
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", t or ""):
        return (None, has_pct, False)

    val = float(t)
    if is_paren_neg:
        val = -val
    return (val, has_pct, True)


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    answer = extract_after_final_answer(predict_str, key="Final Answer:") or ""
    if not answer:
        logger.warning("Missing 'Final Answer:' section. Snippet: %s", truncate(predict_str, 200))

    ans_norm = answer.strip()
    gt_norm = (ground_truth or "").strip()

    if _digits_re.fullmatch(ans_norm) and _digits_re.fullmatch(gt_norm):
        is_correct = ans_norm == gt_norm
        logger.info("digits compare: %s (ans=%s gt=%s)", is_correct, ans_norm, gt_norm)
        return 1.0 if is_correct else 0.0

    a_val, a_pct, a_ok = _parse_numeric_like(ans_norm)
    g_val, g_pct, g_ok = _parse_numeric_like(gt_norm)
    if a_ok and g_ok:
        eps = 1e-9
        if a_pct == g_pct:
            is_eq = abs(a_val - g_val) < eps
            logger.info("numeric compare: %s (ans=%s gt=%s)", is_eq, ans_norm, gt_norm)
            return 1.0 if is_eq else 0.0
        if abs(a_val - g_val) < eps:
            logger.info("numeric compare (direct): True (ans=%s gt=%s)", ans_norm, gt_norm)
            return 1.0
        if a_pct and abs(a_val - 100 * g_val) < eps:
            logger.info("numeric compare (pct<->frac): True (ans=%s gt=%s)", ans_norm, gt_norm)
            return 1.0
        if g_pct and abs(g_val - 100 * a_val) < eps:
            logger.info("numeric compare (pct<->frac): True (ans=%s gt=%s)", ans_norm, gt_norm)
            return 1.0
        logger.info("numeric compare: False (ans=%s gt=%s)", ans_norm, gt_norm)
        return 0.0

    if not _math_sign_re.search(ans_norm) and not _math_sign_re.search(gt_norm):
        acc_val = compare_text_like(ans_norm, gt_norm)
        logger.info("text-like compare: acc=%.3f (thr=%.2f) (ans='%s' gt='%s')", acc_val, THRESH_SIM_TEXT, truncate(ans_norm, 80), truncate(gt_norm, 80))
        return acc_val

    result = grade_answer(ans_norm, gt_norm)
    logger.info("math compare: %s (ans='%s' gt='%s')", result, truncate(ans_norm, 120), truncate(gt_norm, 120))
    return 1.0 if result else 0.0


# ================================================================
# Perceptual scoring (token-only or hidden-states based)
# ================================================================
def _norm_path(p: str) -> str:
    return os.path.realpath(os.path.abspath(p))


@functools.lru_cache(maxsize=8)
def _load_embed_weight_cached(embed_path_norm: str) -> torch.Tensor:
    t = torch.load(embed_path_norm, map_location="cpu")
    if isinstance(t, dict) and "weight" in t and isinstance(t["weight"], torch.Tensor):
        t = t["weight"]
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"embed checkpoint at {embed_path_norm} is not a Tensor")
    logger.info("[embed] loaded %s shape=%s dtype=%s", embed_path_norm, tuple(t.shape), t.dtype)
    return t


def _ensure_1d(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dim() == 1:
        return x
    if x.dim() == 2 and x.shape[0] == 1:
        return x[0]
    raise ValueError(f"{name} must be (T,) or (1,T); got {tuple(x.shape)}")


def _map_to_code_ids(image_token_ids: torch.Tensor, offset: int, K: int) -> torch.Tensor:
    ok = (image_token_ids >= offset) & (image_token_ids < offset + K)
    if not bool(ok.all()):
        bad_pos = (~ok).nonzero(as_tuple=False).flatten().tolist()[:8]
        raise AssertionError(
            f"Out-of-range image tokens (example positions: {bad_pos}). "
            f"Expected in [{offset}, {offset + K}), got min={int(image_token_ids.min())} max={int(image_token_ids.max())}."
        )
    return image_token_ids - offset


class PerceptualLoss(nn.Module):
    def __init__(self, embed_path: str, vocab_size: int = 8192, hidden_size: int = 256, generator_hidden_size: int = 4096):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)

        emb_w = _load_embed_weight_cached(_norm_path(embed_path))
        self.embedding = nn.Embedding.from_pretrained(emb_w, freeze=True)
        self.states_to_hidden = nn.Linear(int(generator_hidden_size), int(hidden_size))
        self.eval()

        self._cur_device = torch.device("cpu")
        self._cur_dtype = self.embedding.weight.dtype

    def move_to(self, device: torch.device, dtype: torch.dtype) -> "PerceptualLoss":
        if (device != self._cur_device) or (dtype != self._cur_dtype):
            self.to(device=device, dtype=dtype)
            self._cur_device = device
            self._cur_dtype = dtype
        return self

    @staticmethod
    def mse_to_score(val: float, tau: float = 1.0, kind: str = "inv") -> float:
        v = float(val)
        tau = max(float(tau), 1e-8)
        k = (kind or "inv").lower()
        if k == "inv":
            return 1.0 / (1.0 + v / tau)
        if k == "exp":
            return float(torch.exp(torch.tensor(-v / tau)).item())
        return 1.0 / (1.0 + v / tau)

    @staticmethod
    def _segments_from_boi_eoi(input_ids: torch.Tensor, boi_id: int, eoi_id: int) -> List[Tuple[int, int]]:
        input_ids = _ensure_1d(input_ids, "input_ids")
        ids = input_ids.tolist()
        segs: List[Tuple[int, int]] = []
        open_s: Optional[int] = None
        for i, tok in enumerate(ids):
            if tok == boi_id:
                if open_s is not None:
                    segs.append((open_s, i))
                open_s = i + 1
            elif tok == eoi_id and open_s is not None:
                segs.append((open_s, i))
                open_s = None
        return [(s, e) for (s, e) in segs if e > s]

    @torch.inference_mode()
    def mse_on_image_tokens_in_segments(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        boi_id: int,
        eoi_id: int,
        min_img_tokens: int = 1,
    ) -> List[float]:
        target_device = hidden_states.device
        target_dtype = hidden_states.dtype
        self.move_to(target_device, target_dtype)

        input_ids = _ensure_1d(input_ids, "input_ids")

        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states[0]
        if hidden_states.dim() != 2:
            raise ValueError(f"hidden_states must be (T,H) or (1,T,H), got {tuple(hidden_states.shape)}")

        in_feat = self.states_to_hidden.in_features
        if hidden_states.size(1) != in_feat:
            raise ValueError(f"hidden_states H={hidden_states.size(1)} does not match expected {in_feat}")

        T = min(input_ids.shape[0], hidden_states.shape[0])
        ids = input_ids[:T].to(device=target_device, dtype=torch.long)
        hids = hidden_states[:T].to(device=target_device, dtype=target_dtype)

        segs = self._segments_from_boi_eoi(ids, boi_id, eoi_id)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[pl.mse] seg_count=%d segs_preview=%s", len(segs), segs[: min(len(segs), SEG_LOG_LIMIT)])

        out: List[float] = []
        for seg_idx, (s, e) in enumerate(segs):
            seg_ids = ids[s:e]
            mask = (seg_ids >= 4) & (seg_ids < 4 + self.vocab_size)
            n_img = int(mask.sum().item())
            if n_img < min_img_tokens:
                if logger.isEnabledFor(logging.DEBUG) and seg_idx < SEG_LOG_LIMIT:
                    logger.debug("[pl.mse] skip seg=%d len=%d img_tokens=%d (<%d)", seg_idx, int(seg_ids.numel()), n_img, min_img_tokens)
                continue

            code_ids = _map_to_code_ids(seg_ids[mask], offset=4, K=self.vocab_size)
            labels = self.embedding(code_ids)
            feats = self.states_to_hidden(hids[s:e][mask])
            mse = torch.mean((feats - labels) ** 2)
            out.append(float(mse.item()))

            if logger.isEnabledFor(logging.DEBUG) and seg_idx < SEG_LOG_LIMIT:
                logger.debug(
                    "[pl.mse] seg=%d len=%d img_tokens=%d mse=%.6f ids_preview=%s",
                    seg_idx,
                    int(seg_ids.numel()),
                    n_img,
                    out[-1],
                    _preview1d(seg_ids),
                )

        if out:
            stats = dict(n=len(out), mean=float(np.mean(out)), std=float(np.std(out)), min=float(np.min(out)), max=float(np.max(out)))
            logger.info("[pl.mse] stats %s", json.dumps(stats))
        return out

    @torch.inference_mode()
    def token_only_perceptual_energy_in_segments(
        self,
        input_ids: torch.Tensor,
        boi_id: int,
        eoi_id: int,
        w1: float = 1.0,
        w2: float = 0.25,
        wcos: float = 0.5,
        min_len: int = 3,
    ) -> List[float]:
        input_ids = _ensure_1d(input_ids, "input_ids")
        self.move_to(input_ids.device, self.embedding.weight.dtype)

        ids = input_ids.to(device=self.embedding.weight.device, dtype=torch.long)
        segs = self._segments_from_boi_eoi(ids, boi_id, eoi_id)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[pl.1d] seg_count=%d segs_preview=%s", len(segs), segs[: min(len(segs), SEG_LOG_LIMIT)])

        D = int(self.embedding.weight.shape[-1])
        out: List[float] = []
        for seg_idx, (s, e) in enumerate(segs):
            seg_ids = ids[s:e]
            mask = (seg_ids >= 4) & (seg_ids < 4 + self.vocab_size)
            idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if idx.numel() < max(2, min_len):
                if logger.isEnabledFor(logging.DEBUG) and seg_idx < SEG_LOG_LIMIT:
                    logger.debug("[pl.1d] skip seg=%d len=%d img_tokens=%d (<%d)", seg_idx, int(seg_ids.numel()), int(idx.numel()), max(2, min_len))
                continue

            code_ids = _map_to_code_ids(seg_ids[idx], offset=4, K=self.vocab_size)
            evec = self.embedding(code_ids)  # (N,D)

            terms: List[torch.Tensor] = []
            if evec.size(0) >= 2:
                d1 = evec[1:] - evec[:-1]
                s1 = (d1.pow(2).sum(dim=-1) / D).mean()
                terms.append(w1 * s1)

                en = F.normalize(evec, dim=-1)
                cos = (en[1:] * en[:-1]).sum(dim=-1).clamp(-1, 1)
                sc = (1.0 - cos).mean()
                terms.append(wcos * sc)

            if evec.size(0) >= 3:
                d2 = evec[2:] - 2 * evec[1:-1] + evec[:-2]
                s2 = (d2.pow(2).sum(dim=-1) / D).mean()
                terms.append(w2 * s2)

            if terms:
                energy = torch.stack(terms).sum()
                out.append(float(energy.item()))
                if logger.isEnabledFor(logging.DEBUG) and seg_idx < SEG_LOG_LIMIT:
                    logger.debug("[pl.1d] seg=%d len=%d img_tokens=%d energy=%.6f ids_preview=%s", seg_idx, int(seg_ids.numel()), int(idx.numel()), out[-1], _preview1d(seg_ids))

        if out:
            stats = dict(n=len(out), mean=float(np.mean(out)), std=float(np.std(out)), min=float(np.min(out)), max=float(np.max(out)))
            logger.info("[pl.1d] stats %s", json.dumps(stats))
        return out

    @torch.inference_mode()
    def token_only_tv2d_energy_in_segments(
        self,
        input_ids: torch.Tensor,
        boi_id: int,
        eoi_id: int,
        grid_h: int,
        grid_w: int,
        w_tv: float = 1.0,
    ) -> List[float]:
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError("grid_h and grid_w must be positive")

        input_ids = _ensure_1d(input_ids, "input_ids")
        self.move_to(input_ids.device, self.embedding.weight.dtype)

        ids = input_ids.to(device=self.embedding.weight.device, dtype=torch.long)
        segs = self._segments_from_boi_eoi(ids, boi_id, eoi_id)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[pl.tv2d] seg_count=%d segs_preview=%s", len(segs), segs[: min(len(segs), SEG_LOG_LIMIT)])

        D = int(self.embedding.weight.shape[-1])
        out: List[float] = []
        mismatched = 0
        need = int(grid_h) * int(grid_w)

        for seg_idx, (s, e) in enumerate(segs):
            seg_ids = ids[s:e]
            mask = (seg_ids >= 4) & (seg_ids < 4 + self.vocab_size)
            img_ids = seg_ids[mask]

            if img_ids.numel() != need:
                mismatched += 1
                msg = f"[pl.tv2d] seg={seg_idx} seg_len={int(seg_ids.numel())} img_tokens={int(img_ids.numel())} need={need} -> skip"
                if TV2D_STRICT:
                    raise ValueError(msg)
                if logger.isEnabledFor(logging.DEBUG) and seg_idx < SEG_LOG_LIMIT:
                    logger.debug(msg)
                continue

            code_ids = _map_to_code_ids(img_ids, offset=4, K=self.vocab_size)
            Z = self.embedding(code_ids).view(grid_h, grid_w, D)

            tv = 0.0
            if grid_w > 1:
                tv_x = (Z[:, 1:, :] - Z[:, :-1, :]).pow(2).sum(-1).mean()
                tv += tv_x
            if grid_h > 1:
                tv_y = (Z[1:, :, :] - Z[:-1, :, :]).pow(2).sum(-1).mean()
                tv += tv_y

            tv = tv / D
            out.append(float((w_tv * tv).item()))

            if logger.isEnabledFor(logging.DEBUG) and seg_idx < SEG_LOG_LIMIT:
                logger.debug("[pl.tv2d] seg=%d energy=%.6f ids_preview=%s", seg_idx, out[-1], _preview1d(seg_ids))

        if out:
            stats = dict(
                n=len(out),
                mean=float(np.mean(out)),
                std=float(np.std(out)),
                min=float(np.min(out)),
                max=float(np.max(out)),
                mismatched=mismatched,
            )
            logger.info("[pl.tv2d] stats %s", json.dumps(stats))
        elif mismatched:
            logger.info("[pl.tv2d] no segment scored (all mismatched) need=%d mismatched=%d", need, mismatched)

        return out


_PL_CACHE: Dict[Tuple[str, int, int, int], PerceptualLoss] = {}


def _ensure_pl(rw_kwargs: Dict[str, Any]) -> PerceptualLoss:
    env_path = os.environ.get("PERCEPTUAL_EMBED_PATH")
    val = rw_kwargs.get("perceptual_embed_path", None)
    need_env = (val is None) or (isinstance(val, str) and val.strip() == "")
    if env_path and need_env:
        rw_kwargs = dict(rw_kwargs)
        rw_kwargs["perceptual_embed_path"] = env_path

    embed_path = _norm_path(rw_kwargs.get("perceptual_embed_path", "embed.ckpt"))
    cfg_key = (
        embed_path,
        int(rw_kwargs.get("perceptual_vocab_size", 8192)),
        int(rw_kwargs.get("perceptual_hidden_size", 256)),
        int(rw_kwargs.get("perceptual_generator_hidden_size", 4096)),
    )

    pl = _PL_CACHE.get(cfg_key)
    if pl is None:
        pl = PerceptualLoss(
            embed_path=embed_path,
            vocab_size=cfg_key[1],
            hidden_size=cfg_key[2],
            generator_hidden_size=cfg_key[3],
        )
        _PL_CACHE[cfg_key] = pl
        logger.info("[pl] created embed=%s cfg=%s", embed_path, cfg_key[1:])
    return pl


@torch.inference_mode()
def _perceptual_score_from_extra(extra_info: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Any]]:
    if not isinstance(extra_info, dict):
        return None, {}

    with _log_scope("perceptual", extra_keys=list(extra_info.keys())) as tid:
        ids = _first_present(extra_info, ("responses", "response_ids", "output_ids"))
        hids = _first_present(extra_info, ("generated_hidden_states", "hidden_states"))
        rw_kwargs = _as_dict(extra_info.get("reward_kwargs", {}), name="reward_kwargs")

        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "[perceptual] ids_type=%s ids_shape=%s hids_type=%s hids_shape=%s",
                type(ids).__name__,
                _short_shape(ids),
                type(hids).__name__,
                _short_shape(hids),
            )

        if ids is None:
            return None, rw_kwargs

        ids_t = _coerce_ids_tensor(ids, name="responses/ids")
        _maybe_dump(tid, "ids_first", ids_t[: min(len(ids_t), 2048)])

        pl = _ensure_pl(rw_kwargs)

        boi = _as_int(rw_kwargs.get("boi_token_id", 8197), 8197)
        eoi = _as_int(rw_kwargs.get("eoi_token_id", 8196), 8196)
        agg = _as_str(rw_kwargs.get("pl_seg_agg", "min"), "min").lower()
        tau = _as_float(rw_kwargs.get("pl_tau", 3.0), 3.0)
        mapk = _as_str(rw_kwargs.get("pl_map", "inv"), "inv").lower()

        grid_h = _as_int(rw_kwargs.get("pl_grid_h", GRID_H_DEFAULT), GRID_H_DEFAULT)
        grid_w = _as_int(rw_kwargs.get("pl_grid_w", GRID_W_DEFAULT), GRID_W_DEFAULT)

        logger.info("[perceptual] cfg boi=%d eoi=%d agg=%s map=%s tau=%.4f grid=(%d,%d)", boi, eoi, agg, mapk, tau, grid_h, grid_w)

        if hids is not None:
            hids_t = _coerce_hids_tensor(hids, name="generated_hidden_states/hidden_states")
            mses = pl.mse_on_image_tokens_in_segments(ids_t, hids_t, boi, eoi)
            if _is_empty_seq(mses):
                logger.info("[perceptual] no MSE segments scored")
                return None, rw_kwargs
            seg_scores = [PerceptualLoss.mse_to_score(m, tau=tau, kind=mapk) for m in mses]
        else:
            energies: List[float] = []
            if grid_h > 0 and grid_w > 0:
                energies = pl.token_only_tv2d_energy_in_segments(
                    ids_t,
                    boi,
                    eoi,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    w_tv=float(rw_kwargs.get("pl_tok_wtv", 1.0)),
                )

            if _is_empty_seq(energies):
                energies = pl.token_only_perceptual_energy_in_segments(
                    ids_t,
                    boi,
                    eoi,
                    w1=float(rw_kwargs.get("pl_tok_w1", 1.0)),
                    w2=float(rw_kwargs.get("pl_tok_w2", 0.25)),
                    wcos=float(rw_kwargs.get("pl_tok_wcos", 0.5)),
                )

            if _is_empty_seq(energies):
                logger.info("[perceptual] no token-only segments scored")
                return None, rw_kwargs

            _maybe_dump(tid, "energies_first", np.array(energies[:256], dtype=np.float32))
            seg_scores = [PerceptualLoss.mse_to_score(e, tau=tau, kind=mapk) for e in energies]

        if agg == "max":
            score = float(max(seg_scores))
        elif agg == "min":
            score = float(min(seg_scores))
        elif agg == "median":
            score = float(np.median(seg_scores))
        else:
            score = float(sum(seg_scores) / len(seg_scores))

        logger.info("[perceptual] final score=%.6f (agg=%s)", score, agg)
        return score, rw_kwargs


# ================================================================
# Main API
# ================================================================
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,
) -> Dict[str, Any]:
    with _log_scope("compute_score", data_source=data_source) as _tid:
        has_resp = isinstance(extra_info, dict) and any(
            (k in extra_info) and (extra_info[k] is not None)
            for k in ("responses", "response_ids", "output_ids")
        )
        has_hids = isinstance(extra_info, dict) and any(
            (k in extra_info) and (extra_info[k] is not None)
            for k in ("generated_hidden_states", "hidden_states")
        )
        logger.info("[inputs] has_responses=%s has_hiddens=%s", has_resp, has_hids)

        fmt = format_reward(solution_str)
        acc = acc_reward(solution_str, ground_truth, use_boxed=True)

        img_score: Optional[float] = None
        rw_kwargs: Dict[str, Any] = {}

        try:
            img_score, rw_kwargs = _perceptual_score_from_extra(extra_info if isinstance(extra_info, dict) else {})
        except Exception as e:
            logger.warning("[compute_score] perceptual scoring failed: %s", e, exc_info=True)

        w_img = _as_float(
            rw_kwargs.get("image_reward_weight", 0.05 if img_score is not None else 0.0),
            0.05 if img_score is not None else 0.0,
        )

        w_fmt_default = 0.1
        w_acc_default = 0.9

        logger.info("[weights] w_img=%.3f (default acc/fmt=0.9/0.1) img_score=%s", w_img, img_score)

        if img_score is None or w_img <= 0.0:
            final_score = w_acc_default * acc + w_fmt_default * fmt
            logger.info(
                "[%s] fmt=%.2f acc=%.3f final=%.3f (no image)\n[SOLUTION]: %s\n[GT]: %s",
                data_source,
                fmt,
                acc,
                final_score,
                truncate(solution_str.strip().replace("\n", "\\n")),
                truncate((ground_truth or "").strip().replace("\n", "\\n")),
            )
            return {"score": final_score, "acc": acc}

        w_remain = max(0.0, 1.0 - w_img)
        w_acc = w_remain * 0.9
        w_fmt = w_remain * 0.1

        final_score = w_acc * acc + w_fmt * fmt + w_img * float(img_score)

        logger.info(
            "[%s] fmt=%.2f acc=%.3f img=%.3f | w=(%.2f,%.2f,%.2f) final=%.3f\n[SOLUTION]: %s\n[GT]: %s",
            data_source,
            fmt,
            acc,
            float(img_score),
            w_acc,
            w_fmt,
            w_img,
            final_score,
            truncate(solution_str.strip().replace("\n", "\\n")),
            truncate((ground_truth or "").strip().replace("\n", "\\n")),
        )
        return {"score": final_score, "acc": acc, "image": float(img_score)}

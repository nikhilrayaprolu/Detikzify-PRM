# generate_prm_dataset.py

import io
import re
import os
import base64
import random
import logging
import gc
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Enable TF32 for massive speedups on RTX 30/40 series GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    train_parquet: str = "datikz-train.parquet"
    test_parquet:  str = "datikz-test.parquet"
    output_dir:    str = "prm_dataset"

    error_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"

    # Batch size for both DataLoader CPU chunks and GPU generation
    batch_size: int = 128

    # How many partial-code prefixes to sample per source example
    num_prefixes_per_sample: int = 3

    max_samples_train: Optional[int] = 5000
    max_samples_test:  Optional[int] = 500

    device:     str = "auto"
    torch_dtype: str = "bfloat16"


# ---------------------------------------------------------------------------
# Error-introduction prompt
# ---------------------------------------------------------------------------

ERROR_TYPES = [
    "wrong_angle", "missing_label", "shifted_coordinate", "wrong_line_style",
    "extra_element", "wrong_color", "wrong_scale", "missing_arrow", "wrong_shape",
]

ERROR_PROMPT = """\
You are a TikZ code mutator.

Below is one line of TikZ code. Rewrite it with ONE subtle error of type "{error_type}".

Error type meanings:
- wrong_angle        : change an angle value by 15-45 degrees
- missing_label      : remove a node label or annotation
- shifted_coordinate : shift one (x,y) coordinate by 0.3-0.7 units
- wrong_line_style   : change dashed/dotted/solid or add/remove an arrow tip
- extra_element      : append a small extra node or path command
- wrong_color        : replace one color with a visually similar but wrong color
- wrong_scale        : slightly scale one numeric size/radius by 0.5x or 1.5x
- missing_arrow      : remove one arrow tip or reverse it
- wrong_shape        : change circle→ellipse, rectangle→square (wrong ratio), etc.

Rules:
1. Change ONLY this single line.
2. Keep the line syntactically valid TikZ.
3. Return ONLY the modified line — no explanation, no markdown fences.

Original line:
{line}

Modified line:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_image(row: pd.Series) -> Optional[Image.Image]:
    col = row.get("image")
    if col is None: return None
    if isinstance(col, Image.Image): return col.convert("RGB")
    if isinstance(col, bytes): return Image.open(io.BytesIO(col)).convert("RGB")
    if isinstance(col, dict):
        raw = col.get("bytes") or col.get("data")
        if raw: return Image.open(io.BytesIO(raw)).convert("RGB")
    if isinstance(col, str):
        try: return Image.open(io.BytesIO(base64.b64decode(col))).convert("RGB")
        except Exception:
            try: return Image.open(col).convert("RGB")
            except Exception: return None
    return None

def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def split_tikz_lines(code: str) -> List[str]:
    return [ln for ln in code.splitlines() if ln.strip()]

def is_boilerplate(line: str) -> bool:
    PATTERNS = [
        r"^\s*\\documentclass", r"^\s*\\usepackage", r"^\s*\\usetikzlibrary",
        r"^\s*\\begin\{document\}", r"^\s*\\end\{document\}",
        r"^\s*\\begin\{tikzpicture\}", r"^\s*\\end\{tikzpicture\}",
        r"^\s*\\tikzset", r"^\s*\\pgfmathsetmacro", r"^\s*\\definecolor",
        r"^\s*\\newcommand", r"^\s*%", r"^\s*$",
    ]
    return any(re.match(p, line) for p in PATTERNS)

def pick_scorable_prefix_indices(lines: List[str], n: int) -> List[int]:
    candidates = []
    depth = 0
    for i, ln in enumerate(lines):
        depth += ln.count(r"\begin{tikzpicture}")
        depth -= ln.count(r"\end{tikzpicture}")
        if depth > 0 and not is_boilerplate(ln) and len(ln.strip()) <= 400:
            candidates.append(i)

    if not candidates: return []
    step = max(1, len(candidates) // n)
    return candidates[::step][:n]


# ---------------------------------------------------------------------------
# Multiprocessing PyTorch Dataset
# ---------------------------------------------------------------------------

class TikzPreparationDataset(Dataset):
    """
    Dataset to handle the heavily CPU-bound tasks (Image Decoding, TikZ parsing)
    across multiple CPU cores via PyTorch's DataLoader.
    """
    def __init__(self, df: pd.DataFrame, config: DatasetConfig, split: str):
        self.df = df
        self.config = config
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = extract_image(row)
        if image is None: return [], []

        code = row.get("code", "")
        if not isinstance(code, str) or len(code.strip()) < 20: return [], []

        image_b64 = image_to_base64(image)
        lines = split_tikz_lines(code)
        prefix_indices = pick_scorable_prefix_indices(lines, self.config.num_prefixes_per_sample)

        pos_rows, neg_tasks = [], []
        for last_idx in prefix_indices:
            prefix_lines = lines[: last_idx + 1]
            last_line    = lines[last_idx]

            pos_rows.append({
                "image_b64":    image_b64,
                "partial_code": "\n".join(prefix_lines),
                "last_line":    last_line,
                "score":        1.0,
                "error_type":   "none",
                "split":        self.split,
                "source_idx":   int(idx),
            })

            neg_tasks.append({
                "last_line":    last_line,
                "prefix_lines": prefix_lines,
                "error_type":   random.choice(ERROR_TYPES),
                "image_b64":    image_b64,
                "source_idx":   int(idx),
            })

        return pos_rows, neg_tasks

def collate_tasks(batch):
    """Flattens the lists of outputs from the CPU workers."""
    all_pos = []
    all_neg = []
    for pos, neg in batch:
        all_pos.extend(pos)
        all_neg.extend(neg)
    return all_pos, all_neg


# ---------------------------------------------------------------------------
# Error Generator (Batched LLM Generation)
# ---------------------------------------------------------------------------

class ErrorGenerator:
    def __init__(self, model_name: str, device: str, torch_dtype):
        logger.info(f"Loading error model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device
        )
        self.model.eval()
        self._dev = self.model.device

    def corrupt_lines_batched(self, lines: List[str], error_types: List[str], batch_size: int) -> List[str]:
        results = []
        
        # Inner loop to handle chunks safely passed by the DataLoader
        for i in range(0, len(lines), batch_size):
            batch_lines = lines[i : i + batch_size]
            batch_errors = error_types[i : i + batch_size]

            prompts = [
                ERROR_PROMPT.format(error_type=e, line=l.strip())
                for e, l in zip(batch_errors, batch_lines)
            ]

            chats = [[{"role": "user", "content": p}] for p in prompts]
            texts = [
                self.tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                for c in chats
            ]

            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self._dev)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=128, do_sample=True, temperature=0.7,
                    top_p=0.9, pad_token_id=self.tokenizer.pad_token_id,
                )

            input_lens = inputs["input_ids"].shape[1]
            new_tokens = outputs[:, input_lens:]
            decoded_responses = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for orig_line, err_type, resp in zip(batch_lines, batch_errors, decoded_responses):
                first_line = resp.splitlines()[0].strip() if resp.strip() else ""
                first_line = re.sub(r"^```[a-z]*\s*", "", first_line)
                first_line = re.sub(r"\s*```$", "", first_line)

                if not first_line or first_line == orig_line.strip():
                    first_line = self._rule_based(orig_line, err_type)
                results.append(first_line)
                
            del inputs, outputs, new_tokens
            gc.collect()
            torch.cuda.empty_cache()

        return results

    def _rule_based(self, line: str, error_type: str) -> str:
        if error_type == "wrong_color":
            swaps = {"blue": "cyan", "cyan": "blue", "red": "orange", "orange": "red", "green": "lime"}
            for src, dst in swaps.items():
                if src in line: return line.replace(src, dst, 1)

        if error_type == "wrong_line_style":
            for src, dst in [("dashed", "dotted"), ("dotted", "dashed"), ("thick", "thin"), ("thin", "thick")]:
                if src in line: return line.replace(src, dst, 1)

        if error_type == "shifted_coordinate":
            matches = list(re.finditer(r'\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)', line))
            if matches:
                m = random.choice(matches)
                x, y = float(m.group(1)), float(m.group(2))
                return line[:m.start()] + f"({x + random.choice([-0.5, 0.5]):.1f},{y})" + line[m.end():]

        if error_type == "missing_label": return re.sub(r'\{[^{}]{1,60}\}', '{}', line, count=1)
        if error_type == "wrong_angle":
            return re.sub(r'\b\d{2,3}\b', lambda m: str(int(float(m.group(0)) + 30)), line, count=1)

        return line


# ---------------------------------------------------------------------------
# Pipelined Dataset builder
# ---------------------------------------------------------------------------

def build_rows(
    df: pd.DataFrame,
    error_gen: ErrorGenerator,
    config: DatasetConfig,
    split: str,
) -> pd.DataFrame:
    
    dataset = TikzPreparationDataset(df, config, split)
    
    # Cap workers at 16 to avoid RAM exhaustion, use CPU count if lower
    num_workers = min(os.cpu_count() or 4, 16)
    
    loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_tasks,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=False  # Only for Tensors, disabled to avoid overhead on strings/dicts
    )

    all_pos_rows = []
    all_neg_rows = []

    # CPU and GPU pipeline execution loop
    for pos_chunk, neg_tasks_chunk in tqdm(loader, desc=f"Pipelined Processing [{split}]"):
        all_pos_rows.extend(pos_chunk)
        
        if not neg_tasks_chunk:
            continue

        lines_to_corrupt = [t["last_line"] for t in neg_tasks_chunk]
        errors_to_apply  = [t["error_type"] for t in neg_tasks_chunk]

        # The GPU processes this chunk while the CPU is prefetching the next chunk
        corrupted_lines = error_gen.corrupt_lines_batched(
            lines_to_corrupt, errors_to_apply, batch_size=config.batch_size
        )

        for task, corrupted in zip(neg_tasks_chunk, corrupted_lines):
            if corrupted == task["last_line"]:
                continue
                
            neg_lines = task["prefix_lines"][:-1] + [corrupted]
            all_neg_rows.append({
                "image_b64":    task["image_b64"],
                "partial_code": "\n".join(neg_lines),
                "last_line":    corrupted,
                "score":        0.0,
                "error_type":   task["error_type"],
                "split":        split,
                "source_idx":   task["source_idx"],
            })

    return pd.DataFrame(all_pos_rows + all_neg_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = DatasetConfig()
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if config.torch_dtype == "bfloat16" else torch.float32
    error_gen = ErrorGenerator(config.error_model_name, config.device, dtype)

    train_path = Path(config.train_parquet)
    test_path = Path(config.test_parquet)

    df_train_src = pd.read_parquet(train_path)
    logger.info(f"Loaded {len(df_train_src)} rows from {train_path}")

    if not test_path.exists():
        logger.warning(f"Test parquet '{test_path}' not found. Extracting test samples from train parquet.")
        df_train_src = df_train_src.sample(frac=1.0, random_state=42).reset_index(drop=True)
        num_test = config.max_samples_test if config.max_samples_test else int(0.1 * len(df_train_src))
        num_test = min(num_test, max(1, len(df_train_src) // 5)) 
        
        df_test_src = df_train_src.iloc[:num_test].reset_index(drop=True)
        df_train_src = df_train_src.iloc[num_test:].reset_index(drop=True)
        logger.info(f"Extracted {len(df_test_src)} rows for testing. Remaining train rows: {len(df_train_src)}")
    else:
        df_test_src = pd.read_parquet(test_path)
        logger.info(f"Loaded {len(df_test_src)} rows from {test_path}")

    if config.max_samples_train and len(df_train_src) > config.max_samples_train:
        df_train_src = df_train_src.sample(n=config.max_samples_train, random_state=42).reset_index(drop=True)
        
    if config.max_samples_test and len(df_test_src) > config.max_samples_test:
        df_test_src = df_test_src.sample(n=config.max_samples_test, random_state=42).reset_index(drop=True)

    for split, df_src in [("train", df_train_src), ("test", df_test_src)]:
        logger.info(f"Processing '{split}' split with {len(df_src)} source examples...")
        
        df_out = build_rows(df_src, error_gen, config, split)

        out_path = Path(config.output_dir) / f"prm_{split}.parquet"
        df_out.to_parquet(out_path, index=False)

        pos = (df_out["score"] == 1.0).sum()
        neg = (df_out["score"] == 0.0).sum()
        logger.info(f"{split}: {len(df_out)} rows saved → {out_path}  (pos={pos}, neg={neg})")

if __name__ == "__main__":
    main()
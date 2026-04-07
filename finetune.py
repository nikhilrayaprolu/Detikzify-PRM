import io
import re
import base64
import logging
import random
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    model_name:  str = "Qwen/Qwen3-VL-2B-Instruct"
    output_dir:  str = "qwen3vl-prm-finetuned"
    train_data:  str = "prm_dataset/prm_train.parquet"
    eval_data:   str = "prm_dataset/prm_test.parquet"

    # LoRA
    lora_r:       int = 16
    lora_alpha:   int = 32
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    num_epochs:       int   = 3
    batch_size:       int   = 8
    grad_accumulation: int  = 8
    learning_rate:    float = 2e-5
    weight_decay:     float = 0.01
    warmup_ratio:     float = 0.1
    max_grad_norm:    float = 1.0
    max_seq_length:   int   = 512

    eval_steps:    int = 100
    save_steps:    int = 200
    logging_steps: int = 10

    seed: int = 42


# ---------------------------------------------------------------------------
# Prompt (same format used at inference time)
# ---------------------------------------------------------------------------

SCORE_PROMPT = """\
You are a TikZ code verifier. Given a target image and a partial TikZ program, \
assess how well the last line of code is progressing toward reproducing the target image.

Partial TikZ code (the LAST line is the one being evaluated):
```latex
{code}
```

Rate progress from 0.0 (wrong/harmful last line) to 1.0 (correct last line).
Respond with ONLY a single decimal number.

Score:"""


def make_messages(image: Image.Image, partial_code: str) -> List[Dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": SCORE_PROMPT.format(code=partial_code)},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def b64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


class PRMDataset(Dataset):
    def __init__(self, parquet_path: str, processor, max_seq_length: int):
        self.processor      = processor
        self.max_seq_length = max_seq_length

        df = pd.read_parquet(parquet_path)
        # Shuffle so positives and negatives are interleaved
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.df = df
        logger.info(
            f"Dataset loaded: {len(df)} rows  "
            f"(pos={( df['score']==1.0).sum()}, neg={(df['score']==0.0).sum()})"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row   = self.df.iloc[idx]
        image = b64_to_image(row["image_b64"])
        code  = row["partial_code"]
        score = float(row["score"])

        # Target string the model should generate
        target_str = f"{score:.1f}"

        messages = make_messages(image, code)

        # Apply chat template → text string
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Encode input
        inputs = self.processor(
            images=[image],
            text=text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )
        input_ids = inputs["input_ids"].squeeze(0)   # (L,)

        # Encode target tokens (the score string + EOS)
        target_ids = self.processor.tokenizer(
            target_str,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0)
        eos = torch.tensor([self.processor.tokenizer.eos_token_id])
        target_ids = torch.cat([target_ids, eos])    # (T,)

        # Full sequence: prompt + target
        full_ids = torch.cat([input_ids, target_ids])

        # Labels: -100 for prompt tokens, actual ids for target tokens
        labels = torch.full_like(full_ids, fill_value=-100)
        labels[len(input_ids):] = target_ids

        return {
            "input_ids":      full_ids,
            "labels":         labels,
            "attention_mask": torch.ones_like(full_ids),
            "pixel_values":   inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    max_len = max(s["input_ids"].shape[0] for s in batch)

    input_ids      = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels         = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, s in enumerate(batch):
        n = s["input_ids"].shape[0]
        input_ids[i, :n]      = s["input_ids"]
        attention_mask[i, :n] = s["attention_mask"]
        labels[i, :n]         = s["labels"]

    result = {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }

    # Stack visual tensors (Qwen3-VL uses flat pixel_values across the batch)
    pv_list  = [s["pixel_values"]   for s in batch if s.get("pixel_values")   is not None]
    thw_list = [s["image_grid_thw"] for s in batch if s.get("image_grid_thw") is not None]
    if pv_list:
        result["pixel_values"]   = torch.cat(pv_list,  dim=0)
    if thw_list:
        result["image_grid_thw"] = torch.cat(thw_list, dim=0)

    return result


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PRMTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

    # ------------------------------------------------------------------
    def _setup_model(self):
        logger.info(f"Loading {self.config.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_targets,
            bias="none",
            inference_mode=False,
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()
        self.model.train()

    # ------------------------------------------------------------------
    def _setup_data(self):
        self.train_ds = PRMDataset(
            self.config.train_data, self.processor, self.config.max_seq_length
        )
        self.eval_ds = PRMDataset(
            self.config.eval_data, self.processor, self.config.max_seq_length
        )
        self.train_loader = DataLoader(
            self.train_ds, batch_size=self.config.batch_size,
            shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True, prefetch_factor=2
        )
        self.eval_loader = DataLoader(
            self.eval_ds, batch_size=self.config.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=4, prefetch_factor=2
        )

    # ------------------------------------------------------------------
    def _setup_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        steps_per_epoch  = math.ceil(len(self.train_loader) / self.config.grad_accumulation)
        total_steps      = steps_per_epoch * self.config.num_epochs
        warmup_steps     = int(total_steps * self.config.warmup_ratio)
        self.scheduler   = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )
        self.total_steps = total_steps
        self.global_step = 0
        logger.info(f"total_steps={total_steps}, warmup={warmup_steps}")

    # ------------------------------------------------------------------
    def _forward(self, batch: Dict):
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        return self.model(
            input_ids      = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            pixel_values   = batch.get("pixel_values"),
            image_grid_thw = batch.get("image_grid_thw"),
            labels         = batch["labels"],
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss, n = 0.0, 0
        for batch in tqdm(self.eval_loader, desc="Eval", leave=False):
            loss = self._forward(batch).loss
            total_loss += loss.item()
            n += 1
        self.model.train()
        return total_loss / max(n, 1)

    # ------------------------------------------------------------------
    def _save(self, tag: str):
        save_dir = Path(self.config.output_dir) / f"checkpoint-{tag}"
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)
        logger.info(f"Saved → {save_dir}")

    # ------------------------------------------------------------------
    def train(self):
        logger.info("=" * 50)
        logger.info("Starting SFT fine-tuning of Qwen3-VL PRM")
        logger.info("=" * 50)

        best_eval_loss = float("inf")

        for epoch in range(self.config.num_epochs):
            logger.info(f"\n── Epoch {epoch + 1}/{self.config.num_epochs} ──")
            self.optimizer.zero_grad()
            running_loss = 0.0

            for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):

                loss = self._forward(batch).loss / self.config.grad_accumulation
                loss.backward()
                running_loss += loss.item() * self.config.grad_accumulation

                if (step + 1) % self.config.grad_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % self.config.logging_steps == 0:
                        avg = running_loss / self.config.logging_steps
                        lr  = self.scheduler.get_last_lr()[0]
                        logger.info(
                            f"step={self.global_step}/{self.total_steps} "
                            f"loss={avg:.4f}  lr={lr:.2e}"
                        )
                        running_loss = 0.0

                    if self.global_step % self.config.eval_steps == 0:
                        eval_loss = self.evaluate()
                        logger.info(f"eval_loss={eval_loss:.4f} @ step {self.global_step}")
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            self._save("best")

                    if self.global_step % self.config.save_steps == 0:
                        self._save(f"step_{self.global_step}")

        self._save("final")
        logger.info(f"Training done. Best eval loss: {best_eval_loss:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trainer = PRMTrainer(TrainConfig())
    trainer.train()

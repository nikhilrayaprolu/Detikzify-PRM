

import re
import traceback
from time import time
from typing import List, Optional, Union

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

from .logger import logger

import gc

class Qwen3VLVerifier:
    """
    Zero-shot Process Reward Model using Qwen3-VL.
    Scores partial TikZ code against a target image at each generation step.
    Boilerplate/preamble lines are assigned a neutral score without calling the model.
    Only actual drawing commands inside \\begin{tikzpicture}...\\end{tikzpicture} are scored.
    """

    DRAWING_PROMPT = """You are a TikZ code verifier. Given a target image and a partial TikZ program, assess how well the drawing commands so far are progressing toward reproducing the target image.

    Focus on:
    - Are the shapes, positions, and relationships between elements consistent with the target?
    - Does the overall structure match what's visible in the image?
    - Are colors, line styles, and proportions reasonable?

    Partial TikZ Code:
    ```latex
    {code}
    ```

    Rate the overall progress toward reproducing the target image.
    Respond with ONLY a single decimal number from 0.0 to 1.0. Do not include any other text, explanation, or context.

    Score:"""

    STRUCTURAL_PROMPT = """You are a TikZ code verifier. Given a target image and a partial TikZ program that is still in the preamble/setup phase, assess whether the document setup is reasonable for reproducing the target image.

    Partial TikZ Code:
    ```latex
    {code}
    ```

    Is this setup reasonable? Respond with ONLY a single decimal number from 0.0 to 1.0. Do not include any other text, explanation, or context.

    Score:"""

    BOILERPLATE_PATTERNS = [
        r"^\s*\\documentclass",
        r"^\s*\\usepackage",
        r"^\s*\\usetikzlibrary",
        r"^\s*\\begin\{document\}",
        r"^\s*\\end\{document\}",
        r"^\s*\\begin\{tikzpicture\}",
        r"^\s*\\tikzset",
        r"^\s*\\pgfmathsetmacro",
        r"^\s*\\definecolor",
        r"^\s*\\newcommand",
        r"^\s*\\renewcommand",
        r"^\s*\\pagestyle",
        r"^\s*\\thispagestyle",
        r"^\s*\\begin\{center\}",
        r"^\s*\\end\{center\}",
        r"^\s*\\centering",
        r"^\s*\\begin\{figure\}",
        r"^\s*\\end\{figure\}",
        r"^\s*%",  # comments
        r"^\s*$",  # blank lines
    ]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: Optional[str] = None,
        torch_dtype=torch.bfloat16,
        boilerplate_score: float = 0.5,
    ):
        logger.info(f"[Verifier] Loading Qwen3-VL verifier: {model_name}")
        load_start = time()
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Ensure padding token is set for batched processing
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device or "auto",
        )
        self.model.eval()
        self._device = next(self.model.parameters()).device
        self._call_count = 0
        self._total_time = 0.0
        self._skipped_count = 0
        self.boilerplate_score = boilerplate_score
        logger.info(
            f"[Verifier] Loaded in {time() - load_start:.1f}s on device={self._device}"
        )

    def _is_boilerplate(self, line: str) -> bool:
        """Check if a line is boilerplate that shouldn't be scored against the image."""
        for pattern in self.BOILERPLATE_PATTERNS:
            if re.match(pattern, line):
                return True
        return False

    def _is_inside_tikzpicture(self, lines: List[str]) -> bool:
        """Check if the end of the lines list is inside a tikzpicture environment."""
        depth = 0
        for line in lines:
            if r"\begin{tikzpicture}" in line:
                depth += 1
            if r"\end{tikzpicture}" in line:
                depth -= 1
        return depth > 0

    def _count_drawing_lines(self, lines: List[str]) -> int:
        """Count lines that are actual drawing commands inside tikzpicture."""
        count = 0
        inside = False
        for line in lines:
            if r"\begin{tikzpicture}" in line:
                inside = True
                continue
            if r"\end{tikzpicture}" in line:
                inside = False
                continue
            if inside and not self._is_boilerplate(line) and line.strip():
                count += 1
        return count

    @torch.inference_mode()
    def score(
        self,
        image: Image.Image,
        partial_code: str,
        last_line: str = "",
    ) -> float:
        """Score a single partial TikZ code. Proxy to batch_score for backward compatibility."""
        return self.batch_score(image, [partial_code], [last_line])[0]

    @torch.inference_mode()
    def batch_score(
        self,
        image: Image.Image,
        partial_codes: List[str],
        last_lines: List[str],
    ) -> List[float]:
        """
        Score a batch of partial TikZ codes against the target image.
        Boilerplate lines get a neutral score without calling the VLM.
        """
        if not partial_codes:
            return []

        self._call_count += len(partial_codes)
        call_start = time()

        scores = [0.5] * len(partial_codes)
        to_score_indices = []
        messages_batch = []

        for i, (code, last_line) in enumerate(zip(partial_codes, last_lines)):
            if not last_line:
                code_lines = code.strip().split("\n")
                last_line = code_lines[-1] if code_lines else ""

            if self._is_boilerplate(last_line):
                self._skipped_count += 1
                scores[i] = self.boilerplate_score
                continue

            lines = code.strip().split("\n")
            inside_tikz = self._is_inside_tikzpicture(lines)
            drawing_lines = self._count_drawing_lines(lines)

            if inside_tikz and drawing_lines >= 1:
                prompt = self.DRAWING_PROMPT.format(code=code)
            else:
                prompt = self.STRUCTURAL_PROMPT.format(code=code)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            messages_batch.append(messages)
            to_score_indices.append(i)

        if messages_batch:
            inputs = self.processor.apply_chat_template(
                messages_batch,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            inputs = inputs.to(self._device)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                temperature=1.0,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            responses = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for idx, response in zip(to_score_indices, responses):
                scores[idx] = self._parse_score(response.strip())

        elapsed = time() - call_start
        self._total_time += elapsed

        if to_score_indices:
            logger.debug(
                f"[Verifier] batch_score for {len(partial_codes)} items "
                f"({len(to_score_indices)} model calls) in {elapsed:.2f}s"
            )

        return scores

    def log_stats(self):
        """Log cumulative verifier statistics."""
        if self._call_count > 0:
            avg = self._total_time / self._call_count
            logger.info(
                f"[Verifier] Stats: {self._call_count} calls "
                f"({self._skipped_count} skipped as boilerplate), "
                f"total={self._total_time:.1f}s, avg={avg:.2f}s/call"
            )

    @staticmethod
    def _parse_score(response: str) -> float:
        """Extract a numerical score from the verifier's response."""
        clean = response.strip()

        patterns = [
            r"^(\d+\.\d+)",  # starts with decimal like 0.75
            r"(\d+\.\d+)",  # decimal anywhere
            r"(\d+)/(\d+)",  # fraction like 7/10
            r"(\d+)%",  # percentage like 75%
            r"^(\d+)$",  # just an integer
        ]

        for pattern in patterns:
            match = re.search(pattern, clean)
            if match:
                groups = match.groups()
                if len(groups) == 2:  # fraction
                    denom = float(groups[1])
                    if denom == 0:
                        return 0.5
                    return max(0.0, min(1.0, float(groups[0]) / denom))
                val = float(groups[0])
                if val > 1.0:
                    if val <= 10:
                        return val / 10.0
                    elif val <= 100:
                        return val / 100.0
                return max(0.0, min(1.0, val))

        logger.warning(
            f'[Verifier] Could not parse score from: "{clean[:100]}", defaulting to 0.5'
        )
        return 0.5





class FinetunedVerifier(Qwen3VLVerifier):
    """
    LoRA-finetuned Qwen3-VL PRM verifier with memory optimizations.
    """

    FINETUNE_PROMPT = """You are a TikZ code verifier. Given a target image and a partial TikZ program, assess how well the last line of code is progressing toward reproducing the target image.

Partial TikZ code (the LAST line is the one being evaluated):
```latex
{code}
```

Rate progress from 0.0 (wrong/harmful last line) to 1.0 (correct last line).
Respond with ONLY a single decimal number.

Score:"""

    def __init__(
        self,
        checkpoint_path: str,
        base_model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: Optional[str] = None,
        torch_dtype=torch.bfloat16,
        boilerplate_score: float = 0.5,
        merge_lora: bool = True,           # ← Merge LoRA for inference
        load_in_4bit: bool = False,        # ← Enable 4-bit quantization
        load_in_8bit: bool = False,        # ← Enable 8-bit quantization
    ):
        logger.info(
            f"[FinetunedVerifier] Initializing: checkpoint={checkpoint_path}, "
            f"base={base_model_name}, merge_lora={merge_lora}, "
            f"load_in_4bit={load_in_4bit}, load_in_8bit={load_in_8bit}"
        )
        load_start = time()

        self.boilerplate_score = boilerplate_score
        self._call_count = 0
        self._skipped_count = 0
        self._total_time = 0.0

        try:
            self.processor = AutoProcessor.from_pretrained(base_model_name)
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = (
                    self.processor.tokenizer.eos_token
                )

            # --- Quantization config ---
            quantization_config = None
            if load_in_4bit or load_in_8bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                logger.info(
                    f"[FinetunedVerifier] Using "
                    f"{'4-bit' if load_in_4bit else '8-bit'} quantization"
                )

            logger.info(f"[FinetunedVerifier] Loading base model: {base_model_name}")
            base = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_name,
                torch_dtype=torch_dtype if not quantization_config else None,
                device_map=device or "auto",
                quantization_config=quantization_config,
            )

            logger.info(
                f"[FinetunedVerifier] Loading PEFT adapter from {checkpoint_path}"
            )
            self.model = PeftModel.from_pretrained(
                base,
                checkpoint_path,
                is_trainable=False,    # ← Inference only, saves memory
            )

            # --- Merge LoRA weights into base model ---
            # Eliminates separate adapter tensors, reduces memory overhead
            if merge_lora and not (load_in_4bit or load_in_8bit):
                logger.info(
                    "[FinetunedVerifier] Merging LoRA weights into base model..."
                )
                self.model = self.model.merge_and_unload()
                logger.info("[FinetunedVerifier] LoRA merged successfully")

            self.model.eval()
            self._device = next(self.model.parameters()).device

            # Force cleanup after loading
            gc.collect()
            torch.cuda.empty_cache()

            logger.info(
                f"[FinetunedVerifier] Loaded in {time() - load_start:.1f}s "
                f"on device={self._device}"
            )
            self._log_memory("after_init")

        except Exception as e:
            logger.error(f"[FinetunedVerifier] Failed to initialize: {e}")
            traceback.print_exc()
            raise

    def _log_memory(self, tag: str = ""):
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self._device) / 1e9
            reserved = torch.cuda.memory_reserved(self._device) / 1e9
            logger.debug(
                f"[FinetunedVerifier] GPU Memory [{tag}]: "
                f"allocated={allocated:.2f}GB, reserved={reserved:.2f}GB"
            )

    @torch.inference_mode()   # ← Critical: was missing in original
    def batch_score(
        self,
        image: Image.Image,
        partial_codes: List[str],
        last_lines: List[str],
    ) -> List[float]:
        """
        Score partial TikZ codes against the target image.
        """
        if not partial_codes:
            return []

        self._call_count += len(partial_codes)
        call_start = time()

        scores = [self.boilerplate_score] * len(partial_codes)
        to_score_idx = []
        messages_batch = []

        # --- Step 1: Filter boilerplate without calling model ---
        for i, (code, last_line) in enumerate(zip(partial_codes, last_lines)):
            if not last_line:
                lines = code.strip().splitlines()
                last_line = lines[-1] if lines else ""

            if self._is_boilerplate(last_line):
                self._skipped_count += 1
                continue

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": self.FINETUNE_PROMPT.format(code=code),
                        },
                    ],
                }
            ]
            messages_batch.append(messages)
            to_score_idx.append(i)

        # --- Step 2: Single batched inference call with proper cleanup ---
        if messages_batch:
            inputs = None
            generated = None
            try:
                inputs = self.processor.apply_chat_template(
                    messages_batch,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                ).to(self._device)

                logger.debug(
                    f"[FinetunedVerifier] Inference: "
                    f"batch_size={inputs['input_ids'].shape[0]}, "
                    f"seq_len={inputs['input_ids'].shape[-1]}"
                )
                self._log_memory("before_generate")

                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=16,
                    do_sample=False,
                    use_cache=True,       # ← KV cache for efficiency
                )

                trimmed = [
                    out[len(inp):]
                    for inp, out in zip(inputs["input_ids"], generated)
                ]
                responses = self.processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                for idx, resp in zip(to_score_idx, responses):
                    scores[idx] = self._parse_score(resp.strip())

            finally:
                # --- Explicit cleanup to free GPU memory immediately ---
                if inputs is not None:
                    del inputs
                if generated is not None:
                    del generated
                torch.cuda.empty_cache()
                self._log_memory("after_generate")

        self._total_time += time() - call_start
        logger.debug(
            f"[FinetunedVerifier] batch_score: {len(partial_codes)} items "
            f"({len(to_score_idx)} model calls) in "
            f"{time() - call_start:.2f}s"
        )
        return scores
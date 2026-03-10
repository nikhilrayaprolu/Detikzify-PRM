from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from math import sqrt
from multiprocessing.pool import ThreadPool
from re import sub
from time import time
from types import SimpleNamespace as Namespace
from typing import Any, Dict, Generator, List, Literal, Optional, Set, Tuple, Union
import logging
import re

from PIL import Image
import torch
from torchmetrics import Metric
from transformers import StoppingCriteriaList, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.generation.streamers import BaseStreamer

from ..evaluate.imagesim import ImageSim
from ..model.adapter import has_adapter
from ..util import (
    ExplicitAbort,
    StreamerList,
    TokenStreamer,
    cache_cast,
    expand,
    load,
    unwrap_processor as unwrap,
)
from .tikz import TikzDocument

Numeric = Union[int, float]


class SimpleLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")

    def debug(self, msg):
        print(f"[DEBUG] {msg}")

    def warning(self, msg):
        print(f"[WARNING] {msg}")

    def error(self, msg):
        print(f"[ERROR] {msg}")


logger = SimpleLogger()


@dataclass
class BeamCandidate:
    """Represents a single beam in the verified beam search."""
    token_ids: torch.Tensor
    lines: List[str]
    score: float = 0.0
    finished: bool = False
    scored_steps: int = 0  # number of non-boilerplate steps that were actually scored

    def __hash__(self):
        return hash(tuple(self.token_ids.tolist()))

    def __eq__(self, other):
        try:
            return self.token_ids.equal(other.token_ids)
        except (AttributeError, TypeError):
            return False

    def summary(self, max_line_len: int = 80) -> str:
        """Return a short summary string for logging."""
        last_line = self.lines[-1] if self.lines else "<empty>"
        if len(last_line) > max_line_len:
            last_line = last_line[: max_line_len - 3] + "..."
        status = "FINISHED" if self.finished else "active"
        return (
            f"[{status}] score={self.score:.4f} lines={len(self.lines)} "
            f"scored_steps={self.scored_steps} "
            f"tokens={self.token_ids.numel()} last_line=\"{last_line}\""
        )


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
            dtype=torch_dtype,
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


class DetikzifyGenerator:
    """
    Generator that uses Verified Beam Search with Qwen3-VL as a
    Process Reward Model (PRM).

    At each step:
    1. DeTikZify generates K candidate next-lines via batched nucleus sampling.
    2. Boilerplate lines (preamble, \\usepackage, etc.) get a neutral pass-through score.
    3. Drawing commands are scored by the Qwen3-VL verifier against the target image.
    4. The top-B candidates (beam width) are retained.
    5. Search terminates when \\end{document} is generated or max depth reached.
    """

    def __init__(
        self,
        model,
        processor,
        image: Optional[Image.Image],
        text: Optional[str] = None,
        metric: Optional[Metric] = None,
        compile_timeout: Optional[int] = 60,
        beam_timeout: Optional[int] = None,
        streamer: Optional[BaseStreamer] = None,
        control: Optional[ExplicitAbort] = None,
        num_candidates: int = 5,
        beam_width: int = 3,
        max_lines: int = 50,
        verifier: Optional[Qwen3VLVerifier] = None,
        verifier_model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        strict: bool = False,
        **gen_kwargs,
    ):
        self.model = model
        self.processor = processor
        self.metric = metric
        self.image = image
        self.text = text
        self.compile_timeout = compile_timeout
        self.beam_timeout = beam_timeout
        self.streamer = streamer
        self.strict = strict
        self.gen_kwargs = gen_kwargs

        self.num_candidates = num_candidates
        self.beam_width = beam_width
        self.max_lines = max_lines

        self.control = control or ExplicitAbort()

        if verifier is not None:
            self.verifier = verifier
        else:
            self.verifier = Qwen3VLVerifier(model_name=verifier_model_name)

        # Cache processor inputs immediately to avoid doing vision encoding iteratively
        with torch.inference_mode():
            self._cached_proc_inputs = self.processor(
                images=self.image,
                text=self.text,
                text_kwargs={"truncation": True},
                return_tensors="pt",
            ).to(self.model.device)

            self.initial_input_ids = self._cached_proc_inputs.input_ids.squeeze()
            
            self._cached_adapter_kwargs = {
                k: v
                for k, v in self._cached_proc_inputs.items()
                if k.startswith("adapter")
            }

        self._decode_cache = {}

        logger.info(
            f"[Generator] Initialized: beam_width={beam_width}, "
            f"num_candidates={num_candidates}, max_lines={max_lines}, "
            f"beam_timeout={beam_timeout}, "
            f"initial_tokens={self.initial_input_ids.numel()}, "
            f"image={'yes' if image is not None else 'no'}, "
            f"text={'yes' if text is not None else 'no'}"
        )

    def __call__(self, *args, **kwargs):
        return self.simulate(*args, **kwargs)

    def simulate(
        self, expansions: Optional[Numeric] = 1
    ) -> Generator[Tuple[Numeric, TikzDocument], None, None]:
        """
        Run Verified Beam Search. Yields (score, TikzDocument) tuples.
        """
        start_time = time()
        iteration = 0
        total_results = 0

        logger.info(
            f"[Simulate] Starting simulation: expansions={expansions}, "
            f"beam_timeout={self.beam_timeout}"
        )

        while expansions is None or iteration < expansions:
            iteration += 1
            iter_start = time()
            logger.info(f"[Simulate] === Round {iteration}/{expansions or '∞'} ===")

            results = list(self._beam_search())
            round_elapsed = time() - iter_start

            if results:
                results.sort(key=lambda x: x[0], reverse=True)
                logger.info(
                    f"[Simulate] Round {iteration} completed in {round_elapsed:.1f}s: "
                    f"{len(results)} results"
                )
                for rank, (score, tikz) in enumerate(results, 1):
                    is_raster = tikz.is_rasterizable
                    has_errors = tikz.compiled_with_errors if is_raster else None
                    code_lines = (
                        len(tikz.code.strip().split("\n")) if tikz.code else 0
                    )
                    logger.info(
                        f"[Simulate]   Result #{rank}: score={score:.4f}, "
                        f"rasterizable={is_raster}, errors={has_errors}, "
                        f"code_lines={code_lines}"
                    )
                    total_results += 1
                    yield (score, tikz)
            else:
                logger.warning(
                    f"[Simulate] Round {iteration}: No beam results, "
                    f"falling back to single sample"
                )
                tikz = self.sample()
                score = self._score_tikz(tikz)
                logger.info(
                    f"[Simulate]   Fallback sample: score={score:.4f}, "
                    f"rasterizable={tikz.is_rasterizable}"
                )
                total_results += 1
                yield (score, tikz)

            total_elapsed = time() - start_time
            if self.beam_timeout is not None and total_elapsed > self.beam_timeout:
                logger.info(
                    f"[Simulate] Timeout reached: {total_elapsed:.1f}s > "
                    f"{self.beam_timeout}s after {iteration} rounds, "
                    f"{total_results} total results"
                )
                return

        total_elapsed = time() - start_time
        logger.info(
            f"[Simulate] Finished: {iteration} rounds, {total_results} results, "
            f"{total_elapsed:.1f}s total"
        )
        self.verifier.log_stats()

    def _beam_search(
        self,
    ) -> Generator[Tuple[Numeric, TikzDocument], None, None]:
        """
        Perform one round of Verified Beam Search.
        """
        search_start = time()
        tokenizer = unwrap(self.processor).tokenizer
        eos_token_id = tokenizer.eos_token_id

        beams: List[BeamCandidate] = [
            BeamCandidate(
                token_ids=self.initial_input_ids.clone(),
                lines=[],
                score=0.0,
                finished=False,
                scored_steps=0,
            )
        ]

        logger.info(
            f"[BeamSearch] Starting beam search: beam_width={self.beam_width}, "
            f"num_candidates={self.num_candidates}, max_lines={self.max_lines}"
        )

        for step in range(self.max_lines):
            step_start = time()

            if self.control.should_stop:
                logger.info(
                    f"[BeamSearch] Step {step}: Abort signal received, stopping"
                )
                break

            active_beams = [b for b in beams if not b.finished]
            finished_beams = [b for b in beams if b.finished]

            logger.info(
                f"[BeamSearch] ── Step {step + 1}/{self.max_lines} ── "
                f"active={len(active_beams)}, finished={len(finished_beams)}"
            )

            for i, beam in enumerate(beams):
                logger.debug(f"[BeamSearch]   Beam {i}: {beam.summary()}")

            if not active_beams:
                logger.info(
                    f"[BeamSearch] All {len(beams)} beams finished at step {step + 1}"
                )
                break

            all_candidates: List[BeamCandidate] = []
            new_candidates_info = []

            # Step 1: Generate candidates across all beams
            for beam_idx, beam in enumerate(beams):
                if beam.finished:
                    all_candidates.append(beam)
                    continue

                candidates = self._generate_candidate_lines(beam, num_candidates=self.num_candidates)
                
                for cand_idx, (candidate_line, candidate_token_ids) in enumerate(candidates):
                    partial_lines = beam.lines + [candidate_line]
                    is_boilerplate = self.verifier._is_boilerplate(candidate_line)
                    
                    is_finished = (
                        "\\end{document}" in candidate_line or 
                        (candidate_token_ids.numel() > 0 and candidate_token_ids[-1].item() == eos_token_id) or
                        candidate_line.strip() == "```"
                    )
                    
                    new_candidates_info.append({
                        "beam": beam,
                        "cand_line": candidate_line,
                        "cand_token_ids": candidate_token_ids,
                        "partial_lines": partial_lines,
                        "is_boilerplate": is_boilerplate,
                        "is_finished": is_finished,
                        "beam_idx": beam_idx
                    })

            # Step 2: Batched Process Reward Model verification
            if new_candidates_info:
                if self.image is not None:
                    score_start = time()
                    partial_codes = ["\n".join(info["partial_lines"]) for info in new_candidates_info]
                    last_lines = [info["cand_line"] for info in new_candidates_info]
                    
                    verifier_scores = self.verifier.batch_score(self.image, partial_codes, last_lines)
                    logger.debug(f"[BeamSearch]   Scored {len(partial_codes)} unique candidates in {time() - score_start:.2f}s")
                else:
                    verifier_scores = [0.5] * len(new_candidates_info)

                # Step 3: Compute discounted cumulative scores
                for info, v_score in zip(new_candidates_info, verifier_scores):
                    beam = info["beam"]
                    candidate_line = info["cand_line"]
                    
                    if info["is_boilerplate"]:
                        cumulative_score = beam.score
                        new_scored_steps = beam.scored_steps
                    else:
                        new_scored_steps = beam.scored_steps + 1
                        # Exponential recency weighting emphasizes the quality of recent lines
                        alpha = 0.7 
                        if beam.scored_steps == 0:
                            cumulative_score = v_score
                        else:
                            cumulative_score = alpha * v_score + (1 - alpha) * beam.score

                    # Diversity discount: penalize if the line appears identically in other existing beams
                    discount = 1.0
                    for other_beam in beams:
                        if candidate_line in other_beam.lines:
                            discount *= 0.95
                    cumulative_score *= discount

                    cand_line_preview = candidate_line[:80] + ("..." if len(candidate_line) > 80 else "")
                    logger.debug(
                        f"[BeamSearch]     Beam {info['beam_idx']} -> "
                        f"verifier={v_score:.4f} cumulative={cumulative_score:.4f} "
                        f"scored_steps={new_scored_steps} "
                        f'line="{cand_line_preview}"'
                    )

                    all_candidates.append(
                        BeamCandidate(
                            token_ids=info["cand_token_ids"],
                            lines=info["partial_lines"],
                            score=cumulative_score,
                            finished=info["is_finished"],
                            scored_steps=new_scored_steps,
                        )
                    )

            if not all_candidates:
                logger.warning(
                    f"[BeamSearch] Step {step + 1}: No candidates generated, stopping"
                )
                break

            # Sort by score descending and keep top-B
            all_candidates.sort(key=lambda c: c.score, reverse=True)

            logger.debug(
                f"[BeamSearch]   Ranking {len(all_candidates)} total candidates "
                f"(keeping top {self.beam_width}):"
            )
            for rank, cand in enumerate(all_candidates):
                if rank < self.beam_width:
                    logger.debug(f"[BeamSearch]     ✓ Rank {rank + 1}: {cand.summary()}")

            beams = all_candidates[: self.beam_width]

            step_elapsed = time() - step_start
            scores_str = ", ".join(f"{b.score:.4f}" for b in beams)
            logger.info(
                f"[BeamSearch]   Step {step + 1} done in {step_elapsed:.1f}s. "
                f"Retained beam scores: [{scores_str}]"
            )

            # Check timeout
            if self.beam_timeout is not None:
                elapsed = time() - search_start
                if elapsed > self.beam_timeout:
                    logger.info(
                        f"[BeamSearch] Beam timeout reached: {elapsed:.1f}s > "
                        f"{self.beam_timeout}s at step {step + 1}"
                    )
                    break

        search_elapsed = time() - search_start
        logger.info(
            f"[BeamSearch] Search complete in {search_elapsed:.1f}s. "
            f"Decoding {len(beams)} beams..."
        )

        for beam_idx, beam in enumerate(beams):
            decode_start = time()
            tikz = self.decode(beam.token_ids)
            final_score = self._final_score(tikz)
            decode_elapsed = time() - decode_start

            code_preview = tikz.code.strip()[:200] if tikz.code else "<empty>"
            code_lines = len(tikz.code.strip().split("\n")) if tikz.code else 0

            logger.info(
                f"[BeamSearch] Final beam {beam_idx}: "
                f"beam_score={beam.score:.4f}, final_score={final_score:.4f}, "
                f"scored_steps={beam.scored_steps}, "
                f"rasterizable={tikz.is_rasterizable}, "
                f"code_lines={code_lines}, decode_time={decode_elapsed:.2f}s"
            )
            logger.debug(
                f"[BeamSearch] Final beam {beam_idx} code preview:\n{code_preview}"
            )

            yield (final_score, tikz)

    def _generate_candidate_lines(
        self,
        beam: BeamCandidate,
        num_candidates: int,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Generate K candidate next-lines using batched generation to avoid duplicate effort.
        Returns list of (line_text, new_token_ids) tuples perfectly aligning token histories.
        """
        candidates = []
        seen_lines: Set[str] = set()
        tokenizer = unwrap(self.processor).tokenizer
        eos_token_id = tokenizer.eos_token_id

        max_length = {
            **self.model.generation_config.to_dict(),
            **self.gen_kwargs,
        }.get("max_length", 4096)

        input_ids = beam.token_ids.clone()

        if (
            input_ids.numel() and input_ids[-1] == eos_token_id
        ) or input_ids.numel() >= max_length:
            return candidates

        gen_start = time()
        with torch.inference_mode():
            batch_size = num_candidates * 2  # Padding to account for likely duplicates
            
            gen_kwargs = self.gen_kwargs.copy()
            gen_kwargs["do_sample"] = True
            
            # Briefly boost temperature to force more diverse candidate generations
            if gen_kwargs.get("temperature", 1.0) < 1.0:
                gen_kwargs["temperature"] = min(1.0, gen_kwargs.get("temperature", 0.7) + 0.3)

            generated = self.model.generate(
                input_ids=input_ids.unsqueeze(0),
                num_return_sequences=batch_size,
                bad_words_ids=[[self.model.config.image_token_id]],
                begin_suppress_tokens=[self.model.config.text_config.eos_token_id],
                pixel_values=self._cached_proc_inputs.get("pixel_values"),
                image_grid_thw=self._cached_proc_inputs.get("image_grid_thw"),
                **self._cached_adapter_kwargs,
                **gen_kwargs,
                max_new_tokens=128,
            )

        gen_elapsed = time() - gen_start
        duplicates_skipped = 0

        for seq_idx in range(generated.shape[0]):
            if len(candidates) >= num_candidates:
                break

            seq = generated[seq_idx]
            new_tokens = seq[input_ids.numel() :]
            if new_tokens.numel() == 0:
                continue

            # Decode iteratively to locate the absolute token boundary mapping to the first '\n'
            line_tokens_list = []
            candidate_line = ""
            for i in range(new_tokens.numel()):
                token_id = new_tokens[i].item()
                line_tokens_list.append(token_id)
                if token_id == eos_token_id:
                    candidate_line = tokenizer.decode(line_tokens_list, skip_special_tokens=True)
                    break
                
                text_so_far = tokenizer.decode(line_tokens_list, skip_special_tokens=True)
                if "\n" in text_so_far:
                    candidate_line = text_so_far.split("\n", 1)[0]
                    break
            else:
                candidate_line = tokenizer.decode(line_tokens_list, skip_special_tokens=True)

            if not candidate_line.strip() and not (line_tokens_list and line_tokens_list[-1] == eos_token_id):
                continue

            if candidate_line in seen_lines:
                duplicates_skipped += 1
                continue

            seen_lines.add(candidate_line)

            line_tokens_tensor = torch.tensor(line_tokens_list, device=beam.token_ids.device)
            new_token_ids = torch.cat([beam.token_ids, line_tokens_tensor])

            candidates.append((candidate_line, new_token_ids))

        if duplicates_skipped > 0:
            logger.debug(f"[CandGen] {duplicates_skipped} duplicate candidates skipped in batch")

        logger.debug(
            f"[CandGen] Batched generation returned {len(candidates)} unique candidates "
            f"in {gen_elapsed:.2f}s"
        )
        return candidates

    def decode(self, token_ids: torch.Tensor) -> TikzDocument:
        """Decode token IDs into a TikzDocument."""
        cache_key = tuple(token_ids.tolist())
        if cache_key in self._decode_cache:
            return self._decode_cache[cache_key]

        tikz = TikzDocument(
            timeout=self.compile_timeout,
            code=self.processor.decode(
                token_ids=token_ids[len(self.initial_input_ids) :],
                skip_special_tokens=True,
            ),
        )
        self._decode_cache[cache_key] = tikz
        return tikz

    def _score_tikz(self, tikz: TikzDocument) -> Numeric:
        """Score a completed TikZ document using the metric or compiler logs."""
        if self.metric and tikz.is_rasterizable and not (
            self.strict and tikz.compiled_with_errors
        ):
            score = self.score_image(tikz.rasterize())
            logger.debug(
                f"[Scoring] Metric score: {score:.4f} (rasterizable, no errors)"
            )
            return score
        elif self.metric:
            logger.debug(
                f"[Scoring] Score: -1 (metric exists but not rasterizable or has errors)"
            )
            return -1
        else:
            scorable = tikz.is_rasterizable and not (
                self.strict and tikz.compiled_with_errors
            )
            score = scorable - tikz.compiled_with_errors
            logger.debug(
                f"[Scoring] Compiler-based score: {score} "
                f"(rasterizable={tikz.is_rasterizable}, "
                f"errors={tikz.compiled_with_errors})"
            )
            return score

    def _final_score(self, tikz: TikzDocument) -> Numeric:
        """Compute final score for a completed beam."""
        return self._score_tikz(tikz)

    def score_image(self, image: Image.Image) -> Numeric:
        """Score a rendered image against the target."""
        assert self.metric
        self.metric.update(img1=image, img2=self.image, text2=self.text)
        score = self.metric.compute()
        self.metric.reset()
        logger.debug(f"[Scoring] Image similarity score: {score:.4f}")
        return score

    def sample(self) -> TikzDocument:
        """Generate a single sample using greedy/sampling (no beam search)."""
        logger.info("[Sample] Generating single sample (no beam search)")
        sample_start = time()

        streamers = StreamerList(filter(bool, [self.streamer]))
        max_length = {
            **self.model.generation_config.to_dict(),
            **self.gen_kwargs,
        }.get("max_length", 4096)

        input_ids = self.initial_input_ids

        if (
            input_ids.numel()
            and input_ids[-1] == unwrap(self.processor).tokenizer.eos_token_id
        ) or input_ids.numel() >= max_length:
            logger.info("[Sample] Already at EOS or max_length, returning as-is")
            streamers.end()
            return self.decode(input_ids)

        with torch.inference_mode():
            token_ids = self.model.generate(
                input_ids=input_ids.unsqueeze(0),
                bad_words_ids=[[self.model.config.image_token_id]],
                begin_suppress_tokens=[
                    self.model.config.text_config.eos_token_id
                ],
                pixel_values=self._cached_proc_inputs.get("pixel_values"),
                image_grid_thw=self._cached_proc_inputs.get("image_grid_thw"),
                streamer=streamers,
                **self._cached_adapter_kwargs,
                **self.gen_kwargs,
            ).squeeze()

        tikz = self.decode(token_ids)
        sample_elapsed = time() - sample_start
        code_lines = len(tikz.code.strip().split("\n")) if tikz.code else 0
        logger.info(
            f"[Sample] Done in {sample_elapsed:.1f}s: "
            f"tokens={token_ids.numel()}, code_lines={code_lines}, "
            f"rasterizable={tikz.is_rasterizable}"
        )
        return tikz


class DetikzifyPipeline:
    """
    Main pipeline for DeTikZify with Verified Beam Search.

    Uses Qwen3-VL as a Process Reward Model to guide line-by-line
    beam search over TikZ code generation.
    """

    def __init__(
        self,
        model,
        processor,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        compile_timeout: Optional[int] = 60,
        metric: Union[Literal["model", "fast"], Metric] = "model",
        num_candidates: int = 5,
        beam_width: int = 3,
        max_lines: int = 50,
        verifier: Optional[Qwen3VLVerifier] = None,
        verifier_model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        **gen_kwargs,
    ):
        self.model = model
        self.processor = processor
        self.num_candidates = num_candidates
        self.beam_width = beam_width
        self.max_lines = max_lines
        self.verifier = verifier
        self.verifier_model_name = verifier_model_name

        if metric == "model":
            self.metric = ImageSim.from_detikzify(
                model, processor, sync_on_compute=False
            )
        elif metric == "fast":
            self.metric = None
        else:
            self.metric = metric

        self.gen_kwargs: Dict[str, Any] = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_length=unwrap(processor).tokenizer.model_max_length,
            do_sample=True,
            compile_timeout=compile_timeout,
            repetition_penalty=1.0,
            **gen_kwargs,
        )

        logger.info(
            f"[Pipeline] Initialized: temperature={temperature}, top_p={top_p}, "
            f"top_k={top_k}, beam_width={beam_width}, "
            f"num_candidates={num_candidates}, max_lines={max_lines}, "
            f"metric={'model' if metric == 'model' else 'fast' if metric == 'fast' else type(metric).__name__}"
        )

    def load(self, image: Union[Image.Image, str], preprocess: bool = True):
        image = load(image)
        if preprocess:
            return expand(image, max(image.size), do_trim=True)
        return image

    def check_inputs(self, image, text):
        assert text is None or has_adapter(
            self.model
        ), "You need to load an adapter for textual inputs!"
        assert image or text, "Either image or text (or both) required!"

    def _get_or_create_verifier(self) -> Qwen3VLVerifier:
        """Lazily initialize the Qwen3-VL verifier."""
        if self.verifier is None:
            logger.info("[Pipeline] Lazily initializing Qwen3-VL verifier...")
            self.verifier = Qwen3VLVerifier(
                model_name=self.verifier_model_name
            )
        return self.verifier

    def sample(
        self,
        image: Optional[Union[Image.Image, str]] = None,
        text: Optional[str] = None,
        preprocess: bool = True,
        **gen_kwargs,
    ) -> TikzDocument:
        """
        DeTikZify a raster image. Samples a single image and returns it
        (no beam search, just greedy/sampling).
        """
        self.check_inputs(image, text)
        logger.info(
            f"[Pipeline] sample() called: image={'yes' if image else 'no'}, "
            f"text={'yes' if text else 'no'}, preprocess={preprocess}"
        )
        generator = DetikzifyGenerator(
            model=self.model,
            processor=self.processor,
            image=self.load(image, preprocess=preprocess)
            if image is not None
            else None,
            text=text,
            verifier=self._get_or_create_verifier(),
            num_candidates=self.num_candidates,
            beam_width=self.beam_width,
            max_lines=self.max_lines,
            **self.gen_kwargs,
            **gen_kwargs,
        )
        return generator.sample()

    def simulate(
        self,
        image: Optional[Union[Image.Image, str]] = None,
        text: Optional[str] = None,
        preprocess: bool = True,
        expansions: Optional[Numeric] = None,
        timeout: Optional[int] = None,
        **gen_kwargs,
    ) -> Generator[Tuple[Numeric, TikzDocument], None, None]:
        """
        DeTikZify a raster image using Verified Beam Search with Qwen3-VL PRM.
        Returns an iterator yielding (score, tikzdoc) tuples.

        Args:
            image: the target image
            text: textual instruction
            preprocess: whether to preprocess the image
            expansions: number of beam search rounds (None for infinite)
            timeout: timeout in seconds (None for infinite)
            gen_kwargs: additional generation kwargs
        """
        self.check_inputs(image, text)
        logger.info(
            f"[Pipeline] simulate() called: image={'yes' if image else 'no'}, "
            f"text={'yes' if text else 'no'}, preprocess={preprocess}, "
            f"expansions={expansions}, timeout={timeout}"
        )
        generator = DetikzifyGenerator(
            model=self.model,
            processor=self.processor,
            metric=self.metric,
            beam_timeout=timeout or None,
            image=self.load(image, preprocess=preprocess)
            if image is not None
            else None,
            text=text,
            verifier=self._get_or_create_verifier(),
            num_candidates=self.num_candidates,
            beam_width=self.beam_width,
            max_lines=self.max_lines,
            **self.gen_kwargs,
            **gen_kwargs,
        )

        yield from generator.simulate(expansions or None)

    def __call__(self, *args, **kwargs) -> TikzDocument:
        return self.sample(*args, **kwargs)
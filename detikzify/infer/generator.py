from time import time
from typing import Generator, List, Optional, Set, Tuple, Union

import torch
from PIL import Image
from torchmetrics import Metric
from transformers.generation.streamers import BaseStreamer

from ..util import (
    ExplicitAbort,
    StreamerList,
    unwrap_processor as unwrap,
)
from .beam import BeamCandidate
from .logger import logger
from .tikz import TikzDocument
from .verifier import Qwen3VLVerifier

Numeric = Union[int, float]

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
        num_candidates: int = 3,
        beam_width: int = 3,
        max_lines: int = 100,
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
                logger.save()
                return

        total_elapsed = time() - start_time
        logger.info(
            f"[Simulate] Finished: {iteration} rounds, {total_results} results, "
            f"{total_elapsed:.1f}s total"
        )
        self.verifier.log_stats()
        logger.save()

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
                n_scored = 0
                score_elapsed = 0.0
                if self.image is not None:
                    score_start = time()
                    partial_codes = ["\n".join(info["partial_lines"]) for info in new_candidates_info]
                    last_lines = [info["cand_line"] for info in new_candidates_info]
                    
                    verifier_scores = self.verifier.batch_score(self.image, partial_codes, last_lines)
                    score_elapsed = time() - score_start
                    n_scored = len(partial_codes)
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

                if n_scored > 0 or self.image is not None:
                    logger.debug(f"[BeamSearch]   Scored {n_scored} unique candidates in {score_elapsed:.2f}s")

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

        # Finished search. Return all beams as result TikzDocuments
        for score, beam in [(b.score, b) for b in beams if b.finished] + [
            (b.score, b) for b in beams if not b.finished
        ]:
            yield (score, self.decode(beam.token_ids))

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
        }.get("max_length", 512)

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
        }.get("max_length", 512)

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

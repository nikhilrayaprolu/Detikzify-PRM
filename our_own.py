class Qwen3VLVerifier:
    """
    Zero-shot Process Reward Model using Qwen3-VL.
    Only scores lines that are inside the tikzpicture environment.
    Preamble/boilerplate lines get a neutral pass-through score.
    """

    # Changed: Two-tier prompt system
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
Respond with ONLY a decimal number from 0.0 to 1.0, nothing else.

Score:"""

    STRUCTURAL_PROMPT = """You are a TikZ code verifier. Given a target image and a partial TikZ program that is still in the preamble/setup phase, assess whether the document setup is reasonable for reproducing the target image.

Partial TikZ Code:
```latex
{code}
```

Is this setup reasonable? Respond with ONLY a decimal number from 0.0 to 1.0.

Score:"""

    # Lines that are boilerplate and shouldn't be scored against the image
    BOILERPLATE_PATTERNS = [
        r'^\s*\\documentclass',
        r'^\s*\\usepackage',
        r'^\s*\\usetikzlibrary',
        r'^\s*\\begin\{document\}',
        r'^\s*\\end\{document\}',
        r'^\s*\\begin\{tikzpicture\}',
        r'^\s*\\tikzset',
        r'^\s*\\pgfmathsetmacro',
        r'^\s*\\definecolor',
        r'^\s*\\newcommand',
        r'^\s*%',            # comments
        r'^\s*$',            # blank lines
    ]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: Optional[str] = None,
        torch_dtype=torch.bfloat16,
        boilerplate_score: float = 0.5,  # neutral score for preamble lines
    ):
        logger.info(f"[Verifier] Loading Qwen3-VL verifier: {model_name}")
        load_start = time()
        self.processor = AutoProcessor.from_pretrained(model_name)
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
        """Check if a line is boilerplate that shouldn't be scored."""
        for pattern in self.BOILERPLATE_PATTERNS:
            if re.match(pattern, line):
                return True
        return False

    def _is_inside_tikzpicture(self, lines: List[str]) -> bool:
        """Check if we're currently inside a tikzpicture environment."""
        depth = 0
        for line in lines:
            if r'\begin{tikzpicture}' in line:
                depth += 1
            if r'\end{tikzpicture}' in line:
                depth -= 1
        return depth > 0

    def _count_drawing_lines(self, lines: List[str]) -> int:
        """Count lines that are actual drawing commands."""
        count = 0
        inside = False
        for line in lines:
            if r'\begin{tikzpicture}' in line:
                inside = True
                continue
            if r'\end{tikzpicture}' in line:
                inside = False
                continue
            if inside and not self._is_boilerplate(line) and line.strip():
                count += 1
        return count

    @torch.inference_mode()
    def score(self, image: Image.Image, partial_code: str, last_line: str = "") -> float:
        """
        Score partial TikZ code against the target image.

        Key change: boilerplate/preamble lines get a neutral score (0.5)
        without calling the model. Only actual drawing commands are scored.
        """
        self._call_count += 1
        call_start = time()

        # Determine the last line if not provided
        if not last_line:
            code_lines = partial_code.strip().split("\n")
            last_line = code_lines[-1] if code_lines else ""

        # Skip scoring for boilerplate lines
        if self._is_boilerplate(last_line):
            self._skipped_count += 1
            elapsed = time() - call_start

            line_preview = last_line.strip()[:80]
            logger.info(
                f"[Verifier] call #{self._call_count}: score={self.boilerplate_score:.4f} "
                f"(boilerplate skip) elapsed={elapsed:.4f}s "
                f"last_line=\"{line_preview}\""
            )
            return self.boilerplate_score

        # Determine which prompt to use based on context
        lines = partial_code.strip().split("\n")
        inside_tikz = self._is_inside_tikzpicture(lines)
        drawing_lines = self._count_drawing_lines(lines)

        if inside_tikz and drawing_lines >= 1:
            prompt = self.DRAWING_PROMPT.format(code=partial_code)
        else:
            prompt = self.STRUCTURAL_PROMPT.format(code=partial_code)

        # Build the chat-style input for Qwen3-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=16,  # Reduced: we only need a number
            do_sample=False,
            temperature=1.0,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        parsed_score = self._parse_score(response)
        elapsed = time() - call_start
        self._total_time += elapsed

        line_preview = last_line.strip()[:100]
        logger.info(
            f"[Verifier] call #{self._call_count}: score={parsed_score:.4f} "
            f"response=\"{response[:60]}\" elapsed={elapsed:.2f}s "
            f"inside_tikz={inside_tikz} drawing_lines={drawing_lines} "
            f"last_line=\"{line_preview}\""
        )

        return parsed_score

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
        # Try to find a decimal number first (most reliable)
        # Look at the very start of the response for cleaner extraction
        clean = response.strip().split('\n')[0].strip()

        patterns = [
            r'^(\d+\.\d+)',    # starts with decimal like 0.75
            r'(\d+\.\d+)',     # decimal anywhere
            r'(\d+)/(\d+)',    # fraction like 7/10
            r'(\d+)%',         # percentage like 75%
            r'^(\d+)$',        # just an integer
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
            f"[Verifier] Could not parse score from: \"{clean}\", defaulting to 0.5"
        )
        return 0.5
```

---

## Updated Beam Search Scoring Logic

The cumulative scoring also needs to change. Replace the averaging approach with one that **only accumulates scores from meaningful lines**:

```python
def _beam_search(self) -> Generator[Tuple[Numeric, TikzDocument], None, None]:
    """Perform one round of Verified Beam Search."""
    search_start = time()

    beams: List[BeamCandidate] = [
        BeamCandidate(
            token_ids=self.initial_input_ids.clone(),
            lines=[],
            score=0.0,
            finished=False,
            scored_steps=0,  # NEW: track how many steps were actually scored
        )
    ]
    # ... rest of search setup ...

    for step in range(self.max_lines):
        # ... existing beam iteration code ...

        for beam_idx, beam in enumerate(beams):
            if beam.finished:
                all_candidates.append(beam)
                continue

            candidates = self._generate_candidate_lines(beam, self.num_candidates)

            for cand_idx, (candidate_line, candidate_token_ids) in enumerate(candidates):
                partial_lines = beam.lines + [candidate_line]
                partial_code = "\n".join(partial_lines)

                if self.image is not None:
                    verifier_score = self.verifier.score(
                        self.image, partial_code, last_line=candidate_line
                    )
                else:
                    verifier_score = 0.5

                is_finished = "\\end{tikzpicture}" in candidate_line

                # NEW: Only count non-boilerplate scores in the average
                is_boilerplate = self.verifier._is_boilerplate(candidate_line)

                if is_boilerplate:
                    # Carry forward the existing score unchanged
                    cumulative_score = beam.score
                    new_scored_steps = beam.scored_steps
                else:
                    # Running average of meaningful scores only
                    new_scored_steps = beam.scored_steps + 1
                    cumulative_score = (
                        beam.score * beam.scored_steps + verifier_score
                    ) / new_scored_steps

                all_candidates.append(
                    BeamCandidate(
                        token_ids=candidate_token_ids,
                        lines=partial_lines,
                        score=cumulative_score,
                        finished=is_finished,
                        scored_steps=new_scored_steps,
                    )
                )
```

Update the dataclass:

```python
@dataclass
class BeamCandidate:
    """Represents a single beam in the verified beam search."""
    token_ids: torch.Tensor
    lines: List[str]
    score: float = 0.0
    finished: bool = False
    scored_steps: int = 0  # NEW: number of non-boilerplate steps scored

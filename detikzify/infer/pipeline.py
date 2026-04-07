from typing import Any, Dict, Generator, Literal, Optional, Tuple, Union

import torch
from PIL import Image
from torchmetrics import Metric

from ..evaluate.imagesim import ImageSim
from ..model.adapter import has_adapter
from ..util import (
    expand,
    load,
    unwrap_processor as unwrap,
)
from .generator import DetikzifyGenerator, Numeric
from .logger import logger
from .tikz import TikzDocument
from .verifier import FinetunedVerifier, Qwen3VLVerifier

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
        num_candidates: int = 3,
        beam_width: int = 3,
        max_lines: int = 50,
        verifier: Optional[Union[Qwen3VLVerifier, FinetunedVerifier]] = None,
        verifier_model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        verifier_checkpoint: Optional[str] = None,
        **gen_kwargs,
    ):
        self.model = model
        self.processor = processor
        self.num_candidates = num_candidates
        self.beam_width = beam_width
        self.max_lines = max_lines
        self.verifier = verifier
        self.verifier_model_name = verifier_model_name
        self.verifier_checkpoint = verifier_checkpoint

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

    def _get_or_create_verifier(self) -> Union[Qwen3VLVerifier, FinetunedVerifier]:
        """Lazily initialize the verifier."""
        if self.verifier is None:
            if self.verifier_checkpoint:
                logger.info("[Pipeline] Lazily initializing Finetuned verifier...")
                self.verifier = FinetunedVerifier(
                    checkpoint_path=self.verifier_checkpoint,
                    base_model_name=self.verifier_model_name,
                    load_in_4bit=True,
                )
            else:
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

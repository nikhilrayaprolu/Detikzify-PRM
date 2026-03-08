#!/usr/bin/env python
from argparse import ArgumentParser

from PIL import UnidentifiedImageError
from torch import bfloat16, float16
from torch.cuda import is_available as is_cuda_available, is_bf16_supported
from transformers import TextStreamer, set_seed
from transformers.utils import is_flash_attn_2_available

from detikzify.infer import DetikzifyPipeline
from detikzify.model import load


def parse_args():
    p = ArgumentParser(description="Inference helper for fine-tuned models.")
    p.add_argument(
        "--model_name_or_path",
        required=True,
        help="model checkpoint for weights initialization (local or hub)",
    )
    p.add_argument(
        "--image",
        required=True,
        type=str,
        help="path or URL to the input image to detikzify",
    )
    p.add_argument(
        "--mode",
        choices=["greedy", "simulate"],
        default="greedy",
        help="'greedy' for single sample, 'simulate' for MCTS (default: greedy)",
    )
    p.add_argument(
        "--expansions",
        type=int,
        default=10,
        help="number of MCTS expansions in simulate mode (default: 10)",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="MCTS timeout in seconds (default: None)",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="compile and report diagnostics",
    )
    p.add_argument(
        "--save-image",
        type=str,
        default=None,
        metavar="PATH",
        help="save rasterized output (e.g., output.png)",
    )
    p.add_argument(
        "--save-tex",
        type=str,
        default=None,
        metavar="PATH",
        help="save TikZ code (e.g., output.tex)",
    )
    p.add_argument(
        "--save-pdf",
        type=str,
        default=None,
        metavar="PATH",
        help="save compiled PDF (e.g., output.pdf)",
    )
    p.add_argument(
        "--metric",
        choices=["model", "fast"],
        default="model",
        help="scoring metric for simulate mode (default: model)",
    )
    p.add_argument(
        "--compile-timeout",
        type=int,
        default=60,
        help="LaTeX compilation timeout in seconds (default: 60)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed (default: 0)",
    )
    return p.parse_args()


def handle_output(tikzdoc, args):
    """Handle compilation reporting and file saving."""
    needs_compile = args.compile or args.save_image or args.save_pdf

    if needs_compile:
        print("\n--- Compilation ---")
        print(f"  Status : {tikzdoc.status}")
        print(f"  PDF    : {'available' if tikzdoc.pdf is not None else 'NOT available'}")

        if tikzdoc.compiled_with_errors:
            print("  Result : ⚠ compiled with errors")
            for line, msg in tikzdoc.errors.items():
                print(f"    Line {line}: {msg}")
        else:
            print("  Result : ✓ compiled successfully")

        if tikzdoc.pdf is not None:
            if tikzdoc.is_rasterizable:
                print("  Raster : ✓ rasterizable")
            else:
                print("  Raster : ✗ PDF exists but rasterization failed")
        else:
            print("  Raster : ✗ no PDF available")
            if tikzdoc.log:
                print("\n  [Last 500 chars of log]")
                for ln in tikzdoc.log[-500:].splitlines():
                    print(f"    {ln}")

    # ── save files ───────────────────────────────────────────────────────
    if args.save_tex:
        try:
            tikzdoc.save(args.save_tex)
            print(f"\n✓ TikZ code saved to {args.save_tex}")
        except Exception as e:
            print(f"\n✗ Failed to save .tex: {e}")

    if args.save_pdf:
        if tikzdoc.pdf is not None:
            try:
                tikzdoc.save(args.save_pdf)
                print(f"✓ PDF saved to {args.save_pdf}")
            except Exception as e:
                print(f"✗ Failed to save PDF: {e}")
        else:
            print("✗ Cannot save PDF: no PDF returned by compilation server")

    if args.save_image:
        if tikzdoc.is_rasterizable:
            try:
                tikzdoc.save(args.save_image)
                print(f"✓ Image saved to {args.save_image}")
            except Exception as e:
                print(f"✗ Failed to save image: {e}")
        else:
            reason = (
                "no PDF available"
                if tikzdoc.pdf is None
                else "rasterization failed"
            )
            print(f"✗ Cannot save image: {reason}")


def run_greedy(pipe, image, args):
    """Single-sample generation."""
    tikzdoc = pipe.sample(image=image)
    handle_output(tikzdoc, args)
    return tikzdoc


def run_simulate(pipe, image, args):
    """MCTS-based generation."""
    print(
        f"\n🔍 MCTS simulation  "
        f"(expansions={args.expansions}, timeout={args.timeout}, "
        f"metric={args.metric})"
    )

    best_score, best_tikz = float("-inf"), None

    for i, (score, tikzdoc) in enumerate(
        pipe.simulate(
            image=image,
            expansions=args.expansions,
            timeout=args.timeout,
        ),
        start=1,
    ):
        flag = "✓" if tikzdoc.is_rasterizable else "✗"
        pdf_ok = "pdf" if tikzdoc.pdf is not None else "no-pdf"
        print(f"  Rollout {i}: score={score:+.4f}  {flag}  ({pdf_ok})")
        if score > best_score:
            best_score, best_tikz = score, tikzdoc

    if best_tikz is not None:
        print(f"\n🏆 Best score: {best_score:+.4f}")
        print("--- Best TikZ Code ---")
        print(best_tikz.code)
        print("--- End ---")
        handle_output(best_tikz, args)
    else:
        print("No rollouts produced.")

    return best_tikz


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    print(f"Model : {args.model_name_or_path}")
    print(f"Image : {args.image}")
    print(f"Mode  : {args.mode}")

    model, processor = load(
        model_name_or_path=args.model_name_or_path,
        device_map="auto",
        torch_dtype=(
            bfloat16
            if is_cuda_available() and is_bf16_supported()
            else float16
        ),
        attn_implementation=(
            "flash_attention_2" if is_flash_attn_2_available() else None
        ),
    )

    streamer = (
        TextStreamer(
            tokenizer=processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        if args.mode == "greedy"
        else None
    )

    pipe = DetikzifyPipeline(
        model=model,
        processor=processor,
        streamer=streamer,
        compile_timeout=args.compile_timeout,
        metric=args.metric,
    )

    try:
        if args.mode == "simulate":
            run_simulate(pipe, args.image, args)
        else:
            run_greedy(pipe, args.image, args)
    except (
        UnidentifiedImageError,
        FileNotFoundError,
        AttributeError,
        ValueError,
    ) as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
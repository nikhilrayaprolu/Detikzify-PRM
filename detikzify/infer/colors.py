"""
ANSI colour constants and helper functions.
Imported by every other visualizer module.
"""


class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    BLACK   = "\033[30m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

    BG_BLACK   = "\033[40m"
    BG_RED     = "\033[41m"
    BG_GREEN   = "\033[42m"
    BG_YELLOW  = "\033[43m"
    BG_BLUE    = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN    = "\033[46m"
    BG_WHITE   = "\033[47m"

    BRIGHT_RED     = "\033[91m"
    BRIGHT_GREEN   = "\033[92m"
    BRIGHT_YELLOW  = "\033[93m"
    BRIGHT_BLUE    = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN    = "\033[96m"
    BRIGHT_WHITE   = "\033[97m"


def colored(text: str, *codes: str) -> str:
    """Wrap text in ANSI codes, always reset at end."""
    return "".join(codes) + str(text) + C.RESET


def score_color(score: float) -> str:
    """Return an ANSI color code based on a 0-1 score."""
    if score >= 0.9:
        return C.BRIGHT_GREEN
    elif score >= 0.7:
        return C.GREEN
    elif score >= 0.5:
        return C.YELLOW
    elif score >= 0.3:
        return C.BRIGHT_YELLOW
    else:
        return C.BRIGHT_RED


def score_bar(score: float, width: int = 10) -> str:
    """Render a compact colored progress bar for a 0-1 score."""
    filled = int(round(score * width))
    empty  = width - filled
    color  = score_color(score)
    bar    = colored("█" * filled, color) + colored("░" * empty, C.DIM)
    pct    = colored(f"{score:.2f}", color, C.BOLD)
    return f"[{bar}] {pct}"
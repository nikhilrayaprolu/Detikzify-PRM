from dataclasses import dataclass
from typing import List
import torch

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

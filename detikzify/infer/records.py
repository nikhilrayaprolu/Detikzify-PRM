"""
Dataclasses shared across visualizer and graph-builder modules.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StepRecord:
    step:            int
    active:          int
    finished:        int
    elapsed_s:       float
    beam_scores:     List[float]
    candidates_seen: int   = 0
    candidates_kept: int   = 0
    gen_time_s:      float = 0.0
    score_time_s:    float = 0.0


@dataclass
class BeamRecord:
    beam_idx:     int
    score:        float
    lines:        int
    tokens:       int
    last_line:    str
    finished:     bool
    selected:     bool
    scored_steps: int


@dataclass
class CandidateRecord:
    beam_idx:     int
    verifier:     float
    cumulative:   float
    scored_steps: int
    line:         str
    selected:     bool        = False
    rank:         Optional[int] = None


@dataclass
class GraphNode:
    """One node in the beam search DAG."""
    node_id:      str
    step:         int
    beam_idx:     int
    score:        float
    verifier:     float
    cumulative:   float
    lines:        int
    tokens:       int
    scored_steps: int
    last_line:    str
    finished:     bool
    selected:     bool
    rank:         Optional[int] = None
    label:        str           = ""

    def __post_init__(self):
        if not self.label:
            short = self.last_line[:40] + (
                "…" if len(self.last_line) > 40 else ""
            )
            self.label = f"[{self.step}:{self.beam_idx}] {short}"


@dataclass
class GraphEdge:
    """Directed edge from parent to child node."""
    edge_id:        str
    source:         str
    target:         str
    verifier_score: float
    label:          str = ""
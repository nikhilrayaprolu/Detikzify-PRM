"""
Rich terminal visualizer for verified beam search.
Buffers all messages within a step and renders them together.
"""
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .colors  import C, colored, score_color, score_bar
from .records import BeamRecord, CandidateRecord, StepRecord


class BeamSearchVisualizer:
    """
    Pretty-prints beam search progress to the terminal.
    All messages within a step are buffered and rendered together
    so candidates are always shown in context with their parent beam.
    """

    BEAM_COLORS  = [C.BRIGHT_CYAN, C.BRIGHT_MAGENTA, C.BRIGHT_YELLOW]
    BEAM_SYMBOLS = ["◆", "●", "▲"]

    def __init__(
        self,
        beam_width:     int            = 3,
        terminal_width: Optional[int]  = None,
    ):
        self.beam_width = beam_width
        self.tw = terminal_width or shutil.get_terminal_size((120, 40)).columns
        self.tw = max(self.tw, 80)

        # Score history for sparklines
        self._score_history: Dict[int, List[float]] = defaultdict(list)
        self._history:       List[StepRecord]        = []

        # ── per-step buffers (reset on each step header) ──────────────────
        self._step:           int                    = 0
        self._total_steps:    int                    = 0
        self._active:         int                    = 0
        self._finished_count: int                    = 0

        self._beam_states:  List[BeamRecord]         = []
        self._candidates:   List[CandidateRecord]    = []
        self._ranked:       List[Tuple[int, str, bool]] = []
        self._gen_batches:  List[Tuple[int, float]]  = []
        self._dup_skipped:  int                      = 0
        self._score_elapsed: float                   = 0.0
        self._scored_n:     int                      = 0

        self._suppress_patterns = [
            "max_new_tokens", "max_length",
            "A decoder-only architecture", "padding_side",
            "generation flags are not valid", "TRANSFORMERS_VERBOSITY",
        ]

    # ── public logger API ─────────────────────────────────────────────────

    def info(self, msg: str)    -> None: self._dispatch(msg, "INFO")
    def debug(self, msg: str)   -> None: self._dispatch(msg, "DEBUG")
    def warning(self, msg: str) -> None: self._dispatch(msg, "WARN")
    def error(self, msg: str)   -> None: self._dispatch(msg, "ERROR")

    # ── router ────────────────────────────────────────────────────────────

    def _dispatch(self, msg: str, level: str) -> None:
        if any(p in msg for p in self._suppress_patterns):
            return
        tag, body = self._split_tag(msg)
        routes = {
            "[BeamSearch]": self._handle_beam_search,
            "[CandGen]":    self._handle_cand_gen,
            "[Verifier]":   self._handle_verifier,
            "[Scoring]":    self._handle_scoring,
            "[Simulate]":   self._handle_simulate,
            "[Generator]":  self._handle_generator,
            "[Pipeline]":   self._handle_pipeline,
            "[Sample]":     self._handle_sample,
        }
        handler = routes.get(tag)
        if handler:
            handler(body, level)
        else:
            self._plain(msg, level)

    # ── per-tag handlers ──────────────────────────────────────────────────

    def _handle_beam_search(self, body: str, level: str) -> None:

        # step header → reset buffers
        m = re.match(
            r"── Step (\d+)/(\d+) ── active=(\d+), finished=(\d+)", body
        )
        if m:
            step, total, active, finished = (int(x) for x in m.groups())
            self._step           = step
            self._total_steps    = total
            self._active         = active
            self._finished_count = finished
            self._beam_states    = []
            self._candidates     = []
            self._ranked         = []
            self._gen_batches    = []
            self._dup_skipped    = 0
            self._score_elapsed  = 0.0
            self._scored_n       = 0
            return

        # current beam state
        m = re.match(
            r"Beam (\d+): (\[.*?\]) score=([\d.]+) lines=(\d+) "
            r"scored_steps=(\d+) tokens=(\d+) last_line=\"(.*)\"",
            body,
        )
        if m:
            idx, status, score, lines, ss, tokens, last = m.groups()
            self._beam_states.append(BeamRecord(
                beam_idx     = int(idx),
                score        = float(score),
                lines        = int(lines),
                tokens       = int(tokens),
                scored_steps = int(ss),
                last_line    = last,
                finished     = "FINISHED" in status,
                selected     = True,
            ))
            total_beams = self._active + self._finished_count
            if len(self._beam_states) == total_beams:
                self._render_step_header()
                self._render_beam_state()
            return

        # scored candidate
        m = re.match(
            r"Beam (\d+) -> verifier=([\d.]+) cumulative=([\d.]+) "
            r"scored_steps=(\d+) line=\"(.*)\"",
            body,
        )
        if m:
            bidx, ver, cum, ss, line = m.groups()
            self._candidates.append(CandidateRecord(
                beam_idx     = int(bidx),
                verifier     = float(ver),
                cumulative   = float(cum),
                scored_steps = int(ss),
                line         = line,
            ))
            return

        # scored N unique candidates
        m = re.match(r"Scored (\d+) unique candidates in ([\d.]+)s", body)
        if m:
            self._scored_n      = int(m.group(1))
            self._score_elapsed = float(m.group(2))
            self._render_candidates_table()
            return

        # ranking header — handled implicitly
        m = re.match(
            r"Ranking (\d+) total candidates \(keeping top (\d+)\)", body
        )
        if m:
            return

        # ranked entry
        m = re.match(r"([✓✗]) Rank (\d+): (.*)", body)
        if m:
            tick, rank, summary = m.group(1), int(m.group(2)), m.group(3)
            self._ranked.append((rank, summary, tick == "✓"))
            return

        # step footer
        m = re.match(
            r"Step (\d+) done in ([\d.]+)s\. Retained beam scores: \[(.*)\]",
            body,
        )
        if m:
            step, elapsed, scores_str = m.groups()
            scores = [float(s) for s in scores_str.split(", ") if s.strip()]
            self._render_ranking_table()
            self._render_step_footer(int(step), float(elapsed), scores)
            return

        # all beams finished
        m = re.match(r"All (\d+) beams finished at step (\d+)", body)
        if m:
            self._render_all_finished(int(m.group(1)), int(m.group(2)))
            return

        # search complete
        m = re.match(
            r"Search complete in ([\d.]+)s\. Decoding (\d+) beams\.\.\.", body
        )
        if m:
            self._render_search_complete(
                float(m.group(1)), int(m.group(2))
            )
            return

        # final beam result
        m = re.match(
            r"Final beam (\d+): beam_score=([\d.]+), final_score=([\d.]+), "
            r"scored_steps=(\d+), rasterizable=(\w+), "
            r"code_lines=(\d+), decode_time=([\d.]+)s",
            body,
        )
        if m:
            bidx, bs, fs, ss, rast, cl, dt = m.groups()
            self._render_final_beam(
                int(bidx), float(bs), float(fs),
                int(ss), rast == "True", int(cl), float(dt),
            )
            return

        # suppress raw code preview dumps
        if "code preview" in body:
            return

    def _handle_cand_gen(self, body: str, level: str) -> None:
        m = re.match(r"(\d+) duplicate candidates skipped in batch", body)
        if m:
            self._dup_skipped += int(m.group(1))
            return

        m = re.match(
            r"Batched generation returned (\d+) unique candidates in ([\d.]+)s",
            body,
        )
        if m:
            self._gen_batches.append((int(m.group(1)), float(m.group(2))))
            return

        self._plain(f"[CandGen] {body}")

    def _handle_verifier(self, body: str, level: str) -> None:
        m = re.match(
            r"Stats: (\d+) calls \((\d+) skipped as boilerplate\), "
            r"total=([\d.]+)s, avg=([\d.]+)s/call",
            body,
        )
        if m:
            calls, skipped, total, avg = m.groups()
            self._render_verifier_stats(
                int(calls), int(skipped), float(total), float(avg)
            )
            return
        icon = colored("🔍", C.BRIGHT_CYAN)
        print(f"  {icon}  {colored('[Verifier]', C.CYAN)} {body}")

    def _handle_scoring(self, body: str, level: str) -> None:
        m = re.match(r"(Image similarity|Metric) score: ([\d.]+)(.*)", body)
        if m:
            kind, score, extra = m.groups()
            icon = "🖼 " if "Image" in kind else "📊"
            bar  = score_bar(float(score), width=12)
            print(
                f"  {icon}  {colored(kind + ' score:', C.DIM)} "
                f"{bar}{colored(extra, C.DIM)}"
            )
            return
        self._plain(f"[Scoring] {body}")

    def _handle_simulate(self, body: str, level: str) -> None:
        m = re.match(
            r"Result #(\d+): score=([\d.]+), rasterizable=(\w+), "
            r"errors=(\w+), code_lines=(\d+)",
            body,
        )
        if m:
            rank, score, rast, err, cl = m.groups()
            rast_icon = "✅" if rast == "True" else "❌"
            err_icon  = "⚠ " if err  == "True" else "  "
            bar       = score_bar(float(score), width=14)
            print(
                f"  {rast_icon} {err_icon} Result "
                f"#{colored(rank, C.BOLD)}: {bar}  "
                f"{colored(cl + ' lines', C.DIM)}"
            )
            return
        m = re.match(
            r"(Round \d+ completed|Finished|Starting simulation)(.*)", body
        )
        if m:
            self._section(f"[Simulate] {body}", icon="🔄")
            return
        self._plain(f"[Simulate] {body}")

    def _handle_generator(self, body: str, level: str) -> None:
        if level == "INFO":
            self._section(
                f"[Generator] {body}", icon="⚙ ", color=C.BRIGHT_BLUE
            )

    def _handle_pipeline(self, body: str, level: str) -> None:
        if level == "INFO":
            self._section(
                f"[Pipeline] {body}", icon="🚀", color=C.BRIGHT_MAGENTA
            )

    def _handle_sample(self, body: str, level: str) -> None:
        self._plain(f"[Sample] {body}")

    # ── rendering primitives ──────────────────────────────────────────────

    def _hline(self, char: str = "─", color: str = C.DIM) -> None:
        print(colored(char * self.tw, color))

    def _section(
        self,
        text:  str,
        icon:  str = "▸",
        color: str = C.BRIGHT_WHITE,
    ) -> None:
        print(colored(f"\n{icon}  {text}", color))

    def _plain(self, msg: str, level: str = "INFO") -> None:
        lvl_color = {
            "INFO":  C.BRIGHT_WHITE,
            "DEBUG": C.DIM,
            "WARN":  C.BRIGHT_YELLOW,
            "ERROR": C.BRIGHT_RED,
        }.get(level, C.RESET)
        print(colored(f"  {msg}", lvl_color))

    @staticmethod
    def _split_tag(msg: str) -> Tuple[str, str]:
        m = re.match(r"(\[[A-Za-z0-9_]+\])\s*(.*)", msg, re.DOTALL)
        return (m.group(1), m.group(2).strip()) if m else ("", msg)

    def _beam_color(self, idx: int) -> str:
        return self.BEAM_COLORS[idx % len(self.BEAM_COLORS)]

    def _beam_symbol(self, idx: int) -> str:
        return self.BEAM_SYMBOLS[idx % len(self.BEAM_SYMBOLS)]

    # ── step header ───────────────────────────────────────────────────────

    def _render_step_header(self) -> None:
        step, total = self._step, self._total_steps
        print()
        self._hline("═", C.BLUE)

        progress_width = 30
        filled = int((step - 1) / total * progress_width) if total > 1 else 0
        bar = (
            colored("█" * filled, C.BRIGHT_BLUE)
            + colored("░" * (progress_width - filled), C.DIM)
        )
        pct      = f"{(step - 1) / total * 100:.0f}%"
        title    = colored(
            f" STEP {step}/{total} ",
            C.BOLD + C.BG_BLUE + C.BRIGHT_WHITE,
        )
        active_s = colored(f"{self._active} active", C.BRIGHT_GREEN)
        fin_s    = colored(
            f"{self._finished_count} done",
            C.BRIGHT_CYAN if self._finished_count else C.DIM,
        )
        prog_s = f"[{bar}] {colored(pct, C.BLUE)}"
        print(f"  {title}  {active_s}  {fin_s}    {prog_s}")
        self._hline("─", C.DIM)

    # ── beam state table ──────────────────────────────────────────────────

    def _render_beam_state(self) -> None:
        beams    = self._beam_states
        n        = len(beams)
        col_w    = max(24, (self.tw - 4) // max(n, 1))
        max_line = col_w - 4

        # header row
        headers = []
        for b in beams:
            sym   = self._beam_symbol(b.beam_idx)
            col   = self._beam_color(b.beam_idx)
            label = colored(f"{sym} Beam {b.beam_idx}", col, C.BOLD)
            if b.finished:
                label += colored(" [DONE]", C.BRIGHT_GREEN)
            headers.append(label)
        print("  " + "   ".join(h.ljust(col_w + 12) for h in headers))
        print("  " + colored("   ".join(["─" * col_w] * n), C.DIM))

        # score bars
        cells = []
        for b in beams:
            bar = score_bar(b.score, width=col_w - 12)
            cells.append(f"Score {bar}")
        print("  " + "   ".join(cells))

        # metadata rows
        for label, getter in [
            ("Lines  ", lambda b: str(b.lines)),
            ("Tokens ", lambda b: str(b.tokens)),
            ("Steps  ", lambda b: str(b.scored_steps)),
        ]:
            row = []
            for b in beams:
                row.append(colored(f"{label}: {getter(b)}", C.DIM))
            print("  " + "   ".join(c.ljust(col_w + 8) for c in row))

        # last generated line
        print("  " + colored("   ".join(["─" * col_w] * n), C.DIM))
        for b in beams:
            col  = self._beam_color(b.beam_idx)
            line = b.last_line[:max_line] + (
                "…" if len(b.last_line) > max_line else ""
            )
            print(f"  {colored(line, col)}")

    # ── candidate table ───────────────────────────────────────────────────

    def _render_candidates_table(self) -> None:
        total_unique = sum(n for n, _ in self._gen_batches)
        total_gen_t  = sum(t for _, t in self._gen_batches)
        n_batches    = len(self._gen_batches)

        print()
        self._hline("─", C.DIM)

        # generation / scoring summary line
        print(
            f"  {colored('⚡ Generation', C.BRIGHT_CYAN, C.BOLD)}  "
            f"{colored(str(n_batches), C.BOLD)} batch(es)  "
            f"{colored(str(total_unique), C.BRIGHT_WHITE)} unique  "
            f"{colored(str(self._dup_skipped), C.DIM)} dupes skipped  "
            f"⏱ {colored(f'{total_gen_t:.2f}s', C.DIM)}"
        )
        print(
            f"  {colored('🔍 Scoring', C.BRIGHT_CYAN, C.BOLD)}    "
            f"{colored(str(self._scored_n), C.BRIGHT_WHITE)} candidates  "
            f"⏱ {colored(f'{self._score_elapsed:.2f}s', C.DIM)}"
        )
        print()

        # group candidates by beam
        by_beam: Dict[int, List[CandidateRecord]] = defaultdict(list)
        for c in self._candidates:
            by_beam[c.beam_idx].append(c)

        max_line = max(20, self.tw - 52)

        for beam in self._beam_states:
            bid   = beam.beam_idx
            sym   = self._beam_symbol(bid)
            col   = self._beam_color(bid)
            cands = by_beam.get(bid, [])

            print(
                f"  {colored(sym, col, C.BOLD)} "
                f"{colored(f'Beam {bid}', col, C.BOLD)}  "
                f"{colored(f'→ {len(cands)} candidate(s)', C.DIM)}"
            )

            if not cands:
                print(
                    f"    {colored('(boilerplate — no model call)', C.DIM)}"
                )
                print()
                continue

            # column headers
            print(
                f"    {colored('Verifier', C.DIM):>10}  "
                f"{colored('Cumul', C.DIM):>8}  "
                f"{colored('Steps', C.DIM):>5}  "
                f"{colored('Generated line', C.DIM)}"
            )
            print(f"    {colored('─' * (self.tw - 8), C.DIM)}")

            for c in cands:
                ver_bar = score_bar(c.verifier, width=6)
                cum_col = score_color(c.cumulative)
                cum_s   = colored(f"{c.cumulative:.4f}", cum_col)
                steps_s = colored(str(c.scored_steps), C.DIM)
                line    = c.line[:max_line] + (
                    "…" if len(c.line) > max_line else ""
                )
                print(
                    f"    {ver_bar}  "
                    f"{cum_s:>8}  "
                    f"{steps_s:>5}  "
                    f"{colored(line, col)}"
                )

            print()

    # ── ranking table ─────────────────────────────────────────────────────

    def _render_ranking_table(self) -> None:
        if not self._ranked:
            return

        kept = sum(1 for _, _, s in self._ranked if s)
        print(
            f"  {colored('🏅 Kept after ranking', C.BRIGHT_YELLOW, C.BOLD)} "
            f"{colored(f'(top {kept} of {len(self._ranked)})', C.DIM)}"
        )

        medals  = ["🥇", "🥈", "🥉"]
        max_s   = self.tw - 32

        for rank, summary, selected in self._ranked:
            icon  = (
                colored("✓", C.BRIGHT_GREEN)
                if selected
                else colored("✗", C.DIM)
            )
            medal = medals[rank - 1] if rank <= 3 else f" #{rank}"
            sm    = re.search(r"score=([\d.]+)", summary)
            bar   = score_bar(float(sm.group(1)), width=8) if sm else ""
            short = summary[:max_s] + ("…" if len(summary) > max_s else "")
            row_col = (
                C.BRIGHT_GREEN if rank == 1
                else C.GREEN    if selected
                else C.DIM
            )
            print(
                f"  {icon} {medal}  {bar}  "
                f"{colored(short, row_col)}"
            )

    # ── step footer ───────────────────────────────────────────────────────

    def _render_step_footer(
        self, step: int, elapsed: float, scores: List[float]
    ) -> None:
        self._history.append(StepRecord(
            step=step, active=self._active,
            finished=self._finished_count,
            elapsed_s=elapsed, beam_scores=scores,
        ))
        for i, s in enumerate(scores):
            self._score_history[i].append(s)

        self._hline("─", C.DIM)
        print(
            f"  ⏱  Step {step} finished in "
            f"{colored(f'{elapsed:.1f}s', C.BRIGHT_WHITE, C.BOLD)}"
        )

        spark_chars = " ▁▂▃▄▅▆▇█"
        for bidx, hist in sorted(self._score_history.items()):
            col   = self._beam_color(bidx)
            sym   = self._beam_symbol(bidx)
            spark = ""
            for v in hist[-20:]:
                idx    = min(
                    int(v * (len(spark_chars) - 1)), len(spark_chars) - 1
                )
                spark += colored(spark_chars[idx], score_color(v))
            cur = colored(f"{hist[-1]:.4f}", score_color(hist[-1]), C.BOLD)
            print(f"  {colored(sym, col)} Beam {bidx}  {spark}  {cur}")

        self._hline("═", C.BLUE)

    # ── all beams finished ────────────────────────────────────────────────

    def _render_all_finished(self, n: int, step: int) -> None:
        print()
        print(colored(
            f"  🎉  All {n} beams finished at step {step}",
            C.BRIGHT_GREEN, C.BOLD,
        ))

    # ── search complete ───────────────────────────────────────────────────

    def _render_search_complete(self, elapsed: float, n: int) -> None:
        print()
        self._hline("═", C.BRIGHT_GREEN)
        print(colored(
            f"  ✅  Search complete  —  {elapsed:.1f}s  "
            f"—  decoding {n} beam(s)",
            C.BRIGHT_GREEN, C.BOLD,
        ))
        if self._history:
            self._render_score_timeline()
        self._hline("═", C.BRIGHT_GREEN)

    def _render_score_timeline(self) -> None:
        chart_w     = min(len(self._history), self.tw - 20)
        spark_chars = "▁▂▃▄▅▆▇█"

        print(f"\n  {colored('Score timeline (all steps)', C.BOLD)}")
        for bidx in sorted(self._score_history):
            hist  = self._score_history[bidx][-chart_w:]
            col   = self._beam_color(bidx)
            sym   = self._beam_symbol(bidx)
            spark = ""
            for v in hist:
                idx    = min(
                    int(v * (len(spark_chars) - 1)), len(spark_chars) - 1
                )
                spark += colored(spark_chars[idx], score_color(v))
            best = colored(
                f"{hist[-1]:.4f}", score_color(hist[-1]), C.BOLD
            )
            print(f"  {colored(sym, col)} B{bidx}  {spark}  {best}")

        times   = [r.elapsed_s for r in self._history[-chart_w:]]
        max_t   = max(times) if times else 1
        bar_row = ""
        for t in times:
            idx      = min(int(t / max_t * 7), 7)
            bar_row += colored("▁▂▃▄▅▆▇█"[idx], C.DIM)
        print(f"  {colored('⏱ time ', C.DIM)} {bar_row}")

    # ── final beam ────────────────────────────────────────────────────────

    def _render_final_beam(
        self,
        bidx: int, beam_score: float, final_score: float,
        scored_steps: int, rasterizable: bool,
        code_lines: int, decode_time: float,
    ) -> None:
        col  = self._beam_color(bidx)
        sym  = self._beam_symbol(bidx)
        rast = (
            colored("✅ rasterizable",     C.BRIGHT_GREEN)
            if rasterizable
            else colored("❌ not rasterizable", C.BRIGHT_RED)
        )
        print()
        print(
            f"  {colored(sym, col, C.BOLD)}  "
            f"{colored('Final Beam ' + str(bidx), col, C.BOLD)}"
        )
        print(
            f"  {colored('├', C.DIM)} Beam score  : "
            f"{score_bar(beam_score, width=12)}"
        )
        print(
            f"  {colored('├', C.DIM)} Final score : "
            f"{score_bar(final_score, width=12)}"
        )
        print(
            f"  {colored('├', C.DIM)} {rast}   "
            f"{colored(str(code_lines) + ' lines', C.DIM)}   "
            f"⏱ {decode_time:.2f}s"
        )
        print(f"  {colored('└', C.DIM)} Scored steps: {scored_steps}")

    # ── verifier stats ────────────────────────────────────────────────────

    def _render_verifier_stats(
        self, calls: int, skipped: int, total: float, avg: float
    ) -> None:
        print()
        self._hline("─", C.DIM)
        print(
            f"  {colored('🔍 Verifier summary', C.BRIGHT_CYAN, C.BOLD)}"
        )
        print(
            f"  {colored('├', C.DIM)} Total calls : "
            f"{colored(str(calls), C.BOLD)}"
        )
        print(
            f"  {colored('├', C.DIM)} Boilerplate : "
            f"{colored(str(skipped), C.DIM)} skipped"
        )
        print(
            f"  {colored('├', C.DIM)} Total time  : "
            f"{colored(f'{total:.1f}s', C.BOLD)}"
        )
        print(
            f"  {colored('└', C.DIM)} Avg / call  : "
            f"{colored(f'{avg:.3f}s', C.BOLD)}"
        )
        self._hline("─", C.DIM)
from .visualizer import GraphLogger

logger = GraphLogger(
    output_prefix  = "beam_graph",
    formats        = ["html", "graphml", "gexf", "dot"],
    beam_width     = 3,       # must match your DetikzifyPipeline beam_width
    terminal_width = None,    # auto-detect
)

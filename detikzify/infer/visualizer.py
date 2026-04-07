"""
GraphLogger — combines BeamSearchVisualizer (rich terminal output)
with BeamSearchGraph (file export) in a single logger object.
"""
import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .colors             import C, colored, score_color, score_bar
from .records            import GraphNode, GraphEdge
from .beamsearchvisualizer import BeamSearchVisualizer


# ─────────────────────────────────────────────────────────────────────────────
#  Graph container + exporters
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BeamSearchGraph:
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    _node_index: Dict[str, GraphNode] = field(default_factory=dict, repr=False)
    _edge_count: int = field(default=0, repr=False)

    def add_node(self, node: GraphNode) -> None:
        self.nodes.append(node)
        self._node_index[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)
        self._edge_count += 1

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._node_index.get(node_id)

    # ── exporters ─────────────────────────────────────────────────────────

    def to_graphml(self) -> str:
        def xe(s: str) -> str:
            return (s.replace("&","&amp;").replace("<","&lt;")
                     .replace(">","&gt;").replace('"',"&quot;"))

        def d(key, val):
            return f'      <data key="{key}">{xe(str(val))}</data>'

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/graphml"',
            '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            '         xsi:schemaLocation="http://graphml.graphdrawing.org/graphml',
            '         http://graphml.graphdrawing.org/graphml/1.0/graphml.xsd">',
            '',
            '  <key id="n_label"        for="node" attr.name="label"        attr.type="string"/>',
            '  <key id="n_step"         for="node" attr.name="step"         attr.type="int"/>',
            '  <key id="n_beam"         for="node" attr.name="beam_idx"     attr.type="int"/>',
            '  <key id="n_score"        for="node" attr.name="score"        attr.type="double"/>',
            '  <key id="n_verifier"     for="node" attr.name="verifier"     attr.type="double"/>',
            '  <key id="n_cumulative"   for="node" attr.name="cumulative"   attr.type="double"/>',
            '  <key id="n_lines"        for="node" attr.name="lines"        attr.type="int"/>',
            '  <key id="n_tokens"       for="node" attr.name="tokens"       attr.type="int"/>',
            '  <key id="n_scored_steps" for="node" attr.name="scored_steps" attr.type="int"/>',
            '  <key id="n_last_line"    for="node" attr.name="last_line"    attr.type="string"/>',
            '  <key id="n_finished"     for="node" attr.name="finished"     attr.type="boolean"/>',
            '  <key id="n_selected"     for="node" attr.name="selected"     attr.type="boolean"/>',
            '  <key id="n_rank"         for="node" attr.name="rank"         attr.type="int"/>',
            '  <key id="e_label"        for="edge" attr.name="label"        attr.type="string"/>',
            '  <key id="e_verifier"     for="edge" attr.name="verifier_score" attr.type="double"/>',
            '',
            '  <graph id="BeamSearch" edgedefault="directed">',
        ]

        for n in self.nodes:
            lines += [
                f'    <node id="{n.node_id}">',
                d("n_label",        n.label),
                d("n_step",         n.step),
                d("n_beam",         n.beam_idx),
                d("n_score",        round(n.score,       6)),
                d("n_verifier",     round(n.verifier,    6)),
                d("n_cumulative",   round(n.cumulative,  6)),
                d("n_lines",        n.lines),
                d("n_tokens",       n.tokens),
                d("n_scored_steps", n.scored_steps),
                d("n_last_line",    n.last_line),
                d("n_finished",     str(n.finished).lower()),
                d("n_selected",     str(n.selected).lower()),
                d("n_rank",         n.rank if n.rank is not None else -1),
                "    </node>",
            ]

        for e in self.edges:
            lines += [
                f'    <edge id="{e.edge_id}" '
                f'source="{e.source}" target="{e.target}">',
                d("e_label",    e.label),
                d("e_verifier", round(e.verifier_score, 6)),
                "    </edge>",
            ]

        lines += ["  </graph>", "</graphml>"]
        return "\n".join(lines)

    def to_gexf(self) -> str:
        def score_rgb(s: float) -> Tuple[int, int, int]:
            s = max(0.0, min(1.0, s))
            if s >= 0.5:
                t = (s - 0.5) * 2
                return int(255*(1-t)), 255, 0
            t = s * 2
            return 255, int(255*t), 0

        def xe(s):
            return (s.replace("&","&amp;").replace("<","&lt;")
                     .replace(">","&gt;").replace('"',"&quot;"))

        ts    = time.strftime("%Y-%m-%d")
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gexf xmlns="http://gexf.net/1.3"',
            '      xmlns:viz="http://gexf.net/1.3/viz"',
            '      version="1.3">',
            f'  <meta lastmodifieddate="{ts}">',
            '    <creator>BeamSearchVisualizer</creator>',
            '  </meta>',
            '  <graph defaultedgetype="directed">',
            '    <attributes class="node">',
            '      <attribute id="0"  title="step"         type="integer"/>',
            '      <attribute id="1"  title="beam_idx"     type="integer"/>',
            '      <attribute id="2"  title="score"        type="double"/>',
            '      <attribute id="3"  title="verifier"     type="double"/>',
            '      <attribute id="4"  title="cumulative"   type="double"/>',
            '      <attribute id="5"  title="lines"        type="integer"/>',
            '      <attribute id="6"  title="tokens"       type="integer"/>',
            '      <attribute id="7"  title="scored_steps" type="integer"/>',
            '      <attribute id="8"  title="last_line"    type="string"/>',
            '      <attribute id="9"  title="finished"     type="boolean"/>',
            '      <attribute id="10" title="selected"     type="boolean"/>',
            '      <attribute id="11" title="rank"         type="integer"/>',
            '    </attributes>',
            '    <attributes class="edge">',
            '      <attribute id="0" title="verifier_score" type="double"/>',
            '    </attributes>',
            '    <nodes>',
        ]

        for n in self.nodes:
            r, g, b = score_rgb(n.score)
            size    = 10 + n.score * 20
            lines += [
                f'      <node id="{n.node_id}" label="{xe(n.label)}">',
                f'        <attvalues>',
                f'          <attvalue for="0"  value="{n.step}"/>',
                f'          <attvalue for="1"  value="{n.beam_idx}"/>',
                f'          <attvalue for="2"  value="{round(n.score,      6)}"/>',
                f'          <attvalue for="3"  value="{round(n.verifier,   6)}"/>',
                f'          <attvalue for="4"  value="{round(n.cumulative, 6)}"/>',
                f'          <attvalue for="5"  value="{n.lines}"/>',
                f'          <attvalue for="6"  value="{n.tokens}"/>',
                f'          <attvalue for="7"  value="{n.scored_steps}"/>',
                f'          <attvalue for="8"  value="{xe(n.last_line)}"/>',
                f'          <attvalue for="9"  value="{str(n.finished).lower()}"/>',
                f'          <attvalue for="10" value="{str(n.selected).lower()}"/>',
                f'          <attvalue for="11" value="{n.rank if n.rank is not None else -1}"/>',
                f'        </attvalues>',
                f'        <viz:color r="{r}" g="{g}" b="{b}" a="1.0"/>',
                f'        <viz:size value="{round(size, 2)}"/>',
                f'        <viz:position x="{n.step*120}" y="{n.beam_idx*80}" z="0"/>',
                f'        <viz:shape value="{"diamond" if n.finished else "disc"}"/>',
                f'      </node>',
            ]

        lines += ["    </nodes>", "    <edges>"]
        for e in self.edges:
            lines += [
                f'      <edge id="{e.edge_id}" '
                f'source="{e.source}" target="{e.target}" '
                f'label="{xe(e.label)}">',
                f'        <attvalues>',
                f'          <attvalue for="0" value="{round(e.verifier_score,6)}"/>',
                f'        </attvalues>',
                f'      </edge>',
            ]
        lines += ["    </edges>", "  </graph>", "</gexf>"]
        return "\n".join(lines)

    def to_cytoscape_json(self) -> str:
        def hex_color(s: float) -> str:
            s = max(0.0, min(1.0, s))
            if s >= 0.5:
                t = (s-0.5)*2; r = int(255*(1-t)); g = 255
            else:
                t = s*2; r = 255; g = int(255*t)
            return f"#{r:02x}{g:02x}00"

        elements: Dict = {"nodes": [], "edges": []}
        for n in self.nodes:
            elements["nodes"].append({
                "data": {
                    "id": n.node_id, "label": n.label,
                    "step": n.step, "beam_idx": n.beam_idx,
                    "score": round(n.score, 6),
                    "verifier": round(n.verifier, 6),
                    "cumulative": round(n.cumulative, 6),
                    "lines": n.lines, "tokens": n.tokens,
                    "scored_steps": n.scored_steps,
                    "last_line": n.last_line,
                    "finished": n.finished, "selected": n.selected,
                    "rank": n.rank,
                },
                "position": {"x": n.step*180, "y": n.beam_idx*100},
                "style": {
                    "background-color": hex_color(n.score),
                    "border-color": "#00ff00" if n.selected else "#888",
                    "border-width": 3 if n.selected else 1,
                    "shape": "diamond" if n.finished else "ellipse",
                    "width":  20 + n.score*30,
                    "height": 20 + n.score*30,
                },
            })
        for e in self.edges:
            elements["edges"].append({
                "data": {
                    "id": e.edge_id, "source": e.source,
                    "target": e.target, "label": e.label,
                    "verifier_score": round(e.verifier_score, 6),
                },
            })
        return json.dumps({"elements": elements}, indent=2)

    def to_dot(self) -> str:
        def hex_color(s: float) -> str:
            s = max(0.0, min(1.0, s))
            if s >= 0.5:
                t = (s-0.5)*2; r = int(255*(1-t)); g = 255
            else:
                t = s*2; r = 255; g = int(255*t)
            return f"#{r:02x}{g:02x}00"

        def de(s: str) -> str:
            return s.replace("\\","\\\\").replace('"','\\"').replace("\n","\\n")

        lines = [
            "digraph BeamSearch {",
            "  rankdir=LR;",
            '  graph [fontname="Helvetica" bgcolor="#1a1a2e"];',
            '  node  [fontname="Helvetica" fontcolor=white style=filled fontsize=9];',
            '  edge  [fontname="Helvetica" fontcolor="#aaaaaa" fontsize=7 color="#555555"];',
            "",
        ]
        steps: Dict[int, List[GraphNode]] = defaultdict(list)
        for n in self.nodes:
            steps[n.step].append(n)

        for step, snodes in sorted(steps.items()):
            lines.append(f"  subgraph cluster_step{step} {{")
            lines.append(f'    label="Step {step}"; color="#333355"; fontcolor="#aaaaaa";')
            for n in snodes:
                color = hex_color(n.score)
                shape = "diamond" if n.finished else "ellipse"
                pw    = "2" if n.selected else "1"
                short = de(n.last_line[:35] + ("…" if len(n.last_line)>35 else ""))
                lbl   = de(
                    f"B{n.beam_idx} s{n.step}\\n"
                    f"score={n.score:.3f}\\n{short}"
                )
                lines.append(
                    f'    "{n.node_id}" [label="{lbl}" fillcolor="{color}" '
                    f'shape={shape} penwidth={pw}];'
                )
            lines.append("  }")
            lines.append("")

        for e in self.edges:
            color = hex_color(e.verifier_score)
            width = round(0.5 + e.verifier_score*3, 2)
            lines.append(
                f'  "{e.source}" -> "{e.target}" '
                f'[label="{e.verifier_score:.2f}" '
                f'color="{color}" penwidth={width}];'
            )
        lines.append("}")
        return "\n".join(lines)

    def to_d3_html(self) -> str:
        graph_data = {
            "nodes": [
                {
                    "id": n.node_id, "label": n.label,
                    "step": n.step, "beam_idx": n.beam_idx,
                    "score": round(n.score, 6),
                    "verifier": round(n.verifier, 6),
                    "cumulative": round(n.cumulative, 6),
                    "lines": n.lines, "tokens": n.tokens,
                    "scored_steps": n.scored_steps,
                    "last_line": n.last_line,
                    "finished": n.finished, "selected": n.selected,
                    "rank": n.rank,
                }
                for n in self.nodes
            ],
            "links": [
                {
                    "source": e.source, "target": e.target,
                    "verifier_score": round(e.verifier_score, 6),
                    "label": e.label,
                }
                for e in self.edges
            ],
        }
        data_json = json.dumps(graph_data, indent=2)

        # D3 HTML template — identical to previous version
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Beam Search Graph</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d1117;color:#c9d1d9;font-family:monospace}}
  #toolbar{{position:fixed;top:0;left:0;right:0;z-index:10;
    background:#161b22;border-bottom:1px solid #30363d;
    padding:8px 16px;display:flex;gap:16px;align-items:center}}
  #toolbar h1{{font-size:14px;color:#58a6ff}}
  #toolbar label{{font-size:12px;color:#8b949e}}
  #legend{{position:fixed;bottom:16px;left:16px;z-index:10;
    background:#161b22cc;border:1px solid #30363d;
    padding:10px;border-radius:6px;font-size:11px;min-width:160px}}
  .legend-row{{display:flex;align-items:center;gap:8px;margin:3px 0}}
  .legend-dot{{width:12px;height:12px;border-radius:50%}}
  #tooltip{{position:fixed;pointer-events:none;
    background:#161b22ee;border:1px solid #30363d;
    border-radius:6px;padding:10px;font-size:11px;
    max-width:320px;z-index:20;display:none;line-height:1.6}}
  #tooltip .tt-title{{color:#58a6ff;font-weight:bold;margin-bottom:6px}}
  #tooltip .tt-row{{display:flex;justify-content:space-between;gap:16px}}
  #tooltip .tt-key{{color:#8b949e}}
  #tooltip .tt-val{{color:#e6edf3}}
  #tooltip .tt-line{{color:#7ee787;margin-top:6px;padding-top:6px;
    border-top:1px solid #30363d;word-break:break-all;white-space:pre-wrap}}
  svg{{display:block;margin-top:48px}}
  .node circle{{stroke-width:2;cursor:grab;transition:stroke 0.2s,stroke-width 0.2s}}
  .node circle:hover{{stroke-width:4}}
  .node.selected circle{{stroke:#58a6ff!important;stroke-width:3}}
  .node.finished circle{{stroke-dasharray:4 2}}
  .node text{{font-size:8px;fill:#c9d1d9;pointer-events:none;text-anchor:middle}}
  .link{{stroke-opacity:0.6;marker-end:url(#arrow)}}
  .step-label{{font-size:11px;fill:#8b949e;text-anchor:middle}}
</style>
</head>
<body>
<div id="toolbar">
  <h1>🔍 Beam Search Graph</h1>
  <label>Layout:
    <select id="layout-select">
      <option value="force">Force-directed</option>
      <option value="grid" selected>Grid (step × beam)</option>
    </select>
  </label>
  <label>Link strength:
    <input type="range" id="link-strength" min="0" max="1" step="0.05" value="0.3"/>
  </label>
  <label><input type="checkbox" id="show-labels" checked/> Labels</label>
  <label><input type="checkbox" id="only-selected"/> Selected only</label>
  <span id="stats" style="margin-left:auto;color:#8b949e;font-size:12px;"></span>
</div>
<div id="legend">
  <div style="font-weight:bold;margin-bottom:6px;color:#58a6ff;">Score colour</div>
  <div class="legend-row"><div class="legend-dot" style="background:#00ff00"></div> 1.0 (best)</div>
  <div class="legend-row"><div class="legend-dot" style="background:#ffff00"></div> 0.5</div>
  <div class="legend-row"><div class="legend-dot" style="background:#ff0000"></div> 0.0 (worst)</div>
  <hr style="border-color:#30363d;margin:6px 0"/>
  <div class="legend-row"><div class="legend-dot" style="background:#444;border:2px solid #58a6ff"></div> selected</div>
  <div class="legend-row"><div class="legend-dot" style="background:#444;border:2px dashed #aaa"></div> finished</div>
</div>
<div id="tooltip"></div>
<svg id="graph"></svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const RAW={data_json};
function scoreColor(s){{
  s=Math.max(0,Math.min(1,s));
  if(s>=0.5){{const t=(s-0.5)*2;return `rgb(${{Math.round(255*(1-t))}},255,0)`;}}
  const t=s*2;return `rgb(255,${{Math.round(255*t)}},0)`;
}}
function fmt(v,dp=4){{return typeof v==='number'?v.toFixed(dp):String(v);}}
const nodeById={{}};
RAW.nodes.forEach(n=>nodeById[n.id]=n);
RAW.links.forEach(l=>{{l.sourceNode=nodeById[l.source];l.targetNode=nodeById[l.target];}});
const maxStep=Math.max(...RAW.nodes.map(n=>n.step));
const maxBeam=Math.max(...RAW.nodes.map(n=>n.beam_idx));
const W=window.innerWidth,H=window.innerHeight-48;
const svg=d3.select("svg#graph").attr("width",W).attr("height",H);
svg.append("defs").append("marker")
  .attr("id","arrow").attr("viewBox","0 -5 10 10")
  .attr("refX",18).attr("refY",0)
  .attr("markerWidth",6).attr("markerHeight",6)
  .attr("orient","auto")
  .append("path").attr("d","M0,-5L10,0L0,5").attr("fill","#555");
const g=svg.append("g");
svg.call(d3.zoom().scaleExtent([0.1,5]).on("zoom",e=>g.attr("transform",e.transform)));
const PAD_X=140,PAD_Y=100;
const STEP_W=Math.min(180,(W-PAD_X*2)/Math.max(1,maxStep));
const BEAM_H=Math.min(120,(H-PAD_Y*2)/Math.max(1,maxBeam+1));
const posMap={{}};
RAW.nodes.forEach(n=>{{
  const key=`${{n.step}}_${{n.beam_idx}}`;
  if(!posMap[key])posMap[key]=[];
  posMap[key].push(n);
}});
RAW.nodes.forEach(n=>{{
  const key=`${{n.step}}_${{n.beam_idx}}`;
  const grp=posMap[key];
  const idx=grp.indexOf(n);
  n._gx=PAD_X+n.step*STEP_W;
  n._gy=PAD_Y+n.beam_idx*BEAM_H+(grp.length>1?(idx-(grp.length-1)/2)*18:0);
}});
const sim=d3.forceSimulation(RAW.nodes)
  .force("link",d3.forceLink(RAW.links).id(d=>d.id).distance(80).strength(0.3))
  .force("charge",d3.forceManyBody().strength(-120))
  .force("x",d3.forceX(n=>n._gx).strength(0.8))
  .force("y",d3.forceY(n=>n._gy).strength(0.8))
  .force("collision",d3.forceCollide(18));
const stepSet=[...new Set(RAW.nodes.map(n=>n.step))].sort((a,b)=>a-b);
g.selectAll(".step-label").data(stepSet).enter()
  .append("text").attr("class","step-label")
  .attr("x",s=>PAD_X+s*STEP_W).attr("y",16)
  .text(s=>`Step ${{s}}`);
const tooltip=document.getElementById("tooltip");
let linkSel=g.append("g").selectAll(".link").data(RAW.links).enter()
  .append("line").attr("class","link")
  .attr("stroke",d=>scoreColor(d.verifier_score))
  .attr("stroke-width",d=>0.5+d.verifier_score*3);
let nodeSel=g.append("g").selectAll(".node").data(RAW.nodes).enter()
  .append("g")
  .attr("class",d=>"node"+(d.selected?" selected":"")+(d.finished?" finished":""))
  .call(d3.drag()
    .on("start",(e,d)=>{{if(!e.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y;}})
    .on("drag", (e,d)=>{{d.fx=e.x;d.fy=e.y;}})
    .on("end",  (e,d)=>{{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}}))
  .on("mousemove",(e,d)=>{{
    tooltip.style.display="block";
    tooltip.style.left=(e.clientX+14)+"px";
    tooltip.style.top=(e.clientY-10)+"px";
    tooltip.innerHTML=`
      <div class="tt-title">Step ${{d.step}} · Beam ${{d.beam_idx}}
        ${{d.selected?"✅":""}} ${{d.finished?"🏁":""}}</div>
      <div class="tt-row"><span class="tt-key">Score</span><span class="tt-val">${{fmt(d.score)}}</span></div>
      <div class="tt-row"><span class="tt-key">Verifier</span><span class="tt-val">${{fmt(d.verifier)}}</span></div>
      <div class="tt-row"><span class="tt-key">Cumulative</span><span class="tt-val">${{fmt(d.cumulative)}}</span></div>
      <div class="tt-row"><span class="tt-key">Lines</span><span class="tt-val">${{d.lines}}</span></div>
      <div class="tt-row"><span class="tt-key">Tokens</span><span class="tt-val">${{d.tokens}}</span></div>
      <div class="tt-row"><span class="tt-key">Scored steps</span><span class="tt-val">${{d.scored_steps}}</span></div>
      <div class="tt-row"><span class="tt-key">Rank</span><span class="tt-val">${{d.rank??"—"}}</span></div>
      <div class="tt-line">${{d.last_line}}</div>`;
  }})
  .on("mouseleave",()=>{{tooltip.style.display="none";}});
nodeSel.append("circle")
  .attr("r",d=>10+d.score*14)
  .attr("fill",d=>scoreColor(d.score))
  .attr("stroke",d=>d.selected?"#58a6ff":"#444")
  .attr("stroke-dasharray",d=>d.finished?"4 2":"none");
nodeSel.append("text")
  .attr("dy","0.35em")
  .attr("y",d=>-(12+d.score*14))
  .text(d=>`B${{d.beam_idx}} ${{fmt(d.score,2)}}`);
sim.on("tick",()=>{{
  linkSel.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y)
         .attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
  nodeSel.attr("transform",d=>`translate(${{d.x}},${{d.y}})`);
}});
document.getElementById("layout-select").addEventListener("change",e=>{{
  const useGrid=e.target.value==="grid";
  sim.force("x").strength(useGrid?0.8:0.1);
  sim.force("y").strength(useGrid?0.8:0.1);
  sim.alpha(0.5).restart();
}});
document.getElementById("link-strength").addEventListener("input",e=>{{
  sim.force("link").strength(+e.target.value);sim.alpha(0.3).restart();
}});
document.getElementById("show-labels").addEventListener("change",e=>{{
  nodeSel.selectAll("text").style("display",e.target.checked?null:"none");
}});
document.getElementById("only-selected").addEventListener("change",e=>{{
  const onlySel=e.target.checked;
  nodeSel.style("display",d=>onlySel&&!d.selected?"none":null);
  linkSel.style("display",d=>{{
    if(!onlySel)return null;
    return(d.sourceNode?.selected&&d.targetNode?.selected)?null:"none";
  }});
}});
document.getElementById("stats").textContent=
  `${{RAW.nodes.length}} nodes · ${{RAW.links.length}} edges · `+
  `${{maxStep+1}} steps · ${{maxBeam+1}} beams`;
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  GraphLogger
# ─────────────────────────────────────────────────────────────────────────────

class GraphLogger:
    """
    Drop-in replacement for SimpleLogger.
    Delegates all terminal rendering to BeamSearchVisualizer and
    simultaneously builds a BeamSearchGraph for file export.
    """

    def __init__(
        self,
        output_prefix:  str            = "beam_graph",
        formats:        Optional[List[str]] = None,
        beam_width:     int            = 3,
        terminal_width: Optional[int]  = None,
    ):
        self.prefix  = output_prefix
        self.formats = formats or ["html", "graphml", "gexf", "dot"]
        self.graph   = BeamSearchGraph()

        # All terminal rendering delegated here
        self.viz = BeamSearchVisualizer(
            beam_width     = beam_width,
            terminal_width = terminal_width,
        )

        # Graph-building state
        self._step:         int             = 0
        self._parent_ids:   Dict[int, str]  = {}
        self._cand_buffer:  List[Dict]      = []
        self._node_counter: int             = 0

        self._suppress = [
            "max_new_tokens", "max_length",
            "A decoder-only", "padding_side",
            "generation flags", "TRANSFORMERS_VERBOSITY",
        ]

    # ── public logger API ─────────────────────────────────────────────────

    def info(self, msg: str)    -> None: self._dispatch(msg, "INFO")
    def debug(self, msg: str)   -> None: self._dispatch(msg, "DEBUG")
    def warning(self, msg: str) -> None: self._dispatch(msg, "WARN")
    def error(self, msg: str)   -> None: self._dispatch(msg, "ERROR")

    # ── router ────────────────────────────────────────────────────────────

    def _dispatch(self, msg: str, level: str) -> None:
        if any(s in msg for s in self._suppress):
            return

        # 1. Pretty terminal output via visualizer
        getattr(self.viz, {
            "INFO":  "info",
            "DEBUG": "debug",
            "WARN":  "warning",
            "ERROR": "error",
        }[level])(msg)

        # 2. Silent graph construction
        tag, body = self._split_tag(msg)
        if tag == "[BeamSearch]":
            self._parse_for_graph(body)

    @staticmethod
    def _split_tag(msg: str) -> Tuple[str, str]:
        m = re.match(r"(\[[A-Za-z0-9_]+\])\s*(.*)", msg, re.DOTALL)
        return (m.group(1), m.group(2).strip()) if m else ("", msg)

    # ── graph parser ──────────────────────────────────────────────────────

    def _parse_for_graph(self, body: str) -> None:

        # new step
        m = re.match(
            r"── Step (\d+)/\d+ ── active=\d+, finished=\d+", body
        )
        if m:
            self._step = int(m.group(1))
            self._cand_buffer.clear()
            return

        # current beam state
        m = re.match(
            r"Beam (\d+): \[.*?\] score=([\d.]+) lines=(\d+) "
            r"scored_steps=(\d+) tokens=(\d+) last_line=\"(.*)\"",
            body,
        )
        if m:
            bidx, score, lines, ss, tokens, last = m.groups()
            bid = int(bidx)
            if bid not in self._parent_ids:
                nid  = self._new_node_id(self._step - 1, bid)
                node = GraphNode(
                    node_id      = nid,
                    step         = self._step - 1,
                    beam_idx     = bid,
                    score        = float(score),
                    verifier     = float(score),
                    cumulative   = float(score),
                    lines        = int(lines),
                    tokens       = int(tokens),
                    scored_steps = int(ss),
                    last_line    = last,
                    finished     = False,
                    selected     = True,
                )
                self.graph.add_node(node)
                self._parent_ids[bid] = nid
            return

        # scored candidate
        m = re.match(
            r"Beam (\d+) -> verifier=([\d.]+) cumulative=([\d.]+) "
            r"scored_steps=(\d+) line=\"(.*)\"",
            body,
        )
        if m:
            bidx, ver, cum, ss, line = m.groups()
            self._cand_buffer.append({
                "beam_idx":     int(bidx),
                "verifier":     float(ver),
                "cumulative":   float(cum),
                "scored_steps": int(ss),
                "line":         line,
            })
            return

        # ranking entry → create node + edge
        m = re.match(r"[✓✗] Rank (\d+): (.*)", body)
        if m:
            rank    = int(m.group(1))
            summary = m.group(2)

            sm    = re.search(r"score=([\d.]+)", summary)
            score = float(sm.group(1)) if sm else 0.0

            best = min(
                self._cand_buffer,
                key=lambda c: abs(c["cumulative"] - score),
                default=None,
            )
            if best is None:
                return

            bid  = best["beam_idx"]
            nid  = self._new_node_id(self._step, bid)

            lines_m  = re.search(r"lines=(\d+)",  summary)
            tokens_m = re.search(r"tokens=(\d+)", summary)

            node = GraphNode(
                node_id      = nid,
                step         = self._step,
                beam_idx     = bid,
                score        = score,
                verifier     = best["verifier"],
                cumulative   = best["cumulative"],
                lines        = int(lines_m.group(1))  if lines_m  else 0,
                tokens       = int(tokens_m.group(1)) if tokens_m else 0,
                scored_steps = best["scored_steps"],
                last_line    = best["line"],
                finished     = "FINISHED" in summary,
                selected     = rank <= self.viz.beam_width,
                rank         = rank,
            )
            self.graph.add_node(node)

            parent_id = self._parent_ids.get(bid)
            if parent_id:
                self.graph.add_edge(GraphEdge(
                    edge_id        = f"e_{parent_id}_{nid}",
                    source         = parent_id,
                    target         = nid,
                    verifier_score = best["verifier"],
                    label          = f"{best['verifier']:.2f}",
                ))

            if node.selected:
                self._parent_ids[bid] = nid

            self._cand_buffer.remove(best)
            return

    def _new_node_id(self, step: int, beam: int) -> str:
        self._node_counter += 1
        return f"s{step}_b{beam}_n{self._node_counter}"

    # ── save ──────────────────────────────────────────────────────────────

    def save(self) -> Dict[str, str]:
        dispatch = {
            "graphml":   (self.graph.to_graphml,        ".graphml"),
            "gexf":      (self.graph.to_gexf,           ".gexf"),
            "cytoscape": (self.graph.to_cytoscape_json, "_cytoscape.json"),
            "dot":       (self.graph.to_dot,            ".dot"),
            "html":      (self.graph.to_d3_html,        ".html"),
        }
        written = {}
        print()
        for fmt in self.formats:
            if fmt not in dispatch:
                self.viz.warning(f"Unknown format '{fmt}', skipping.")
                continue
            fn, ext = dispatch[fmt]
            path    = self.prefix + ext
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(fn())
            written[fmt] = path
            print(
                f"  {C.BRIGHT_GREEN}✓{C.RESET}  "
                f"{C.BOLD}{fmt:12s}{C.RESET}  →  "
                f"{C.BRIGHT_CYAN}{path}{C.RESET}"
            )
        return written
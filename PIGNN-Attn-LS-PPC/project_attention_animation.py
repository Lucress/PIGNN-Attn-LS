"""
Animated HTML visualization of AC-OPF attention propagation.

Generates a self-contained HTML file showing how edge attention weights
evolve over K correction steps in PIGNN-Attn-LS.

Layout: side-by-side Supervised (K=15) vs PINN (K=30) on the IEEE 14-bus
graph. Edge thickness + opacity = attention weight at each step k.

Usage (from PIGNN-Attn-LS-PPC/):
    python project_attention_animation.py \\
        --attn_sup  results/attention_supervised \\
        --attn_pinn results/attention_pinn \\
        --out       results/figures/animation/attention_animation.html
"""
from __future__ import annotations
import argparse, json, os
import torch
import numpy as np

# ── Case14 layout (0-indexed buses) ──────────────────────────────────────────
CASE14_POS = {
    0: (0.0,2.0), 1: (1.5,3.0), 2: (3.5,3.5), 3: (4.5,2.5),
    4: (4.0,1.5), 5: (2.5,3.8), 6: (3.2,4.5), 7: (2.8,4.8),
    8: (3.8,4.8), 9: (4.5,4.2), 10:(4.2,3.8), 11:(4.8,3.5),
    12:(5.2,3.8), 13:(5.0,3.2),
}
CASE14_BUS_TYPE = {0:"slack", 1:"gen", 2:"gen", 5:"gen", 7:"gen"}
TYPE_COLOR = {"slack":"#4DA6FF", "gen":"#9F7AEA", "load":"#5A7A99"}

N_BUSES = 14
SVG_W, SVG_H = 400, 340
MARGIN = 32


def svg_coords():
    x_min = min(p[0] for p in CASE14_POS.values())
    x_max = max(p[0] for p in CASE14_POS.values())
    y_min = min(p[1] for p in CASE14_POS.values())
    y_max = max(p[1] for p in CASE14_POS.values())
    out = {}
    for i, (x, y) in CASE14_POS.items():
        sx = MARGIN + (x - x_min) / (x_max - x_min + 1e-9) * (SVG_W - 2*MARGIN)
        sy = SVG_H - MARGIN - (y - y_min) / (y_max - y_min + 1e-9) * (SVG_H - 2*MARGIN)
        out[i] = (round(sx, 1), round(sy, 1))
    return out

SVG_NODES = svg_coords()

# Quadratic Bezier control point for a directed edge (curved to avoid overlap)
def ctrl_pt(p1, p2, sign=+1, offset=12):
    dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
    L = max((dx**2 + dy**2)**0.5, 1e-6)
    nx = -dy/L; ny = dx/L   # unit normal
    mx = (p1[0]+p2[0])/2 + nx*offset*sign
    my = (p1[1]+p2[1])/2 + ny*offset*sign
    return (round(mx,1), round(my,1))


def build_edge_data(edge_index):
    """
    Returns list of edge dicts with SVG geometry.
    For edges (src,dst) and (dst,src): opposite curve directions.
    """
    seen = {}  # undirected pair → canonical direction
    edges = []
    for e_id, (src, dst) in enumerate(edge_index.tolist()):
        key = (min(src,dst), max(src,dst))
        sign = +1 if src < dst else -1
        p1 = SVG_NODES[src]; p2 = SVG_NODES[dst]
        cp = ctrl_pt(p1, p2, sign=sign, offset=14)
        # arrowhead target: midpoint along curve toward dst
        # use t=0.75 on Bezier: (1-t)^2*p1 + 2t(1-t)*cp + t^2*p2
        t = 0.75
        ax = (1-t)**2*p1[0] + 2*t*(1-t)*cp[0] + t**2*p2[0]
        ay = (1-t)**2*p1[1] + 2*t*(1-t)*cp[1] + t**2*p2[1]
        # tangent at t=0.75: 2(1-t)(cp-p1) + 2t(p2-cp)
        tx = 2*(1-t)*(cp[0]-p1[0]) + 2*t*(p2[0]-cp[0])
        ty = 2*(1-t)*(cp[1]-p1[1]) + 2*t*(p2[1]-cp[1])
        edges.append({
            "id": e_id, "src": src, "dst": dst,
            "x1": p1[0], "y1": p1[1],
            "cpx": cp[0], "cpy": cp[1],
            "x2": p2[0], "y2": p2[1],
        })
    return edges


def build_html(edge_data, attn_sup, attn_pinn, K_sup, K_pinn, E):
    node_info = [
        {"x": SVG_NODES[i][0], "y": SVG_NODES[i][1],
         "type": CASE14_BUS_TYPE.get(i, "load"),
         "label": str(i+1)}
        for i in range(N_BUSES)
    ]

    # Normalize: global max over all steps and edges for each model separately
    sup_max  = max(float(np.max(attn_sup)),  1e-9)
    pinn_max = max(float(np.max(attn_pinn)), 1e-9)

    attn_sup_norm  = (attn_sup  / sup_max).tolist()
    attn_pinn_norm = (attn_pinn / pinn_max).tolist()

    data_json = json.dumps({
        "edges":      edge_data,
        "nodes":      node_info,
        "attn_sup":   attn_sup_norm,
        "attn_pinn":  attn_pinn_norm,
        "K_sup":      K_sup,
        "K_pinn":     K_pinn,
        "E":          E,
        "typeColor":  TYPE_COLOR,
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PIGNN-Attn-LS · AC-OPF Attention Animation</title>
<style>
:root {{
  --bg:#0D1B2A; --surface:#162638; --card:#1E3348; --card2:#243B56;
  --teal:#00CFA8; --orange:#FF6B35; --blue:#4DA6FF; --purple:#9F7AEA;
  --text:#E4EDF8; --text2:#A8BDD4; --muted:#5A7A99; --border:#223347;
}}
*, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
body {{
  background:var(--bg); color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  font-size:14px; min-height:100vh; padding:24px;
}}
h1 {{ font-size:20px; font-weight:800; color:var(--teal); margin-bottom:4px; }}
.subtitle {{ color:var(--text2); font-size:13px; margin-bottom:20px; }}
.controls {{
  display:flex; align-items:center; gap:16px; flex-wrap:wrap;
  background:var(--surface); border:1px solid var(--border);
  border-radius:10px; padding:14px 20px; margin-bottom:20px;
}}
button {{
  background:var(--card); border:1px solid var(--border);
  color:var(--text); border-radius:8px; padding:8px 20px;
  font-size:14px; font-weight:700; cursor:pointer;
  transition:background .15s, border-color .15s;
}}
button:hover {{ background:var(--card2); border-color:var(--teal); color:var(--teal); }}
button.active {{ background:var(--teal); color:var(--bg); border-color:var(--teal); }}
.ctrl-group {{ display:flex; flex-direction:column; gap:4px; }}
.ctrl-label {{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }}
input[type=range] {{ width:160px; accent-color:var(--teal); }}
.step-display {{ font-size:13px; color:var(--text2); font-variant-numeric:tabular-nums; }}
.panels {{
  display:grid; grid-template-columns:1fr 1fr; gap:16px;
}}
.panel {{
  background:var(--surface); border:1px solid var(--border);
  border-radius:12px; padding:16px;
}}
.panel-title {{
  font-size:13px; font-weight:700; margin-bottom:4px;
}}
.panel-sub {{ font-size:11px; color:var(--muted); margin-bottom:12px; }}
svg.graph {{ width:100%; height:auto; }}
.legend {{
  display:flex; gap:14px; flex-wrap:wrap; margin-top:14px;
  font-size:12px; color:var(--text2);
}}
.leg-item {{ display:flex; align-items:center; gap:5px; }}
.leg-dot {{ width:12px; height:12px; border-radius:50%; }}
.leg-line {{ width:28px; height:3px; border-radius:2px; }}
.physics-note {{
  background:var(--card); border:1px solid var(--border); border-left:3px solid var(--teal);
  border-radius:8px; padding:14px 18px; margin-top:16px; font-size:13px;
  color:var(--text2); line-height:1.65;
}}
.physics-note strong {{ color:var(--text); }}
.heat-bar {{
  margin-top:12px; display:flex; align-items:center; gap:8px; font-size:11px; color:var(--muted);
}}
.heat-grad {{
  width:120px; height:8px; border-radius:4px;
  background: linear-gradient(to right, rgba(0,207,168,0.1), rgba(0,207,168,1));
}}
.heat-grad.orange {{ background: linear-gradient(to right, rgba(255,107,53,0.1), rgba(255,107,53,1)); }}
@media(max-width:680px) {{ .panels {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<h1>PIGNN-Attn-LS · AC-OPF Attention Animation</h1>
<p class="subtitle">
  How edge self-attention evolves over K iterative correction steps — IEEE 14-bus · 30 test scenarios (averaged)
</p>

<div class="controls">
  <button id="playBtn">▶ Play</button>

  <div class="ctrl-group">
    <span class="ctrl-label">Step</span>
    <input type="range" id="stepSlider" min="0" max="29" value="0" step="1">
  </div>
  <span class="step-display" id="stepDisplay">Step 1 / 30 (PINN) &nbsp;|&nbsp; Step 1 / 15 (Supervised)</span>

  <div class="ctrl-group">
    <span class="ctrl-label">Speed</span>
    <input type="range" id="speedSlider" min="50" max="800" value="250" step="50">
  </div>
  <span class="step-display" id="speedDisplay">250 ms / step</span>
</div>

<div class="panels">
  <div class="panel">
    <div class="panel-title" style="color:var(--teal)">Supervised (K=15)</div>
    <div class="panel-sub" id="sup-sub">Step k=1 · edge attention from segmented softmax</div>
    <svg id="sup-svg" class="graph" viewBox="0 0 {SVG_W} {SVG_H}"></svg>
    <div class="heat-bar">
      <span>low</span><div class="heat-grad"></div><span>high attention</span>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title" style="color:var(--orange)">PINN (K=30)</div>
    <div class="panel-sub" id="pinn-sub">Step k=1 · edge attention from segmented softmax</div>
    <svg id="pinn-svg" class="graph" viewBox="0 0 {SVG_W} {SVG_H}"></svg>
    <div class="heat-bar">
      <span>low</span><div class="heat-grad orange"></div><span>high attention</span>
    </div>
  </div>
</div>

<div class="legend">
  <div class="leg-item"><div class="leg-dot" style="background:#4DA6FF"></div>Slack bus (ref angle)</div>
  <div class="leg-item"><div class="leg-dot" style="background:#9F7AEA"></div>Generator bus (PV)</div>
  <div class="leg-item"><div class="leg-dot" style="background:#5A7A99"></div>Load bus (PQ)</div>
  <div class="leg-item"><div class="leg-line" style="background:var(--teal);opacity:.9"></div>High attention edge (supervised)</div>
  <div class="leg-item"><div class="leg-line" style="background:var(--orange);opacity:.9"></div>High attention edge (PINN)</div>
</div>

<div class="physics-note">
  <strong>Reading this animation:</strong>
  Edge thickness and opacity reflect the attention weight α_ij at each correction step k.
  In the segmented softmax formulation, each destination bus normalises its incoming edge weights to sum to 1.0 —
  so a thick edge means that destination bus draws most of its context from that one source.
  Watch how the pattern <em>settles</em>: the PINN typically locks into a stable routing structure
  from the first step (physics structure is baked in), while the supervised model may shift between steps.
  Generator buses (purple) that are consistently the source of thick outgoing edges are the
  <strong>OPF information hubs</strong> — the nodes whose state most influences the correction network.
</div>

<script>
const DATA = {data_json};

const TYPE_COLOR = DATA.typeColor;
const EDGE_COLOR_SUP  = "#00CFA8";
const EDGE_COLOR_PINN = "#FF6B35";
const NODE_R = 11;

function initSVG(svgId, color) {{
  const svg = document.getElementById(svgId);
  // marker for arrowhead
  const defs = document.createElementNS("http://www.w3.org/2000/svg","defs");
  const marker = document.createElementNS("http://www.w3.org/2000/svg","marker");
  marker.setAttribute("id", svgId+"-arr");
  marker.setAttribute("markerWidth","6");
  marker.setAttribute("markerHeight","6");
  marker.setAttribute("refX","3");
  marker.setAttribute("refY","3");
  marker.setAttribute("orient","auto");
  const poly = document.createElementNS("http://www.w3.org/2000/svg","polygon");
  poly.setAttribute("points","0 0, 6 3, 0 6");
  poly.setAttribute("fill", color);
  poly.setAttribute("opacity","0.7");
  marker.appendChild(poly);
  defs.appendChild(marker);
  svg.appendChild(defs);

  // draw edges first (below nodes)
  const edgeGroup = document.createElementNS("http://www.w3.org/2000/svg","g");
  edgeGroup.setAttribute("id", svgId+"-edges");
  svg.appendChild(edgeGroup);
  DATA.edges.forEach(e => {{
    const path = document.createElementNS("http://www.w3.org/2000/svg","path");
    path.setAttribute("id", svgId+"-e"+e.id);
    path.setAttribute("d",
      `M${{e.x1}},${{e.y1}} Q${{e.cpx}},${{e.cpy}} ${{e.x2}},${{e.y2}}`
    );
    path.setAttribute("fill","none");
    path.setAttribute("stroke", color);
    path.setAttribute("stroke-width","1");
    path.setAttribute("stroke-opacity","0.08");
    path.setAttribute("stroke-linecap","round");
    path.setAttribute("marker-end", `url(#${{svgId+"-arr"}})`);
    edgeGroup.appendChild(path);
  }});

  // draw nodes
  DATA.nodes.forEach((n,i) => {{
    const g = document.createElementNS("http://www.w3.org/2000/svg","g");
    // glow ring for generator/slack
    if(n.type !== "load") {{
      const ring = document.createElementNS("http://www.w3.org/2000/svg","circle");
      ring.setAttribute("cx", n.x); ring.setAttribute("cy", n.y);
      ring.setAttribute("r", NODE_R+4);
      ring.setAttribute("fill", TYPE_COLOR[n.type]);
      ring.setAttribute("opacity","0.18");
      g.appendChild(ring);
    }}
    const circ = document.createElementNS("http://www.w3.org/2000/svg","circle");
    circ.setAttribute("cx", n.x); circ.setAttribute("cy", n.y);
    circ.setAttribute("r", NODE_R);
    circ.setAttribute("fill", TYPE_COLOR[n.type]);
    circ.setAttribute("stroke","#0D1B2A"); circ.setAttribute("stroke-width","2");
    circ.setAttribute("id", svgId+"-n"+i);
    g.appendChild(circ);
    const txt = document.createElementNS("http://www.w3.org/2000/svg","text");
    txt.setAttribute("x", n.x); txt.setAttribute("y", n.y+4);
    txt.setAttribute("text-anchor","middle");
    txt.setAttribute("font-size","9"); txt.setAttribute("font-weight","700");
    txt.setAttribute("fill","#E4EDF8");
    txt.textContent = n.label;
    g.appendChild(txt);
    svg.appendChild(g);
  }});
}}

function updateEdges(svgId, attn_step, color) {{
  DATA.edges.forEach(e => {{
    const w = attn_step[e.id];          // normalized 0-1
    const sw = 0.5 + w * 5.5;
    const so = 0.05 + w * 0.95;
    const path = document.getElementById(svgId+"-e"+e.id);
    if(path) {{
      path.setAttribute("stroke-width", sw.toFixed(2));
      path.setAttribute("stroke-opacity", so.toFixed(3));
    }}
  }});
  // pulse source nodes proportional to their total outgoing attention
  const srcSum = new Float64Array(14);
  const srcCount = new Float64Array(14);
  DATA.edges.forEach(e => {{
    srcSum[e.src] += attn_step[e.id];
    srcCount[e.src] += 1;
  }});
  DATA.nodes.forEach((n,i) => {{
    const circ = document.getElementById(svgId+"-n"+i);
    if(!circ) return;
    const mean = srcCount[i] > 0 ? srcSum[i]/srcCount[i] : 0;
    const r = (NODE_R + mean * 5).toFixed(1);
    circ.setAttribute("r", r);
    circ.setAttribute("opacity", (0.7 + mean*0.3).toFixed(2));
  }});
}}

// ── init ──────────────────────────────────────────────────────────────────
initSVG("sup-svg",  EDGE_COLOR_SUP);
initSVG("pinn-svg", EDGE_COLOR_PINN);

let step = 0;
const K_max = DATA.K_pinn;
let playing = false;
let timer = null;
const slider = document.getElementById("stepSlider");
slider.max = K_max - 1;

function renderStep(k) {{
  const k_sup  = Math.min(k, DATA.K_sup - 1);
  const k_pinn = Math.min(k, DATA.K_pinn - 1);
  updateEdges("sup-svg",  DATA.attn_sup[k_sup],  EDGE_COLOR_SUP);
  updateEdges("pinn-svg", DATA.attn_pinn[k_pinn], EDGE_COLOR_PINN);

  document.getElementById("stepDisplay").textContent =
    `Step ${{k_pinn+1}} / ${{DATA.K_pinn}} (PINN)  |  Step ${{k_sup+1}} / ${{DATA.K_sup}} (Supervised)`;
  document.getElementById("sup-sub").textContent =
    `Step k=${{k_sup+1}} of ${{DATA.K_sup}} · AC-OPF correction`;
  document.getElementById("pinn-sub").textContent =
    `Step k=${{k_pinn+1}} of ${{DATA.K_pinn}} · AC-OPF correction`;
  slider.value = k;
}}

function tick() {{
  step = (step + 1) % K_max;
  renderStep(step);
}}

document.getElementById("playBtn").addEventListener("click", () => {{
  playing = !playing;
  document.getElementById("playBtn").textContent = playing ? "⏸ Pause" : "▶ Play";
  document.getElementById("playBtn").classList.toggle("active", playing);
  if(playing) {{
    const ms = parseInt(document.getElementById("speedSlider").value);
    timer = setInterval(tick, ms);
  }} else {{
    clearInterval(timer);
  }}
}});

slider.addEventListener("input", () => {{
  step = parseInt(slider.value);
  renderStep(step);
}});

const speedSlider = document.getElementById("speedSlider");
speedSlider.addEventListener("input", () => {{
  document.getElementById("speedDisplay").textContent = speedSlider.value + " ms / step";
  if(playing) {{
    clearInterval(timer);
    timer = setInterval(tick, parseInt(speedSlider.value));
  }}
}});

// render initial frame
renderStep(0);
</script>
</body>
</html>"""
    return html


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attn_sup",  default="results/attention_supervised")
    p.add_argument("--attn_pinn", default="results/attention_pinn")
    p.add_argument("--out",       default="results/figures/animation/attention_animation.html")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("[anim] Loading attention tensors ...")
    aw_sup  = torch.load(os.path.join(args.attn_sup,  "attn_weights.pt"), weights_only=False)  # (S,K,E)
    aw_pinn = torch.load(os.path.join(args.attn_pinn, "attn_weights.pt"), weights_only=False)
    ei      = torch.load(os.path.join(args.attn_sup,  "edge_index.pt"),   weights_only=False)  # (E,2)

    S_sup, K_sup, E   = aw_sup.shape
    S_pinn, K_pinn, _ = aw_pinn.shape
    print(f"  Supervised: S={S_sup} K={K_sup} E={E}")
    print(f"  PINN:       S={S_pinn} K={K_pinn} E={E}")

    # mean over scenarios → (K, E) numpy
    mean_sup  = aw_sup.mean(dim=0).numpy()   # (K_sup, E)
    mean_pinn = aw_pinn.mean(dim=0).numpy()  # (K_pinn, E)

    edge_data = build_edge_data(ei)

    html = build_html(edge_data, mean_sup, mean_pinn, K_sup, K_pinn, E)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[anim] Saved -> {args.out}")
    print(f"  Open in browser to see interactive animation.")


if __name__ == "__main__":
    main()

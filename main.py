import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# -----------------------------
# Branding / constants
# -----------------------------
BG = "#dad7cd"
REQUIRED_COLS = ["flagged_columns", "level_0", "level_1"]  # kept as requested (not used in this tree view)

DESC = (
    "WeInterpret reveals the semantic structure of embedding spaces by assigning human-interpretable meaning "
    "to individual dimensions. It links each dimension of static and contextual embeddings to coherent sets "
    "of word senses, enabling systematic analysis, comparison, and controlled manipulation."
)

st.set_page_config(page_title="WeInterpret", page_icon="W", layout="wide")

st.markdown(
    f"""
    <style>
      header[data-testid="stHeader"] {{
        display: none;
      }}
      .block-container {{
        padding-top: 0.8rem;
      }}

      .stApp {{
        background-color: {BG};
      }}
      section.main > div {{
        background-color: {BG};
      }}

      /* readable text */
      .stApp, .stApp * {{
        color: #111111 !important;
      }}

      /* Title */
      h1 {{
        font-family: "Times New Roman", Times, serif !important;
        font-size: 56px !important;
        font-weight: 650 !important;
        margin-bottom: 0.4rem !important;
      }}

      /* Make all Streamlit inputs white */
      div[data-baseweb="input"] > div {{
        background-color: #ffffff !important;
        border-radius: 10px !important;
      }}
      div[data-baseweb="input"] input {{
        color: #111111 !important;
      }}
      div[data-baseweb="textarea"] > div {{
        background-color: #ffffff !important;
        border-radius: 10px !important;
      }}
      div[data-baseweb="textarea"] textarea {{
        color: #111111 !important;
      }}
      div[data-baseweb="select"] > div {{
        background-color: #ffffff !important;
        border-radius: 10px !important;
      }}
      div[data-baseweb="select"] * {{
        color: #111111 !important;
      }}
      span[data-baseweb="tag"] {{
        background-color: #f2f2f2 !important;
        color: #111111 !important;
        border: 1px solid rgba(0,0,0,0.15) !important;
      }}
      label, label * {{
        color: #111111 !important;
      }}
      .stCaption {{
        color: rgba(0,0,0,0.70) !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("WeInterpret")
st.write(DESC)


# -----------------------------
# Data -> filtered Newick
# -----------------------------
def expand_interpretations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        interp_raw = str(r["Interpretation"])
        parts = [p.strip() for p in interp_raw.split(":") if p.strip()]
        for p in parts:
            rows.append(
                {
                    "Cluster_Category": str(r["Cluster_Category"]).strip(),
                    "Interpretation": p,
                    "Dimension_Index": int(r["Dimension_Index"]),
                }
            )
    return pd.DataFrame(rows)


def escape_newick_label(label: str) -> str:
    return (
        str(label)
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace(":", "")
        .replace(";", "")
        .replace("\n", "_")
        .replace("\t", "_")
    )


def df_to_newick(df_expanded: pd.DataFrame, model_name: str) -> str:
    tree = defaultdict(lambda: defaultdict(set))  # tree[cluster][interp] -> set(dim)

    for _, r in df_expanded.iterrows():
        tree[r["Cluster_Category"]][r["Interpretation"]].add(r["Dimension_Index"])

    def dim_node(dim: int) -> str:
        return f"{escape_newick_label(dim)}:1.0"

    def interp_node(interp: str, dims: set[int]) -> str:
        children = ",".join(dim_node(d) for d in sorted(dims))
        return f"({children}){escape_newick_label(interp)}:1.0"

    def cluster_node(cluster: str, interps: dict) -> str:
        children = ",".join(
            interp_node(i, dims) for i, dims in sorted(interps.items(), key=lambda x: x[0])
        )
        return f"({children}){escape_newick_label(cluster)}:1.0"

    clusters = ",".join(
        cluster_node(c, interps) for c, interps in sorted(tree.items(), key=lambda x: x[0])
    )

    return f"({clusters}){escape_newick_label(model_name)};"


def parse_nonneg_int(s: str, default: int) -> int:
    s = (s or "").strip()
    if s == "":
        return default
    try:
        v = int(s)
        return v if v >= 0 else default
    except ValueError:
        return default


# -----------------------------
# Load data (fixed CSV)
# -----------------------------
CSV_PATH = Path("data/clustered_interpretations_mapping_all-MiniLM-L6-v2_emotion.csv")
if not CSV_PATH.exists():
    st.error(f"Fixed CSV not found: {CSV_PATH.as_posix()}")
    st.stop()

df_raw = pd.read_csv(CSV_PATH)
required = {"Interpretation", "Dimension_Index", "Cluster_Category"}
missing = required - set(df_raw.columns)
if missing:
    st.error(f"Missing columns for this visualiser: {sorted(missing)}")
    st.stop()

df = expand_interpretations(df_raw)
all_clusters = sorted(df["Cluster_Category"].dropna().unique().tolist())

# --- layout: filters left, plot right ---
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Filters")
    st.caption("PLACEHOLDER: short description of the filters goes here. (You will update this.)")

    clusters_sel = st.multiselect(
        "Cluster_Category",
        options=all_clusters,
        default=all_clusters[: min(5, len(all_clusters))],
    )

    # TEXT INPUTS (no +/- buttons)
    max_interpretations_txt = st.text_input(
        "Max interpretations per cluster (0 = no cap)",
        value="30",
    )
    max_dims_txt = st.text_input(
        "Max dimensions per interpretation (0 = no cap)",
        value="10",
    )

    max_interpretations = parse_nonneg_int(max_interpretations_txt, default=30)
    max_dims_per_interp = parse_nonneg_int(max_dims_txt, default=10)

    show_cluster_labels = st.checkbox("Show cluster labels", value=True)
    show_interpretation_labels = st.checkbox("Show interpretation labels", value=False)

# Apply filters
df_f = df.copy()

if clusters_sel:
    df_f = df_f[df_f["Cluster_Category"].isin(clusters_sel)]

# Cap interpretations per cluster by frequency
if max_interpretations and max_interpretations > 0:
    keep_parts = []
    for cluster, g in df_f.groupby("Cluster_Category"):
        top_interps = g["Interpretation"].value_counts().head(int(max_interpretations)).index
        keep_parts.append(g[g["Interpretation"].isin(top_interps)])
    df_f = pd.concat(keep_parts, ignore_index=True) if keep_parts else df_f.iloc[0:0]

# Cap dimensions per interpretation by frequency
if max_dims_per_interp and max_dims_per_interp > 0:
    keep_parts = []
    for (cluster, interp), g in df_f.groupby(["Cluster_Category", "Interpretation"]):
        top_dims = g["Dimension_Index"].value_counts().head(int(max_dims_per_interp)).index
        keep_parts.append(g[g["Dimension_Index"].isin(top_dims)])
    df_f = pd.concat(keep_parts, ignore_index=True) if keep_parts else df_f.iloc[0:0]

if df_f.empty:
    st.warning("No rows after filtering.")
    st.stop()

newick = df_to_newick(df_f, model_name="WeInterpret")

# JS-safe injections
newick_js = json.dumps(newick)
show_cluster_js = "true" if show_cluster_labels else "false"
show_interp_js = "true" if show_interpretation_labels else "false"

# IMPORTANT: not an f-string (so ${...} stays JS)
html = """
<div style="font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color:#111;">
  <div id="legend" style="font-size:12px; margin: 6px 0 10px 0; color:#111;"></div>
  <div id="mount" style="position:relative;">
    <div id="tooltip" style="
      position:absolute; display:none; pointer-events:none;
      background: rgba(255,255,255,0.98);
      color: #111;
      border: 1px solid rgba(0,0,0,0.18);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 12px;
      box-shadow: 0 10px 22px rgba(0,0,0,0.12);
      max-width: 340px;
    "></div>
  </div>
</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
  const NEWICK = __NEWICK_JSON__;
  const SHOW_CLUSTER_LABELS = __SHOW_CLUSTER__;
  const SHOW_INTERP_LABELS  = __SHOW_INTERP__;
  const SHOW_BRANCH_LENGTH = true;

  function parseNewick(a) {
    let stack = [], node = {}, tokens = a.split(/\\s*(;|\\(|\\)|,|:)\\s*/);
    for (let i = 0; i < tokens.length; i++) {
      const t = tokens[i];
      switch (t) {
        case "(": {
          const child = {};
          node.children = [child];
          stack.push(node);
          node = child;
          break;
        }
        case ",": {
          const sibling = {};
          stack[stack.length - 1].children.push(sibling);
          node = sibling;
          break;
        }
        case ")":
          node = stack.pop();
          break;
        case ":":
          break;
        default: {
          const prev = tokens[i - 1];
          if (prev === ")" || prev === "(" || prev === ",") node.name = t;
          else if (prev === ":") node.length = +t;
        }
      }
    }
    return node;
  }

  const width = 950;
  const innerRadius = 180;
  const outerRadius = width / 2;

  const legendEl = document.getElementById("legend");
  const mountEl = document.getElementById("mount");
  const tooltipEl = document.getElementById("tooltip");

  let pinned = null;
  const norm = (s) => String(s || "").replaceAll("_", " ");

  function formatTrail(d) {
    const a = d.ancestors().reverse().map(x => x.data.name).filter(Boolean);
    const cluster = a[1] ?? "";
    const interp = a[2] ?? "";
    const isLeaf = !d.children;
    const dim = isLeaf ? (a[a.length - 1] ?? "") : "";
    return { cluster, interp, dim, isLeaf, depth: d.depth };
  }

  function clear() {
    legendEl.innerHTML = "";
    mountEl.querySelectorAll("svg").forEach(n => n.remove());
  }

  function render() {
    clear();

    const root = d3.hierarchy(parseNewick(NEWICK));

    const clusterLayout = d3.cluster()
      .size([2 * Math.PI, innerRadius])
      .separation(() => 1);

    clusterLayout(root);

    if (SHOW_BRANCH_LENGTH) {
      let max = 0;
      root.each(d => {
        d._len = (d.parent ? d.parent._len : 0) + (d.data.length || 0);
        max = Math.max(max, d._len);
      });
      const k = (outerRadius - innerRadius) / (max || 1);
      root.each(d => d.radius = innerRadius + d._len * k);
    } else {
      root.each(d => d.radius = d.y);
    }

    const clusters = (root.children || []).map(d => d.data.name);
    const color = d3.scaleOrdinal().domain(clusters).range(d3.schemeTableau10);

    function setColor(d) {
      if (d.depth === 1) d.color = color(d.data.name);
      else d.color = d.parent ? d.parent.color : "#444";
      if (d.children) d.children.forEach(setColor);
    }
    setColor(root);

    // Legend
    clusters.forEach(c => {
      const row = document.createElement("div");
      row.style.display = "flex";
      row.style.alignItems = "center";
      row.style.gap = "6px";
      row.style.margin = "4px 0";

      const swatch = document.createElement("span");
      swatch.style.width = "14px";
      swatch.style.height = "14px";
      swatch.style.borderRadius = "3px";
      swatch.style.background = color(c);
      swatch.style.display = "inline-block";

      const label = document.createElement("span");
      label.style.color = "#111";
      label.textContent = norm(c);

      row.appendChild(swatch);
      row.appendChild(label);
      legendEl.appendChild(row);
    });

    const svg = d3.create("svg")
      .attr("viewBox", [-outerRadius, -outerRadius, width, width])
      .style("width", "100%")
      .style("max-width", `${width}px`)
      .style("height", "auto")
      .style("display", "block")
      .style("margin", "0 auto");

    const g = svg.append("g");

    const linkRadial = d3.linkRadial()
      .angle(d => d.x)
      .radius(d => d.radius);

    const linksSel = g.append("g")
      .attr("fill", "none")
      .attr("stroke-linecap", "round")
      .selectAll("path")
      .data(root.links())
      .join("path")
        .attr("stroke", d => d.target.color)
        .attr("stroke-width", d => d.source.depth === 0 ? 1.8 : 1.0)
        .attr("opacity", 0.9)
        .attr("d", linkRadial);

    // Dots
    const nodesSel = g.append("g")
      .selectAll("circle")
      .data(root.descendants())
      .join("circle")
        .attr("transform", d => `
          rotate(${d.x * 180 / Math.PI - 90})
          translate(${d.radius},0)
        `)
        .attr("r", d => (d.depth === 0 ? 2.6 : 2.0))
        .attr("fill", d => (d.depth === 0 ? "#111" : (d.color || "#666")))
        .attr("opacity", 0.95);

    // Labels (original positioning)
    let labelNodes = root.leaves(); // dimensions always
    if (SHOW_CLUSTER_LABELS) labelNodes = labelNodes.concat(root.descendants().filter(d => d.depth === 1));
    if (SHOW_INTERP_LABELS)  labelNodes = labelNodes.concat(root.descendants().filter(d => d.depth === 2));

    const labelsSel = g.append("g")
      .selectAll("text")
      .data(labelNodes)
      .join("text")
        .attr("dy", "0.31em")
        .attr("fill", "#111")
        .attr("transform", d => `
          rotate(${d.x * 180 / Math.PI - 90})
          translate(${d.radius + 6},0)
          rotate(${d.x >= Math.PI ? 180 : 0})
        `)
        .attr("text-anchor", d => d.x >= Math.PI ? "end" : "start")
        .attr("font-size", d => (d.depth === 3 ? 12 : 11))
        .attr("font-weight", d => (d.depth <= 2 ? 650 : 400))
        .text(d => norm(d.data.name));

    function pathSet(node) { return new Set(node.ancestors()); }

    function applyHighlight(nodeOrNull) {
      const ps = nodeOrNull ? pathSet(nodeOrNull) : null;

      linksSel
        .attr("opacity", l => (ps ? (ps.has(l.target) ? 1.0 : 0.15) : 0.9))
        .attr("stroke-width", l => {
          if (!ps) return (l.source.depth === 0 ? 1.8 : 1.0);
          return ps.has(l.target) ? 2.6 : 0.8;
        });

      labelsSel
        .attr("opacity", t => (ps ? (ps.has(t) ? 1.0 : 0.25) : 1.0))
        .attr("font-weight", t => (ps && ps.has(t) ? 700 : (t.depth <= 2 ? 650 : 400)));

      nodesSel
        .attr("opacity", n => (ps ? (ps.has(n) ? 1.0 : 0.12) : 0.95))
        .attr("r", n => {
          if (!ps) return (n.depth === 0 ? 2.6 : 2.0);
          return ps.has(n) ? (n.depth === 0 ? 3.0 : 2.6) : 1.6;
        });
    }

    function showTip(evt, d) {
      const { cluster, interp, dim, isLeaf, depth } = formatTrail(d);

      let rows = [];
      rows.push(`<div><b>Cluster</b>: ${norm(cluster)}</div>`);
      if (depth >= 2 && interp) rows.push(`<div><b>Interpretation</b>: ${norm(interp)}</div>`);
      if (isLeaf && dim) rows.push(`<div><b>Dimension</b>: ${norm(dim)}</div>`);
      rows.push(`<div style="margin-top:6px;opacity:0.7">Click to pin highlight</div>`);

      tooltipEl.innerHTML = rows.join("");
      tooltipEl.style.display = "block";

      const r = mountEl.getBoundingClientRect();
      tooltipEl.style.left = `${evt.clientX - r.left + 12}px`;
      tooltipEl.style.top  = `${evt.clientY - r.top + 12}px`;
    }

    function moveTip(evt) {
      if (tooltipEl.style.display === "none") return;
      const r = mountEl.getBoundingClientRect();
      tooltipEl.style.left = `${evt.clientX - r.left + 12}px`;
      tooltipEl.style.top  = `${evt.clientY - r.top + 12}px`;
    }

    function hideTip() { tooltipEl.style.display = "none"; }

    labelsSel
      .on("mousemove", (evt, d) => {
        if (pinned) return;
        applyHighlight(d);
        showTip(evt, d);
      })
      .on("mouseleave", () => {
        if (pinned) return;
        applyHighlight(null);
        hideTip();
      })
      .on("click", (evt, d) => {
        evt.stopPropagation();
        if (pinned === d) {
          pinned = null;
          applyHighlight(null);
          hideTip();
        } else {
          pinned = d;
          applyHighlight(d);
          showTip(evt, d);
        }
      });

    svg.on("mousemove", (evt) => moveTip(evt));
    svg.on("click", () => {
      pinned = null;
      applyHighlight(null);
      hideTip();
    });

    const zoom = d3.zoom()
      .scaleExtent([0.6, 8])
      .on("zoom", (evt) => g.attr("transform", evt.transform));

    svg.call(zoom);

    mountEl.appendChild(svg.node());
  }

  render();
</script>
"""

html = (
    html.replace("__NEWICK_JSON__", newick_js)
        .replace("__SHOW_CLUSTER__", show_cluster_js)
        .replace("__SHOW_INTERP__", show_interp_js)
)

with right:
    components.html(html, height=1100, scrolling=True)
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from Bio import Phylo
from Bio.Phylo.Newick import Clade, Tree


LIFE_TXT_URL = (
    "https://gist.githubusercontent.com/mbostock/c034d66572fd6bd6815a/raw/"
    "98778537e42f5605d9eddae5fba3329d969b813c/life.txt"
)

DOMAIN_NAMES = {"Bacteria", "Eukaryota", "Archaea"}
DOMAIN_COLOURS = {
    "Bacteria": "#1f77b4",
    "Eukaryota": "#ff7f0e",
    "Archaea": "#2ca02c",
    "Other": "#333333",
}


@dataclass(frozen=True)
class Style:
    radius: float = 1.0
    label_radius: float = 1.05
    link_width: float = 0.8
    trunk_width: float = 1.2
    label_size: float = 7.5
    node_size: float = 4.0
    background: str = "white"


def fetch_newick(url: str = LIFE_TXT_URL, timeout: int = 30) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def parse_newick(text: str) -> Tree:
    return Phylo.read(io.StringIO(text), "newick")


def build_parent_map(root: Clade) -> Dict[Clade, Optional[Clade]]:
    parent: Dict[Clade, Optional[Clade]] = {root: None}

    def rec(p: Clade) -> None:
        for ch in getattr(p, "clades", []) or []:
            parent[ch] = p
            rec(ch)

    rec(root)
    return parent


def leaves_in_order(root: Clade) -> List[Clade]:
    return list(root.get_terminals(order="preorder"))


def circular_mean(angles: List[float]) -> float:
    x = sum(math.cos(a) for a in angles) / len(angles)
    y = sum(math.sin(a) for a in angles) / len(angles)
    a = math.atan2(y, x)
    return a if a >= 0 else (a + 2.0 * math.pi)


def assign_angles(root: Clade) -> Dict[Clade, float]:
    leaves = leaves_in_order(root)
    n = len(leaves)
    if n == 0:
        return {root: 0.0}

    theta_leaf = {leaf: (2.0 * math.pi * i / n) for i, leaf in enumerate(leaves)}
    theta: Dict[Clade, float] = {}

    def rec(node: Clade) -> float:
        if node in theta_leaf:
            theta[node] = theta_leaf[node]
            return theta[node]
        children = getattr(node, "clades", []) or []
        if not children:
            theta[node] = 0.0
            return 0.0
        vals = [rec(ch) for ch in children]
        theta[node] = circular_mean(vals)
        return theta[node]

    rec(root)
    return theta


def topo_depth(node: Clade, parent: Dict[Clade, Optional[Clade]]) -> int:
    d = 0
    cur = node
    while True:
        p = parent.get(cur)
        if p is None:
            break
        d += 1
        cur = p
    return d


def cum_branch_length(node: Clade, parent: Dict[Clade, Optional[Clade]]) -> float:
    s = 0.0
    cur = node
    while True:
        p = parent.get(cur)
        if p is None:
            break
        bl = cur.branch_length if cur.branch_length is not None else 0.0
        s += float(bl)
        cur = p
    return s


def scale_radii(
    root: Clade,
    parent: Dict[Clade, Optional[Clade]],
    use_branch_lengths: bool,
    non_linear: str = "sqrt",
    max_radius: float = 1.0,
) -> Dict[Clade, float]:
    nodes = list(parent.keys())
    if use_branch_lengths:
        raw = {n: cum_branch_length(n, parent) for n in nodes}
    else:
        raw = {n: float(topo_depth(n, parent)) for n in nodes}

    max_raw = max(raw.values()) if raw else 1.0
    max_raw = max(max_raw, 1e-12)

    def transform(x: float) -> float:
        # Helps mimic D3 spacing and avoids over-compressing inner branches
        z = x / max_raw
        if non_linear == "log":
            # log1p scaled to [0,1]
            return math.log1p(9.0 * z) / math.log1p(9.0)
        if non_linear == "sqrt":
            return math.sqrt(z)
        return z

    return {n: transform(raw[n]) * max_radius for n in nodes}


def polar_to_xy(theta: float, r: float) -> Tuple[float, float]:
    # rotate so theta=0 at 12 o'clock
    a = theta - (math.pi / 2.0)
    return (r * math.cos(a), r * math.sin(a))


def label_text(node: Clade) -> str:
    return (node.name or "").replace("_", " ")


def domain_of(node: Clade, parent: Dict[Clade, Optional[Clade]]) -> str:
    cur: Optional[Clade] = node
    while cur is not None:
        nm = (cur.name or "").strip()
        if nm in DOMAIN_NAMES:
            return nm
        cur = parent.get(cur)
    return "Other"


def bezier_link(
    x0: float, y0: float, x1: float, y1: float, curvature: float = 0.18
) -> Path:
    """
    Cubic Bézier link between (x0,y0) and (x1,y1).
    Control points are pulled towards the origin to mimic radial bundle feel.
    """
    # Control points: slightly towards origin from each endpoint
    c0x, c0y = (1.0 - curvature) * x0, (1.0 - curvature) * y0
    c1x, c1y = (1.0 - curvature) * x1, (1.0 - curvature) * y1

    verts = [(x0, y0), (c0x, c0y), (c1x, c1y), (x1, y1)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
    return Path(verts, codes)


def draw_tree(
    use_branch_lengths: bool,
    out: Optional[str],
    style: Style,
    dpi: int,
    show: bool,
) -> None:
    newick = fetch_newick()
    tree = parse_newick(newick)
    root = tree.root

    parent = build_parent_map(root)
    theta = assign_angles(root)
    r = scale_radii(
        root,
        parent,
        use_branch_lengths=use_branch_lengths,
        non_linear="sqrt",
        max_radius=style.radius,
    )

    # Precompute positions
    xy: Dict[Clade, Tuple[float, float]] = {n: polar_to_xy(theta[n], r[n]) for n in parent.keys()}

    # Figure
    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.set_facecolor(style.background)
    fig.patch.set_facecolor(style.background)
    ax.axis("off")

    # Draw links (parent -> child), coloured by the child's domain
    # Draw trunk in darker width for readability
    for child, p in parent.items():
        if p is None:
            continue
        (x0, y0) = xy[p]
        (x1, y1) = xy[child]

        dom = domain_of(child, parent)
        col = DOMAIN_COLOURS.get(dom, DOMAIN_COLOURS["Other"])
        lw = style.link_width

        # Root "trunk" feels better slightly thicker
        if parent.get(p) is None:
            lw = style.trunk_width

        path = bezier_link(x0, y0, x1, y1, curvature=0.22)
        patch = PathPatch(path, facecolor="none", edgecolor=col, lw=lw, alpha=0.95, capstyle="round")
        ax.add_patch(patch)

    # Nodes (small)
    xs = [xy[n][0] for n in parent.keys()]
    ys = [xy[n][1] for n in parent.keys()]
    ax.scatter(xs, ys, s=style.node_size, c="#0b0b0b", alpha=0.55, linewidths=0)

    # Leaf labels: rotated and flipped for readability
    leaves = leaves_in_order(root)
    for leaf in leaves:
        txt = label_text(leaf)
        if not txt:
            continue
        ang = theta[leaf]
        lx, ly = polar_to_xy(ang, style.label_radius)

        deg = (ang * 180.0 / math.pi) - 90.0  # tangential orientation
        # Flip on left side so text is not upside down
        if 90.0 < deg % 360.0 < 270.0:
            deg += 180.0
            ha = "right"
        else:
            ha = "left"

        ax.text(
            lx,
            ly,
            txt,
            rotation=deg,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            fontsize=style.label_size,
            color="#111111",
        )

    # Legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=DOMAIN_COLOURS["Bacteria"], label="Bacteria"),
        mpatches.Patch(color=DOMAIN_COLOURS["Eukaryota"], label="Eukaryota"),
        mpatches.Patch(color=DOMAIN_COLOURS["Archaea"], label="Archaea"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10)

    title = "Tree of Life (matplotlib)  "
    title += "(branch length ON)" if use_branch_lengths else "(branch length OFF)"
    fig.suptitle(title, y=0.98, fontsize=14)

    # Tight bounds
    pad = 0.12
    ax.set_xlim(-1.0 - pad, 1.0 + pad)
    ax.set_ylim(-1.0 - pad, 1.0 + pad)

    if out:
        fig.savefig(out, bbox_inches="tight")
        print(f"Saved: {out}")

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch-length", action="store_true", help="Use branch lengths for radius")
    ap.add_argument("--out", type=str, default="tree_of_life.png", help="Output file (png/svg/pdf)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-show", action="store_true", help="Do not open a window")
    args = ap.parse_args()

    draw_tree(
        use_branch_lengths=args.branch_length,
        out=args.out,
        dpi=args.dpi,
        show=(not args.no_show),
        style=Style(),
    )


if __name__ == "__main__":
    main()
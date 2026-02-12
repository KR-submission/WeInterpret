# plot.py
# Sketch-style plot:
# - each point = a word (from level_0/level_1 in peculiar CSV)
# - DBSCAN clusters words (independent of selected dimensions)
# - cluster label = representative word (closest to centroid in 2D)
# - green boundaries = for each dimension d, take its associated words and
#   split them into local components (DBSCAN in 2D), then draw one boundary per component
# - supports PCA / t-SNE / UMAP for the 2D layout

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from transformers import AutoModel, AutoTokenizer

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

# Optional UMAP
try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# Optional ConvexHull (recommended)
try:
    from scipy.spatial import ConvexHull  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_flagged_columns(cell) -> List[int]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    return [int(x) for x in re.findall(r"\d+", str(cell))]


def build_S_from_peculiar(df: pd.DataFrame) -> List[str]:
    """
    S := unique words appearing in level_0 and level_1, preserving first-seen order (case-insensitive).
    """
    for col in ("level_0", "level_1", "flagged_columns"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in peculiar CSV.")

    words: List[str] = []
    seen = set()

    for _, row in df.iterrows():
        for col in ("level_0", "level_1"):
            w = str(row[col]).strip()
            if not w or w.lower() == "nan":
                continue
            key = w.lower()
            if key not in seen:
                seen.add(key)
                words.append(w)

    if not words:
        raise ValueError("No words found in level_0/level_1.")
    return words


def dim_to_wordset(df: pd.DataFrame, S_words: Sequence[str]) -> Dict[int, List[str]]:
    """
    Build mapping dim -> list of words in S associated with dim according to flagged_columns rows.
    A dim is associated with words that appear as level_0/level_1 in rows where dim ∈ flagged_columns.
    """
    S_lower = {w.lower() for w in S_words}
    dim_hits: Dict[int, set] = defaultdict(set)

    for _, row in df.iterrows():
        dims = parse_flagged_columns(row["flagged_columns"])
        if not dims:
            continue

        w0 = str(row["level_0"]).strip()
        w1 = str(row["level_1"]).strip()

        ws = []
        if w0 and w0.lower() in S_lower:
            ws.append(w0.lower())
        if w1 and w1.lower() in S_lower:
            ws.append(w1.lower())

        if not ws:
            continue

        for d in dims:
            for w in ws:
                dim_hits[d].add(w)

    # preserve S order per dim
    out: Dict[int, List[str]] = {}
    for d, hitset in dim_hits.items():
        out[d] = [w for w in S_words if w.lower() in hitset]
    return out


# -----------------------------
# Embeddings
# -----------------------------

def is_special_token_id(token_id: int, tokenizer: AutoTokenizer) -> bool:
    return token_id in set(filter(lambda x: x is not None, [
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    ]))


@torch.inference_mode()
def embed_words_transformer(
    words: Sequence[str],
    model_name: str,
    device: str,
    max_length: int = 16,
) -> np.ndarray:
    """
    One vector per word/phrase: mean pooling over subword tokens excluding special tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)

    embs: List[np.ndarray] = []
    for w in words:
        enc = tokenizer(
            w,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        ).to(device)

        out = model(**enc)
        hidden = out.last_hidden_state.squeeze(0)  # [seq_len, dim]
        input_ids = enc["input_ids"].squeeze(0)

        mask = torch.ones_like(input_ids, dtype=torch.bool)
        for i, tid in enumerate(input_ids.tolist()):
            if is_special_token_id(tid, tokenizer):
                mask[i] = False

        pooled = hidden[mask].mean(dim=0) if mask.sum().item() > 0 else hidden.mean(dim=0)
        embs.append(pooled.detach().cpu().numpy())

    return np.vstack(embs)


# -----------------------------
# Clustering + 2D reduction
# -----------------------------

def cluster_words_dbscan_pca(
    X: np.ndarray,
    eps: float,
    min_samples: int,
    pca_dim: int,
    seed: int,
) -> np.ndarray:
    """
    DBSCAN on words in PCA(pca_dim) space. This is independent of any selected dimension.
    """
    Xn = normalize(X)
    k = min(pca_dim, Xn.shape[1], max(2, Xn.shape[0] - 1))
    Xr = PCA(n_components=k, random_state=seed).fit_transform(Xn)
    return DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(Xr)


def reduce_2d(X: np.ndarray, method: str, seed: int) -> np.ndarray:
    m = method.lower()
    if m == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if m == "tsne":
        perplexity = min(30, max(5, (X.shape[0] - 1) // 3))
        return TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            metric="cosine",
        ).fit_transform(X)
    if m == "umap":
        if not HAS_UMAP:
            raise RuntimeError("UMAP requested but umap-learn is not installed. Install: pip install umap-learn")
        reducer = umap.UMAP(
            n_components=2,
            random_state=seed,
            metric="cosine",
            n_neighbors=min(15, max(5, X.shape[0] // 3)),
            min_dist=0.1,
        )
        return reducer.fit_transform(X)
    raise ValueError("Reducer must be one of: pca, tsne, umap")


# -----------------------------
# Cluster labels (representative word)
# -----------------------------

def representative_word(words: Sequence[str], Z: np.ndarray, idx: np.ndarray) -> str:
    """
    Label cluster by word closest to cluster centroid in 2D.
    """
    pts = Z[idx]
    centroid = pts.mean(axis=0, keepdims=True)
    d = np.linalg.norm(pts - centroid, axis=1)
    return str(words[idx[np.argmin(d)]])


def build_cluster_label_map(words: Sequence[str], Z: np.ndarray, cluster_ids: np.ndarray) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for cid in sorted(set(cluster_ids.tolist())):
        if cid == -1:
            continue
        idx = np.where(cluster_ids == cid)[0]
        if idx.size == 0:
            continue
        out[cid] = representative_word(words, Z, idx)
    return out


# -----------------------------
# Green dimension boundaries (split into local components)
# -----------------------------

def draw_boundary(ax, pts: np.ndarray, alpha: float = 0.12) -> np.ndarray:
    """
    Draw a green boundary around pts.
    Returns the centre point used for label placement.
    """
    if pts.shape[0] < 2:
        return pts.mean(axis=0)

    if HAS_SCIPY and pts.shape[0] >= 3:
        hull = ConvexHull(pts)
        poly = pts[hull.vertices]
        ax.fill(poly[:, 0], poly[:, 1], facecolor="green", alpha=alpha, edgecolor="green", linewidth=2.0)
        return poly.mean(axis=0)

    # Ellipse fallback (rough)
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T) + 1e-6 * np.eye(2)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    t = np.linspace(0, 2*np.pi, 200)
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)
    radii = 2.0 * np.sqrt(vals)
    ellipse = (vecs @ (radii[:, None] * circle)).T + c
    ax.fill(ellipse[:, 0], ellipse[:, 1], facecolor="green", alpha=alpha, edgecolor="green", linewidth=2.0)
    return c


def split_dim_into_components(
    pts: np.ndarray,
    eps: float,
    min_samples: int,
) -> List[np.ndarray]:
    """
    Split a dimension's points into local connected components in 2D using DBSCAN.
    This prevents the long "bridge" hull spanning multiple distant clusters.
    """
    if pts.shape[0] < 3:
        return []

    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(pts)

    components: List[np.ndarray] = []
    for cid in sorted(set(labels.tolist())):
        if cid == -1:
            continue
        comp = pts[labels == cid]
        if comp.shape[0] >= 3:
            components.append(comp)
    return components


# -----------------------------
# Plot (sketch style)
# -----------------------------

def plot_sketch(
    Z: np.ndarray,
    words: Sequence[str],
    cluster_ids: np.ndarray,
    dim_words: Dict[int, List[str]],
    dims_to_draw: Sequence[int],
    dim_comp_eps: float,
    dim_comp_min_samples: int,
    title: str,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    words = list(words)
    w2i = {w.lower(): i for i, w in enumerate(words)}

    # Colour per word-cluster id
    unique_cids = sorted(set(cluster_ids.tolist()))
    non_noise = [cid for cid in unique_cids if cid != -1]
    cmap = plt.get_cmap("tab20")
    cid_to_colour = {cid: cmap(i % cmap.N) for i, cid in enumerate(non_noise)}

    # Points
    for cid in unique_cids:
        idx = np.where(cluster_ids == cid)[0]
        if cid == -1:
            ax.scatter(Z[idx, 0], Z[idx, 1], s=18, alpha=0.20, color="grey")
        else:
            ax.scatter(Z[idx, 0], Z[idx, 1], s=22, alpha=0.85, color=cid_to_colour[cid])

    # Cluster labels (representative word)
    cluster_label_map = build_cluster_label_map(words, Z, cluster_ids)
    for cid, lab in cluster_label_map.items():
        idx = np.where(cluster_ids == cid)[0]
        c = Z[idx].mean(axis=0)
        ax.text(c[0], c[1], lab, fontsize=12, fontweight="bold", color="black")

    # Dimension blobs (split into components)
    for d in dims_to_draw:
        ws = dim_words.get(d, [])
        pts_idx = [w2i[w.lower()] for w in ws if w.lower() in w2i]
        if len(pts_idx) < 3:
            continue

        pts = Z[np.array(pts_idx)]
        comps = split_dim_into_components(pts, eps=dim_comp_eps, min_samples=dim_comp_min_samples)

        # If DBSCAN finds nothing (too strict), fall back to one boundary if enough points
        if not comps and pts.shape[0] >= 3:
            comps = [pts]

        for comp in comps:
            centre = draw_boundary(ax, comp, alpha=0.12)
            ax.text(centre[0], centre[1], str(d), fontsize=12, fontweight="bold", color="green")

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------

@dataclass
class Config:
    model: str
    peculiar_csv: str
    reducer: str
    out_dir: str
    seed: int
    device: str

    # word clustering (DBSCAN on PCA space)
    word_dbscan_eps: float
    word_dbscan_min_samples: int
    word_dbscan_pca_dim: int

    # which dims to draw
    dims: Optional[str]
    top_dims: int
    min_words_per_dim: int

    # dim-component splitting in 2D
    dim_comp_eps: float
    dim_comp_min_samples: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="distilroberta-base")
    parser.add_argument("--peculiar_csv", type=str, required=True)

    parser.add_argument("--reducer", type=str, default="pca", choices=["pca", "tsne", "umap"])
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    # Word DBSCAN (independent of dims)
    parser.add_argument("--word_dbscan_eps", type=float, default=0.7, help="eps for word DBSCAN in PCA space (try 0.5-1.2)")
    parser.add_argument("--word_dbscan_min_samples", type=int, default=6)
    parser.add_argument("--word_dbscan_pca_dim", type=int, default=50)

    # Dimensions to draw
    parser.add_argument("--dims", type=str, default=None, help="Comma-separated dims to draw, e.g. '1,2,14,39'")
    parser.add_argument("--top_dims", type=int, default=8, help="If --dims not set: draw top-K dims by coverage")
    parser.add_argument("--min_words_per_dim", type=int, default=6, help="Only draw dims that cover >= this many words")

    # Split each dim into local components in 2D (fixes long bridges)
    parser.add_argument("--dim_comp_eps", type=float, default=0.9, help="eps for splitting dim blobs in 2D (UMAP scale differs from PCA/TSNE)")
    parser.add_argument("--dim_comp_min_samples", type=int, default=3)

    args = parser.parse_args()

    cfg = Config(
        model=args.model,
        peculiar_csv=args.peculiar_csv,
        reducer=args.reducer,
        out_dir=args.out_dir,
        seed=args.seed,
        device=args.device,
        word_dbscan_eps=args.word_dbscan_eps,
        word_dbscan_min_samples=args.word_dbscan_min_samples,
        word_dbscan_pca_dim=args.word_dbscan_pca_dim,
        dims=args.dims,
        top_dims=args.top_dims,
        min_words_per_dim=args.min_words_per_dim,
        dim_comp_eps=args.dim_comp_eps,
        dim_comp_min_samples=args.dim_comp_min_samples,
    )

    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    df = pd.read_csv(cfg.peculiar_csv)
    S_words = build_S_from_peculiar(df)
    print(f"[info] |S| = {len(S_words)} words")

    dim_words = dim_to_wordset(df, S_words)

    # Choose which dims to draw
    if cfg.dims:
        dims_to_draw = [int(x.strip()) for x in cfg.dims.split(",") if x.strip()]
    else:
        candidates = [(d, len(ws)) for d, ws in dim_words.items() if len(ws) >= cfg.min_words_per_dim]
        candidates.sort(key=lambda x: x[1], reverse=True)
        dims_to_draw = [d for d, _ in candidates[: cfg.top_dims]]

    print(f"[info] dims_to_draw = {dims_to_draw}")

    # Embeddings
    X = embed_words_transformer(S_words, cfg.model, cfg.device)

    # Word clustering (independent of dims)
    word_cluster_ids = cluster_words_dbscan_pca(
        X,
        eps=cfg.word_dbscan_eps,
        min_samples=cfg.word_dbscan_min_samples,
        pca_dim=cfg.word_dbscan_pca_dim,
        seed=cfg.seed,
    )

    # 2D layout for plotting
    Z = reduce_2d(X, cfg.reducer, seed=cfg.seed)

    out_path = os.path.join(cfg.out_dir, f"sketch_{cfg.model.replace('/', '_')}_{cfg.reducer}.png")
    title = f"{cfg.model} | reducer={cfg.reducer} | word-DBSCAN(PCA{cfg.word_dbscan_pca_dim})"

    plot_sketch(
        Z=Z,
        words=S_words,
        cluster_ids=word_cluster_ids,
        dim_words=dim_words,
        dims_to_draw=dims_to_draw,
        dim_comp_eps=cfg.dim_comp_eps,
        dim_comp_min_samples=cfg.dim_comp_min_samples,
        title=title,
        out_path=out_path,
    )
    print(f"[info] saved: {out_path}")


if __name__ == "__main__":
    main()
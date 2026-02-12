import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Optional


def csv_to_newick(
    csv_path: str | Path,
    model_name: str,
    out_path: str | Path | None = None,
    max_clusters: Optional[int] = None,
    max_interpretations: Optional[int] = None,
    max_dimensions: Optional[int] = None,
) -> str:
    """
    Convert CSV with columns:
      Interpretation, Dimension_Index, Cluster_Category

    into a Newick tree with hierarchy:
      model -> Cluster_Category -> Interpretation -> Dimension_Index

    Parameters
    ----------
    max_clusters : int or None
        Keep only top-K Cluster_Category by frequency.

    max_interpretations : int or None
        Keep only top-M Interpretation per Cluster_Category by frequency.

    max_dimensions : int or None
        Keep only top-N Dimension_Index per Interpretation by frequency.
    """

    df = pd.read_csv(csv_path)

    required = {"Interpretation", "Dimension_Index", "Cluster_Category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # ------------------------------------------------------------
    # 1. Expand Interpretation by splitting on ':'
    # ------------------------------------------------------------
    rows = []
    for _, r in df.iterrows():
        interpretations = [x.strip() for x in str(r["Interpretation"]).split(":")]
        for interp in interpretations:
            rows.append(
                {
                    "Cluster_Category": str(r["Cluster_Category"]).strip(),
                    "Interpretation": interp,
                    "Dimension_Index": int(r["Dimension_Index"]),
                }
            )

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # 2. Limit Cluster_Category
    # ------------------------------------------------------------
    if max_clusters is not None:
        top_clusters = (
            df["Cluster_Category"]
            .value_counts()
            .head(max_clusters)
            .index
        )
        df = df[df["Cluster_Category"].isin(top_clusters)]

    # ------------------------------------------------------------
    # 3. Limit Interpretation per Cluster_Category
    # ------------------------------------------------------------
    if max_interpretations is not None:
        keep_rows = []
        for cluster, g in df.groupby("Cluster_Category"):
            top_interps = (
                g["Interpretation"]
                .value_counts()
                .head(max_interpretations)
                .index
            )
            keep_rows.append(g[g["Interpretation"].isin(top_interps)])
        df = pd.concat(keep_rows, ignore_index=True)

    # ------------------------------------------------------------
    # 4. Limit Dimension_Index per Interpretation
    # ------------------------------------------------------------
    if max_dimensions is not None:
        keep_rows = []
        for (cluster, interp), g in df.groupby(
            ["Cluster_Category", "Interpretation"]
        ):
            top_dims = (
                g["Dimension_Index"]
                .value_counts()
                .head(max_dimensions)
                .index
            )
            keep_rows.append(g[g["Dimension_Index"].isin(top_dims)])
        df = pd.concat(keep_rows, ignore_index=True)

    # ------------------------------------------------------------
    # 5. Build nested structure
    # ------------------------------------------------------------
    tree = defaultdict(lambda: defaultdict(set))
    # tree[cluster][interpretation] -> set(dim)

    for _, r in df.iterrows():
        tree[r["Cluster_Category"]][r["Interpretation"]].add(
            r["Dimension_Index"]
        )

    # ------------------------------------------------------------
    # 6. Emit Newick
    # ------------------------------------------------------------
    def escape(label: str) -> str:
        return (
            label.replace(" ", "_")
                 .replace("(", "")
                 .replace(")", "")
                 .replace(",", "")
                 .replace(":", "")
        )

    def dim_node(dim: int) -> str:
        return f"{escape(str(dim))}:1.0"

    def interpretation_node(name: str, dims: set[int]) -> str:
        children = ",".join(
            dim_node(d) for d in sorted(dims)
        )
        return f"({children}){escape(name)}:1.0"

    def cluster_node(name: str, interps: dict) -> str:
        children = ",".join(
            interpretation_node(i, dims)
            for i, dims in sorted(interps.items())
        )
        return f"({children}){escape(name)}:1.0"

    clusters = ",".join(
        cluster_node(c, interps)
        for c, interps in sorted(tree.items())
    )

    newick = f"({clusters}){escape(model_name)};"

    if out_path is not None:
        Path(out_path).write_text(newick)

    return newick


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Input CSV")
    ap.add_argument("--model", required=True, help="Model name (top-level node)")
    ap.add_argument("--out", default="tree.newick", help="Output Newick file")

    ap.add_argument(
        "--max-clusters",
        type=int,
        default=None,
        help="Max number of Cluster_Category values",
    )
    ap.add_argument(
        "--max-interpretations",
        type=int,
        default=None,
        help="Max number of Interpretation per cluster",
    )
    ap.add_argument(
        "--max-dimensions",
        type=int,
        default=None,
        help="Max number of Dimension_Index per interpretation",
    )

    args = ap.parse_args()

    csv_to_newick(
        args.csv,
        args.model,
        out_path=args.out,
        max_clusters=args.max_clusters,
        max_interpretations=args.max_interpretations,
        max_dimensions=args.max_dimensions,
    )

    print("Newick written to:", args.out)
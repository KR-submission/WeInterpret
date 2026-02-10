import streamlit as st, os, pathlib

st.write("CWD:", os.getcwd())
st.write("Files in CWD:", os.listdir("."))

import re
import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


# -----------------------------
# Config
# -----------------------------
DATA_DIR = Path("emotion")   
DEFAULT_DIM = 42
BG = "#dad7cd"
REQUIRED_COLS = ["flagged_columns", "level_0", "level_1"]

DESC = (
    "WeInterpret reveals the semantic structure of embedding spaces by assigning human-interpretable meaning "
    "to individual dimensions. It links each dimension of static and contextual embeddings to coherent sets "
    "of word senses, enabling systematic analysis, comparison, and controlled manipulation."
)

st.set_page_config(page_title="WeInterpret", page_icon="W", layout="wide")


# -----------------------------
# CSS
# -----------------------------
st.markdown(
    f"""
    <style>
      /* Remove Streamlit header (top white band) */
      header[data-testid="stHeader"] {{
        display: none;
      }}
      .block-container {{
        padding-top: 1.2rem;
      }}

      /* Background */
      .stApp {{
        background-color: {BG};
      }}
      section.main > div {{
        background-color: {BG};
      }}

      /* Title */
      .weinterpret-title {{
        font-family: "Times New Roman", Times, serif;
        font-size: 44px;
        font-weight: 600;
        margin: 0 0 0.35rem 0;
        line-height: 1.05;
      }}

      /* Subtitle */
      .weinterpret-subtitle {{
        max-width: 980px;
        font-size: 1.05rem;
        line-height: 1.45;
        opacity: 0.9;
        margin: 0 0 1.2rem 0;
      }}

      /* Style Streamlit bordered containers as cards */
      div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(0,0,0,0.08) !important;
        border-radius: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        padding: 12px 16px;
      }}
      div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        padding-top: 0.2rem;
      }}

      /* Inputs */
      div[data-baseweb="select"] > div {{
        background: rgba(255,255,255,0.92);
      }}
      input {{
        background: rgba(255,255,255,0.92) !important;
      }}

      /* Tighten headings inside containers */
      h1, h2, h3, h4, h5, h6, p {{
        margin-top: 0 !important;
      }}
      .muted {{
        opacity: 0.75;
      }}
      /* Force readable text colour everywhere */
        .stApp, .stApp * {
        color: #1f1f1f;
        }

        /* Keep muted helper, but don’t wash out important text */
        .weinterpret-subtitle, .muted {
        color: rgba(31,31,31,0.82) !important;
        opacity: 1 !important;   /* remove compounded opacity */
        }

        /* Make cards solid for consistent contrast */
        div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #f7f4ee;   /* or plain white */
        }
    </style>
    """,
    
    unsafe_allow_html=True,
)

# Title + description
st.markdown('<div class="weinterpret-title">WeInterpret</div>', unsafe_allow_html=True)
st.markdown(f"<div class='weinterpret-subtitle'>{DESC}</div>", unsafe_allow_html=True)


# -----------------------------
# Helpers
# -----------------------------
def list_data_files(data_dir: Path) -> List[Path]:
    exts = {".csv", ".tsv", ".parquet"}
    if not data_dir.exists():
        return []
    files = [p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def model_name_from_file(path: Path) -> str:
    """
    Pattern:
      peculiar_df_<MODEL>_<DATASET>.csv

    Examples:
      peculiar_df_all-mpnet-base-v2_emotion.csv -> all-mpnet-base-v2
      peculiar_df_word2vec_emotion.csv -> word2vec
    """
    stem = path.stem
    prefix = "peculiar_df_"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    if "_" in stem:
        stem = stem.rsplit("_", 1)[0]  # drop trailing dataset token
    return stem


def read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def parse_flagged_columns(x: Any) -> List[Any]:
    """Accept list/tuple/set, python-list strings, or comma-separated strings."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, set)):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, (list, tuple, set)):
                    return list(v)
            except Exception:
                pass
        return [t.strip() for t in s.split(",") if t.strip()]
    return [x]


def dim_in_token(tok: Any, dim: int) -> bool:
    if isinstance(tok, int):
        return tok == dim
    s = str(tok).strip()
    if s == str(dim):
        return True
    return bool(re.search(rf"(?<!\d){dim}(?!\d)", s))


def infer_dim_max(df: pd.DataFrame) -> Optional[int]:
    """Infer maximum dim index from flagged_columns (used only for optional clamping)."""
    if "flagged_columns" not in df.columns:
        return None
    max_dim = None
    for v in df["flagged_columns"].dropna().head(8000):
        for t in parse_flagged_columns(v):
            m = re.search(r"(\d+)", str(t))
            if not m:
                continue
            d = int(m.group(1))
            max_dim = d if max_dim is None else max(max_dim, d)
    return max_dim


def words_for_dim(df: pd.DataFrame, dim: int) -> List[str]:
    flags = df["flagged_columns"].apply(parse_flagged_columns)
    mask = flags.apply(lambda items: any(dim_in_token(t, dim) for t in items))
    sub = df.loc[mask, ["level_0", "level_1"]]

    vals = pd.concat([sub["level_0"], sub["level_1"]], ignore_index=True).dropna()
    words = [str(w).strip() for w in vals.tolist() if str(w).strip()]
    return sorted(set(words), key=lambda s: s.lower())


def model_info_text(model: str) -> str:
    """Model-centric info only (no dataset/rows/dims)."""
    m = model.lower()

    if "word2vec" in m:
        return (
            "**Word2Vec** is a static word embedding model trained with predictive objectives "
            "(CBOW/Skip-gram). It produces one vector per token and is commonly used as a strong, "
            "efficient baseline for lexical semantics."
        )
    if "glove" in m:
        return (
            "**GloVe** is a static embedding method learned from global co-occurrence statistics. "
            "It is a classic baseline that captures broad semantic regularities with a simple training setup."
        )
    if "fasttext" in m:
        return (
            "**FastText** is a static embedding model that incorporates subword information. "
            "It is typically more robust to rare words and morphology than purely token-based models."
        )

    if "mpnet" in m:
        return (
            "**MPNet-based sentence embeddings** are Transformer encoders tuned for semantic similarity "
            "and retrieval. They are typically strong for clustering, semantic search, and STS-style tasks."
        )
    if "minilm" in m:
        return (
            "**MiniLM sentence embeddings** are compact Transformer encoders optimised for speed. "
            "They are often used when you want good retrieval quality at lower latency."
        )
    if "distilroberta" in m:
        return (
            "**DistilRoBERTa-based embeddings** use a distilled Transformer encoder. "
            "They provide contextual representations with lower compute than full-size encoders."
        )
    if "roberta" in m:
        return (
            "**RoBERTa-based embeddings** come from a Transformer encoder trained with masked-language objectives. "
            "They produce contextual representations and are widely used as general-purpose backbones."
        )
    if "multi-qa" in m:
        return (
            "**Multi-QA sentence embeddings** are tuned for question-answer retrieval settings. "
            "They tend to perform well when queries and documents have asymmetric roles."
        )
    if "paraphrase" in m:
        return (
            "**Paraphrase-tuned sentence embeddings** are trained to bring paraphrases closer in embedding space. "
            "They are commonly used for similarity, clustering, and duplicate detection."
        )

    return (
        "Embedding model. If you want a more specific description, add a rule matching this model name pattern."
    )


# -----------------------------
# Load entries
# -----------------------------
files = list_data_files(DATA_DIR)
if not files:
    st.error(f"No .csv/.tsv/.parquet files found in {DATA_DIR.resolve()}")
    st.stop()

entries: List[str] = []
name_to_path: Dict[str, Path] = {}
seen: Dict[str, int] = {}

for fp in files:
    base = model_name_from_file(fp)
    if base in seen:
        seen[base] += 1
        name = f"{base} ({seen[base]})"
    else:
        seen[base] = 1
        name = base
    entries.append(name)
    name_to_path[name] = fp

entries = sorted(entries, key=lambda s: s.lower())


# -----------------------------
# Layout
# -----------------------------
left, middle, right = st.columns([1.1, 1.7, 1.2], gap="large")

with left:
    with st.container(border=True):
        st.markdown("#### Controls")
        selected_model = st.selectbox("Models", options=entries, index=0)
        dim_raw = st.text_input("Dimension", value=str(DEFAULT_DIM), help="Enter an integer dimension index.")

# Validate dimension
try:
    selected_dim = int(dim_raw.strip())
    if selected_dim < 0:
        raise ValueError
except Exception:
    selected_dim = DEFAULT_DIM
    with left:
        st.error(f"Dimension must be a non-negative integer. Using default {DEFAULT_DIM}.")

# Read selected file
fp = name_to_path[selected_model]
df = read_table(fp)

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Selected file '{fp.name}' is missing required columns: {missing}")
    st.stop()

# Optional clamping to inferred max dimension index
max_dim = infer_dim_max(df)
effective_dim = selected_dim
if max_dim is not None and selected_dim > max_dim:
    effective_dim = max_dim
    with left:
        st.warning(f"Dim {selected_dim} exceeds inferred max {max_dim}. Using {max_dim}.")

words = words_for_dim(df, effective_dim)

with middle:
    with st.container(border=True):
        st.markdown(f"#### Words <span class='muted'>(dim {effective_dim})</span>", unsafe_allow_html=True)
        if not words:
            st.write("No words found.")
        else:
            st.markdown("<br>".join(words), unsafe_allow_html=True)

with right:
    with st.container(border=True):
        st.markdown("#### Model info")
        st.markdown(model_info_text(selected_model))
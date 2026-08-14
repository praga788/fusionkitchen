"""
FusionKitchen — Data Loader (Memory-Optimized)
==============================================
Loads pipeline artifacts with minimal memory footprint:
1. Memory-mapped numpy arrays for vectors (`mmap_mode='r'`).
2. Efficient column loading for recipe lookup table.
3. On-demand partitioned dataset querying for directions lookup
   (zero RAM consumption for the 888MB directions corpus).
"""

import os
import numpy as np
import pandas as pd

import config
from data_downloader import ensure_data_downloaded


def _load_embedding_table(path):
    """Returns (term -> row index dict, L2-normalized embedding matrix), or
    (None, None) if the file doesn't exist."""
    if not os.path.exists(path):
        return None, None
    try:
        vdf = pd.read_parquet(path)
        dim_cols = [c for c in vdf.columns if c.startswith("dim_")]
        terms = vdf["ingredient"].tolist()
        matrix = vdf[dim_cols].to_numpy(dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-8)
        matrix = matrix / norms
        return {t: i for i, t in enumerate(terms)}, matrix
    except Exception as e:
        print(f"[Warning] Failed loading embedding table {path}: {e}")
        return None, None


class DirectionsLookup:
    """Zero-memory on-demand reader for directions_lookup.parquet.
    Streams only the requested rows on demand rather than loading 2.2M text rows into RAM."""

    def __init__(self, path):
        self.path = path
        self._cache = {}
        self._dataset = None
        self._df = None

        if os.path.exists(path):
            try:
                import pyarrow.dataset as ds
                self._dataset = ds.dataset(path, format="parquet")
            except Exception as e:
                print(f"[Warning] Could not initialize pyarrow dataset for directions: {e}")

    def get(self, title: str, link: str):
        key = (title, link)
        if key in self._cache:
            return self._cache[key]

        # 1. Try PyArrow on-demand dataset filter (zero RAM)
        if self._dataset is not None:
            try:
                import pyarrow.dataset as ds
                expr = (ds.field("title") == title) & (ds.field("link") == link)
                scanner = self._dataset.scanner(
                    filter=expr,
                    columns=["NER_clean_str", "ingredients_raw_text", "directions_text"]
                )
                tbl = scanner.to_table()
                if tbl.num_rows > 0:
                    res = {
                        "NER_clean_str": tbl.column("NER_clean_str")[0].as_py() if "NER_clean_str" in tbl.column_names else "",
                        "ingredients_raw_text": tbl.column("ingredients_raw_text")[0].as_py() if "ingredients_raw_text" in tbl.column_names else "",
                        "directions_text": tbl.column("directions_text")[0].as_py() if "directions_text" in tbl.column_names else "",
                    }
                    self._cache[key] = res
                    return res
            except Exception:
                pass

        # 2. Fallback if dataset is a small in-memory df (e.g. in tests)
        if self._df is not None and key in self._df.index:
            row = self._df.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            res = {
                "NER_clean_str": str(row.get("NER_clean_str", "") or ""),
                "ingredients_raw_text": str(row.get("ingredients_raw_text", "") or ""),
                "directions_text": str(row.get("directions_text", "") or ""),
            }
            self._cache[key] = res
            return res

        return None


class PipelineData:
    """Loads once, at app startup. Pass the single instance around rather
    than re-reading files — these can be large."""

    def __init__(self):
        # If files are missing, attempt download from Hugging Face dataset if configured
        if not all(os.path.exists(p) for p in [
            config.RECIPE_VECTORS_W2V_PATH, config.RECIPE_LOOKUP_PATH, config.W2V_VECTORS_PATH,
        ]):
            ensure_data_downloaded()
            config.refresh_paths()

        self.is_real = all(os.path.exists(p) for p in [
            config.RECIPE_VECTORS_W2V_PATH, config.RECIPE_LOOKUP_PATH, config.W2V_VECTORS_PATH,
        ])

        if not self.is_real:
            print("Real pipeline artifacts not found — running in DEMO MODE with sample data.")
            print(f"  (Checked: {config.RECIPE_VECTORS_W2V_PATH})")
            print("  Edit the paths in config.py or set HF_DATASET_REPO to load full dataset.")
            self._init_empty()
            return

        print("Loading real pipeline artifacts (memory-optimized)...")

        # Load recipe lookup table with only required columns
        cols = ["title", "link", "source", "NER_text"]
        try:
            self.recipe_lookup_df = pd.read_parquet(config.RECIPE_LOOKUP_PATH, columns=cols).reset_index(drop=True)
        except Exception:
            self.recipe_lookup_df = pd.read_parquet(config.RECIPE_LOOKUP_PATH).reset_index(drop=True)

        # Use memory mapping for instant loading and minimal RAM usage
        self.recipe_vectors = {"word2vec": np.load(config.RECIPE_VECTORS_W2V_PATH, mmap_mode="r")}
        have_sbert_recipes = os.path.exists(config.RECIPE_VECTORS_SBERT_PATH)
        if have_sbert_recipes:
            self.recipe_vectors["sbert"] = np.load(config.RECIPE_VECTORS_SBERT_PATH, mmap_mode="r")

        self.w2v_term_idx, self.w2v_matrix = _load_embedding_table(config.W2V_VECTORS_PATH)
        self.sbert_term_idx, self.sbert_matrix = _load_embedding_table(config.SBERT_VECTORS_PATH)
        self.have_sbert = self.sbert_matrix is not None and have_sbert_recipes

        self.effective_alpha = config.ALPHA if self.have_sbert else 1.0
        if not self.have_sbert:
            print("  SBERT artifacts not found — running Word2Vec-only (alpha forced to 1.0).")

        if os.path.exists(config.MAPPING_PATH):
            mapping_df = pd.read_parquet(config.MAPPING_PATH)
            self.raw_to_canonical = dict(zip(mapping_df["raw_ingredient"], mapping_df["canonical_ingredient"]))
        else:
            self.raw_to_canonical = {}

        # Directions lookup — streaming on-demand to save ~4GB RAM
        if os.path.exists(config.DIRECTIONS_LOOKUP_PATH):
            self.directions_lookup = DirectionsLookup(config.DIRECTIONS_LOOKUP_PATH)
            self.directions_df = None  # Replaced by directions_lookup for zero memory usage
            self.have_directions = True
        else:
            self.directions_lookup = None
            self.directions_df = None
            self.have_directions = False

        print(f"Loaded {len(self.recipe_lookup_df):,} recipes, "
              f"Word2Vec vocab {len(self.w2v_term_idx):,}, "
              f"SBERT {'available' if self.have_sbert else 'unavailable'}")

    def _init_empty(self):
        self.recipe_lookup_df = None
        self.recipe_vectors = {}
        self.w2v_term_idx = self.w2v_matrix = None
        self.sbert_term_idx = self.sbert_matrix = None
        self.have_sbert = False
        self.effective_alpha = config.ALPHA
        self.raw_to_canonical = {}
        self.directions_lookup = None
        self.directions_df = None
        self.have_directions = False

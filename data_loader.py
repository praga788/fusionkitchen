"""
FusionKitchen — Data Loader
=============================
Loads all Step 1-4 pipeline artifacts once at startup. Everything the rest
of the app needs is exposed as attributes on a single PipelineData object.

If the real files aren't found, `PipelineData.is_real` is False and the UI
layer falls back to demo_data.py automatically — the app never crashes
just because a path is wrong, it just tells you so and keeps running.
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
    vdf = pd.read_parquet(path)
    dim_cols = [c for c in vdf.columns if c.startswith("dim_")]
    terms = vdf["ingredient"].tolist()
    matrix = vdf[dim_cols].to_numpy(dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-8)
    matrix = matrix / norms
    return {t: i for i, t in enumerate(terms)}, matrix


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

        print("Loading real pipeline artifacts...")

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

        if os.path.exists(config.DIRECTIONS_LOOKUP_PATH):
            self.directions_df = pd.read_parquet(config.DIRECTIONS_LOOKUP_PATH).set_index(["title", "link"])
            self.have_directions = True
        else:
            self.directions_df = None
            self.have_directions = False
            print("  directions_lookup.parquet not found — GenAI prompts will use ingredients only, no full directions.")

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
        self.directions_df = None
        self.have_directions = False

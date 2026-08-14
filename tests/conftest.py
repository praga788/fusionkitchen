"""
Shared pytest fixtures.

The fixtures here build a tiny, entirely SYNTHETIC copy of the pipeline
artifacts (a few rows, a handful of ingredients) in a temp directory, then
point config.py at it. This means the test suite is fully self-contained —
it does not need your real 2-million-row dataset to run, so it works in CI
and on any machine, not just yours.
"""

import numpy as np
import pandas as pd
import pytest

import config


TERMS = ["chicken", "garlic", "salt", "turkey", "carrot", "butter", "flour"]


def _make_embedding_table(path, terms, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(len(terms), dim)).astype(np.float32)
    df = pd.DataFrame(vecs, columns=[f"dim_{i}" for i in range(dim)])
    df.insert(0, "ingredient", terms)
    df.to_parquet(path, index=False)


@pytest.fixture
def synthetic_pipeline_dirs(tmp_path, monkeypatch):
    """Builds a tiny, valid set of pipeline artifacts on disk and points
    config.py's path variables at them. Returns the tmp_path root."""
    recommender_dir = tmp_path / "recommender_output"
    embeddings_dir = tmp_path / "embeddings_output"
    genai_dir = tmp_path / "genai_output"
    for d in (recommender_dir, embeddings_dir, genai_dir):
        d.mkdir()

    # --- recipe_lookup_table.parquet: 3 recipes ---
    recipes = pd.DataFrame({
        "title": ["Roast Chicken", "Turkey Soup", "Butter Cookies"],
        "link": ["example.com/1", "example.com/2", "example.com/3"],
        "source": ["Gathered", "Recipes1M", "Gathered"],
        "NER_text": ["chicken garlic salt", "turkey carrot salt", "butter flour salt"],
    })
    recipes.to_parquet(recommender_dir / "recipe_lookup_table.parquet", index=False)

    # --- recipe_vectors_word2vec.npy: one 4-dim vector per recipe above, in the SAME order ---
    rng = np.random.default_rng(1)
    recipe_vecs = rng.normal(size=(3, 4)).astype(np.float32)
    recipe_vecs = recipe_vecs / np.linalg.norm(recipe_vecs, axis=1, keepdims=True)
    np.save(recommender_dir / "recipe_vectors_word2vec.npy", recipe_vecs)

    # --- word2vec_vectors_canonical.parquet: ingredient embedding table ---
    _make_embedding_table(embeddings_dir / "word2vec_vectors_canonical.parquet", TERMS, dim=4, seed=2)

    # --- ingredient_canonical_mapping.parquet: identity mapping (no merges needed for this tiny set) ---
    mapping = pd.DataFrame({"raw_ingredient": TERMS, "canonical_ingredient": TERMS})
    mapping.to_parquet(embeddings_dir / "ingredient_canonical_mapping.parquet", index=False)

    # --- directions_lookup.parquet ---
    directions = pd.DataFrame({
        "title": ["Roast Chicken", "Turkey Soup", "Butter Cookies"],
        "link": ["example.com/1", "example.com/2", "example.com/3"],
        "source": ["Gathered", "Recipes1M", "Gathered"],
        "NER_clean_str": ["chicken; garlic; salt", "turkey; carrot; salt", "butter; flour; salt"],
        "ingredients_raw_text": ["1 chicken; 2 cloves garlic; salt", "1 lb turkey; 2 carrots; salt", "1 c butter; 2 c flour; salt"],
        "directions_text": ["Roast the chicken with garlic and salt until done.",
                             "Simmer turkey with carrots and salt for one hour.",
                             "Cream butter and flour with salt, then bake."],
    })
    directions.to_parquet(genai_dir / "directions_lookup.parquet", index=False)

    monkeypatch.setattr(config, "RECOMMENDER_DIR", str(recommender_dir))
    monkeypatch.setattr(config, "EMBEDDINGS_DIR", str(embeddings_dir))
    monkeypatch.setattr(config, "GENAI_DIR", str(genai_dir))
    monkeypatch.setattr(config, "RECIPE_VECTORS_W2V_PATH", str(recommender_dir / "recipe_vectors_word2vec.npy"))
    monkeypatch.setattr(config, "RECIPE_VECTORS_SBERT_PATH", str(recommender_dir / "recipe_vectors_sbert.npy"))  # intentionally absent
    monkeypatch.setattr(config, "RECIPE_LOOKUP_PATH", str(recommender_dir / "recipe_lookup_table.parquet"))
    monkeypatch.setattr(config, "W2V_VECTORS_PATH", str(embeddings_dir / "word2vec_vectors_canonical.parquet"))
    monkeypatch.setattr(config, "SBERT_VECTORS_PATH", str(embeddings_dir / "sbert_vectors_canonical.parquet"))  # intentionally absent
    monkeypatch.setattr(config, "MAPPING_PATH", str(embeddings_dir / "ingredient_canonical_mapping.parquet"))
    monkeypatch.setattr(config, "DIRECTIONS_LOOKUP_PATH", str(genai_dir / "directions_lookup.parquet"))

    return tmp_path


@pytest.fixture
def pipeline_data(synthetic_pipeline_dirs):
    """A real PipelineData instance loaded from the synthetic fixture above."""
    from data_loader import PipelineData
    return PipelineData()


@pytest.fixture
def recommender(pipeline_data):
    from recommender import RecipeRecommender
    return RecipeRecommender(pipeline_data)

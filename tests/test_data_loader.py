"""Tests for data_loader.py — real-data loading and the demo-mode fallback."""

import config
from data_loader import PipelineData


class TestRealDataLoading:
    def test_loads_successfully_from_synthetic_fixture(self, pipeline_data):
        assert pipeline_data.is_real is True
        assert len(pipeline_data.recipe_lookup_df) == 3
        assert pipeline_data.recipe_vectors["word2vec"].shape == (3, 4)

    def test_builds_canonical_mapping(self, pipeline_data):
        assert pipeline_data.raw_to_canonical.get("chicken") == "chicken"

    def test_loads_directions_when_present(self, pipeline_data):
        assert pipeline_data.have_directions is True


class TestDemoModeFallback:
    def test_falls_back_gracefully_when_no_files_exist(self, tmp_path, monkeypatch):
        """This is the scenario a fresh clone of the repo hits before the
        user has added their data/ folder — the app must not crash, it
        should cleanly report demo mode."""
        empty_dir = tmp_path / "nonexistent"
        monkeypatch.setattr(config, "RECOMMENDER_DIR", str(empty_dir))
        monkeypatch.setattr(config, "RECIPE_VECTORS_W2V_PATH", str(empty_dir / "recipe_vectors_word2vec.npy"))
        monkeypatch.setattr(config, "RECIPE_LOOKUP_PATH", str(empty_dir / "recipe_lookup_table.parquet"))
        monkeypatch.setattr(config, "W2V_VECTORS_PATH", str(empty_dir / "word2vec_vectors_canonical.parquet"))

        data = PipelineData()

        assert data.is_real is False
        assert data.recipe_lookup_df is None
        assert data.recipe_vectors == {}
        assert data.have_sbert is False

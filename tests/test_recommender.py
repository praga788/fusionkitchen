"""Tests for recommender.py — text normalization and the recommendation engine."""

from recommender import clean_formatting, canonicalize_text


class TestTextNormalization:
    def test_strips_punctuation_artifacts(self):
        assert clean_formatting("butter +") == "butter"
        # hyphens are normalized to single spaces (not preserved) —
        # canonicalize_text then strips "extra"/"virgin" as filler words
        assert clean_formatting("extra - virgin olive oil") == "extra virgin olive oil"

    def test_removes_filler_words(self):
        assert canonicalize_text("additional butter") == "butter"
        assert canonicalize_text("extra virgin olive oil") == "olive oil"

    def test_preserves_meaningful_words_that_look_like_fillers(self):
        # "cold" and "warm" are deliberately NOT filler words (Step 2c found
        # that stripping them incorrectly merges "cold water" / "warm water",
        # which are not interchangeable). This test guards against that
        # regression being reintroduced.
        assert canonicalize_text("cold water") == "cold water"
        assert canonicalize_text("warm water") == "warm water"

    def test_lowercases_and_collapses_whitespace(self):
        assert canonicalize_text("  Chicken   Breast  ") == "chicken breast"


class TestPipelineDataLoading:
    def test_loads_real_data_when_present(self, pipeline_data):
        assert pipeline_data.is_real is True
        assert len(pipeline_data.recipe_lookup_df) == 3

    def test_falls_back_when_sbert_missing(self, pipeline_data):
        # The fixture deliberately doesn't create SBERT files
        assert pipeline_data.have_sbert is False
        assert pipeline_data.effective_alpha == 1.0  # forced to Word2Vec-only


class TestRecommend:
    def test_returns_requested_number_of_results(self, recommender):
        results, matched = recommender.recommend(["chicken", "garlic"], top_n=2)
        assert len(results) == 2

    def test_recognizes_known_ingredients(self, recommender):
        results, matched = recommender.recommend(["chicken", "garlic", "salt"], top_n=3)
        assert "chicken" in matched
        assert "garlic" in matched
        assert "salt" in matched

    def test_canonicalizes_query_before_matching(self, recommender):
        # "additional garlic" should be treated the same as "garlic"
        results, matched = recommender.recommend(["additional garlic"], top_n=1)
        assert "garlic" in matched

    def test_handles_completely_unrecognized_ingredients(self, recommender):
        results, matched = recommender.recommend(["xyznonexistentfood"], top_n=5)
        assert results == []

    def test_results_include_similarity_score(self, recommender):
        results, matched = recommender.recommend(["chicken"], top_n=1)
        assert "similarity_score" in results[0]
        assert isinstance(results[0]["similarity_score"], float)


class TestGetSubstituteCandidates:
    def test_returns_candidates_excluding_the_ingredient_itself(self, recommender):
        candidates = recommender.get_substitute_candidates("chicken", k=3)
        assert "chicken" not in candidates
        assert len(candidates) <= 3

    def test_returns_empty_list_for_unknown_ingredient(self, recommender):
        assert recommender.get_substitute_candidates("xyznonexistentfood") == []


class TestGetRecipeIngredients:
    def test_prefers_semicolon_delimited_field_when_directions_available(self, recommender):
        # The synthetic fixture's directions_lookup has "chicken; garlic; salt"
        # for this recipe — this must come back as 3 separate entries, not
        # split further on internal spaces.
        recipe = {"title": "Roast Chicken", "link": "example.com/1", "NER_text": "chicken garlic salt"}
        ingredients = recommender.get_recipe_ingredients(recipe)
        assert ingredients == ["chicken", "garlic", "salt"]

    def test_falls_back_to_space_split_when_directions_unavailable(self, recommender):
        # A recipe with no matching entry in directions_lookup
        recipe = {"title": "Unknown Recipe", "link": "example.com/999", "NER_text": "carrot onion"}
        ingredients = recommender.get_recipe_ingredients(recipe)
        assert ingredients == ["carrot", "onion"]

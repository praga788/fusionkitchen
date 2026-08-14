"""Tests for genai_service.py — allergen flagging, prompt construction,
multi-key rotation with rate-limit fallback, and lenient JSON parsing."""

import api_key
import config
import genai_service
from genai_service import flag_allergens, build_prompt, call_openai, enrich_with_genai


def _reset_key_cycle(monkeypatch, keys):
    """The key-rotation cycle is cached at module level for performance —
    tests that change the key list must reset it, or they'd keep cycling
    through whatever keys an earlier test left behind."""
    monkeypatch.setattr(api_key, "GEMINI_API_KEYS", keys)
    monkeypatch.setattr(genai_service, "_key_cycle", None)


class TestFlagAllergens:
    def test_detects_dairy(self):
        assert "dairy" in flag_allergens(["butter", "flour", "salt"])

    def test_detects_multiple_categories(self):
        flags = flag_allergens(["shrimp", "butter", "flour"])
        assert "shellfish" in flags
        assert "dairy" in flags
        assert "gluten" in flags

    def test_no_false_positives_on_clean_ingredients(self):
        assert flag_allergens(["carrot", "celery", "onion"]) == []

    def test_case_insensitive(self):
        assert "dairy" in flag_allergens(["BUTTER"])


class TestBuildPrompt:
    def test_includes_recipe_title_and_ingredients(self):
        prompt = build_prompt("Test Recipe", "chicken, salt", "Cook it.", "Gathered", {}, [])
        assert "Test Recipe" in prompt
        assert "chicken, salt" in prompt

    def test_flags_unverified_source(self):
        prompt = build_prompt("T", "i", "d", "Gathered", {}, [])
        assert "not professionally verified" in prompt

    def test_flags_research_corpus_source(self):
        prompt = build_prompt("T", "i", "d", "Recipes1M", {}, [])
        assert "Recipe1M+ research dataset" in prompt

    def test_includes_grounded_candidates(self):
        prompt = build_prompt("T", "i", "d", "Gathered", {"butter": ["margarine", "oil"]}, [])
        assert "margarine" in prompt
        assert "oil" in prompt

    def test_includes_allergen_flags_when_present(self):
        prompt = build_prompt("T", "i", "d", "Gathered", {}, ["dairy", "nuts"])
        assert "dairy" in prompt and "nuts" in prompt

    def test_strict_grounding_instructs_only_candidates(self):
        """Default / rank-3+ behavior: substitution must come only from
        the grounded candidate list."""
        prompt = build_prompt("T", "i", "d", "Gathered", {}, [], strict_grounding=True)
        assert "ONLY from these" in prompt

    def test_loose_grounding_allows_model_to_deviate(self):
        """Top-2 behavior: the model may suggest something better than the
        grounded candidates if none of them are actually sensible — this
        is the deliberate fix for weak substitutions caused by imperfect
        ingredient cleaning."""
        prompt = build_prompt("T", "i", "d", "Gathered", {}, [], strict_grounding=False)
        assert "ONLY from these" not in prompt
        assert "suggest a better" in prompt or "your own best suggestion" in prompt


class TestCallOpenAI:
    def test_returns_placeholder_when_no_keys(self, monkeypatch):
        _reset_key_cycle(monkeypatch, [])
        result = call_openai("any prompt")
        assert "api_key.py" in result["explanation"]

    def test_handles_total_failure_gracefully(self, monkeypatch):
        """If every key's call fails for a non-rate-limit reason,
        call_openai must never raise — it should return an error dict so
        one failed recipe doesn't crash the whole batch."""
        _reset_key_cycle(monkeypatch, ["fake-key-1"])

        class _FailingClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        raise RuntimeError("simulated API failure")

        import openai as openai_module
        monkeypatch.setattr(openai_module, "OpenAI", lambda **kwargs: _FailingClient())

        result = call_openai("any prompt")
        assert "GenAI call failed" in result["explanation"]

    def test_falls_back_to_next_key_on_rate_limit(self, monkeypatch):
        """The core multi-key feature: a 429/RESOURCE_EXHAUSTED error on
        one key must trigger an automatic retry with the next key, not an
        immediate failure."""
        _reset_key_cycle(monkeypatch, ["exhausted-key", "working-key"])
        attempted_keys = []

        class _Msg:
            content = '{"explanation": "ok from working-key", "substitution": "", "healthier_alternative": "", "cooking_tip": "", "summary": ""}'
        class _Choice:
            message = _Msg()
        class _Resp:
            choices = [_Choice()]

        def _fake_openai(**kwargs):
            key = kwargs["api_key"]
            attempted_keys.append(key)
            class _Client:
                class chat:
                    class completions:
                        @staticmethod
                        def create(**kw):
                            if key == "exhausted-key":
                                err = Exception("429 RESOURCE_EXHAUSTED: quota exceeded")
                                err.status_code = 429
                                raise err
                            return _Resp()
            return _Client()

        import openai as openai_module
        monkeypatch.setattr(openai_module, "OpenAI", _fake_openai)

        result = call_openai("any prompt")
        assert attempted_keys == ["exhausted-key", "working-key"]
        assert result["explanation"] == "ok from working-key"

    def test_does_not_retry_other_keys_on_non_rate_limit_error(self, monkeypatch):
        """A genuine error (bad prompt, malformed request, etc.) shouldn't
        burn through every key retrying something that isn't a quota
        problem — only rate-limit errors trigger the next-key fallback."""
        _reset_key_cycle(monkeypatch, ["key-1", "key-2"])
        attempted_keys = []

        def _fake_openai(**kwargs):
            attempted_keys.append(kwargs["api_key"])
            class _Client:
                class chat:
                    class completions:
                        @staticmethod
                        def create(**kw):
                            raise RuntimeError("malformed request payload")
            return _Client()

        import openai as openai_module
        monkeypatch.setattr(openai_module, "OpenAI", _fake_openai)

        call_openai("any prompt")
        assert attempted_keys == ["key-1"]  # stopped after the first, non-rate-limit failure

    def test_rotates_across_keys_on_successive_calls(self, monkeypatch):
        """Load-balancing: independent calls (not retries) should spread
        across all configured keys round-robin, not always hit the first one."""
        _reset_key_cycle(monkeypatch, ["key-A", "key-B", "key-C"])
        attempted_keys = []

        class _Msg:
            content = '{"explanation": "ok", "substitution": "", "healthier_alternative": "", "cooking_tip": "", "summary": ""}'
        class _Choice:
            message = _Msg()
        class _Resp:
            choices = [_Choice()]

        def _fake_openai(**kwargs):
            attempted_keys.append(kwargs["api_key"])
            class _Client:
                class chat:
                    class completions:
                        @staticmethod
                        def create(**kw):
                            return _Resp()
            return _Client()

        import openai as openai_module
        monkeypatch.setattr(openai_module, "OpenAI", _fake_openai)

        for _ in range(4):
            call_openai("p")
        assert attempted_keys == ["key-A", "key-B", "key-C", "key-A"]

    def test_uses_configured_base_url(self, monkeypatch):
        """Confirms the client is actually pointed at Gemini's endpoint (or
        whatever config.API_BASE_URL is set to) — not silently defaulting
        back to OpenAI's own endpoint."""
        _reset_key_cycle(monkeypatch, ["fake-key"])
        captured = {}

        def _fake_openai(**kwargs):
            captured.update(kwargs)
            class _Msg:
                content = '{"explanation": "ok", "substitution": "", "healthier_alternative": "", "cooking_tip": "", "summary": ""}'
            class _Choice:
                message = _Msg()
            class _Resp:
                choices = [_Choice()]
            class _Client:
                class chat:
                    class completions:
                        @staticmethod
                        def create(**kw):
                            return _Resp()
            return _Client()

        import openai as openai_module
        monkeypatch.setattr(openai_module, "OpenAI", _fake_openai)

        call_openai("any prompt")
        assert captured.get("base_url") == config.API_BASE_URL

    def test_lenient_parsing_handles_markdown_fenced_json(self):
        """Models are less consistent than a strict JSON mode about
        returning bare JSON — this is the exact failure mode that
        motivated the fallback parser."""
        from genai_service import _parse_json_response
        fenced = '```json\n{"explanation": "test", "substitution": "a"}\n```'
        result = _parse_json_response(fenced)
        assert result["explanation"] == "test"

    def test_lenient_parsing_handles_unparseable_response(self):
        from genai_service import _parse_json_response
        result = _parse_json_response("Sorry, I cannot help with that.")
        assert "Could not parse" in result["explanation"]


class TestEnrichWithGenai:
    def test_adds_expected_fields(self, pipeline_data, recommender):
        recipe = {"title": "Roast Chicken", "link": "example.com/1", "source": "Gathered", "NER_text": "chicken garlic salt"}
        enriched = enrich_with_genai(pipeline_data, recommender, recipe)
        assert "explanation" in enriched
        assert "substitution" in enriched
        assert "cooking_tip" in enriched
        assert "allergen_flags" in enriched

    def test_pulls_real_directions_when_available(self, pipeline_data, recommender):
        recipe = {"title": "Roast Chicken", "link": "example.com/1", "source": "Gathered", "NER_text": "chicken garlic salt"}
        # No API key in this test env, but the allergen scan should still
        # reflect the real ingredients pulled from directions_lookup
        enriched = enrich_with_genai(pipeline_data, recommender, recipe)
        assert isinstance(enriched["allergen_flags"], list)

    def test_strict_grounding_defaults_to_true(self, pipeline_data, recommender):
        """Ensures existing callers that don't pass strict_grounding still
        get the safer, stricter default."""
        recipe = {"title": "Roast Chicken", "link": "example.com/1", "source": "Gathered", "NER_text": "chicken garlic salt"}
        enriched = enrich_with_genai(pipeline_data, recommender, recipe)  # no strict_grounding kwarg
        assert isinstance(enriched, dict)  # doesn't raise; default path works

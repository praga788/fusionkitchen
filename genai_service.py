"""
FusionKitchen — GenAI Service (Gemini, via its OpenAI-compatible endpoint)
=============================================================================
Generates a grounded explanation, substitution, and cooking tip for each
recommended recipe.

Three things this module does deliberately, not just for show:

1. GROUNDED SUBSTITUTIONS — for lower-confidence recipes, the model is
   given real nearest-neighbor candidates from Step 2's Word2Vec embeddings
   and told to choose only from them, rather than inventing a substitute
   freely. For the TOP 2 recipes specifically, grounding is loosened (see
   `strict_grounding` below) — the underlying ingredient vocabulary isn't
   perfectly cleaned, so a strictly-grounded suggestion can occasionally be
   worse than just asking the model directly. This is a deliberate
   precision/grounding trade-off for your best results, not an oversight.

2. A RULE-BASED ALLERGEN FLAG, independent of the LLM entirely (see
   flag_allergens below) — simple keyword matching that can't hallucinate
   the way a language model can. This is the safety backstop.

3. MULTI-KEY ROTATION WITH RATE-LIMIT FALLBACK — Gemini's free tier caps
   requests per key per day. Given several keys (api_key.GEMINI_API_KEYS),
   this rotates across them on every call, and if a call hits a 429/
   RESOURCE_EXHAUSTED error specifically, it immediately retries with the
   NEXT key before giving up — so one exhausted key doesn't fail your
   whole search.
"""

import re
import json
import itertools

import config
import api_key
from recommender import RecipeRecommender


def flag_allergens(ingredients: list) -> list:
    """Rule-based, independent of the LLM."""
    text = " ".join(ingredients).lower()
    return [category for category, keywords in config.ALLERGEN_KEYWORDS.items()
            if any(kw in text for kw in keywords)]


def build_prompt(title, ingredients_text, directions_text, source, substitute_candidates,
                  allergen_flags, strict_grounding=True):
    candidates_text = "\n".join(
        f"  - {ing}: {', '.join(cands) if cands else '(no grounded candidates available)'}"
        for ing, cands in substitute_candidates.items()
    )
    trust_note = (
        f"This recipe's source is '{source}' — "
        + ("scraped directly from the open web, not professionally verified."
           if source == "Gathered" else "merged from the Recipe1M+ research dataset.")
    )
    allergen_note = (
        f"Rule-based scan flagged: {', '.join(allergen_flags)}." if allergen_flags
        else "Rule-based scan found no common allergens."
    )

    if strict_grounding:
        substitution_instruction = (
            "Grounded substitute candidates (choose your substitution suggestion "
            f"ONLY from these):\n{candidates_text}"
        )
        substitution_key_desc = "one swap, from the candidates above only"
    else:
        # Used for the top-ranked recipes: the candidate list below is a
        # HINT, not a hard constraint — the ingredient vocabulary it's
        # drawn from isn't perfectly cleaned, so occasionally none of the
        # candidates are actually sensible. Prefer them when reasonable,
        # but use your own knowledge if a clearly better substitute exists.
        substitution_instruction = (
            "Possible substitute candidates for reference (prefer one of these if "
            "it's genuinely a good fit; if none of them make culinary sense, suggest "
            f"a better real-world substitute yourself instead):\n{candidates_text}"
        )
        substitution_key_desc = "one realistic swap for a key ingredient — from the candidates if a good one exists, otherwise your own best suggestion"

    return f"""You are helping a home cook understand a recipe. Be concise and factual — do not invent nutrition claims or safety information that isn't given to you.

Recipe: {title}
Ingredients: {ingredients_text}
Directions: {directions_text or "(not available)"}

{trust_note}
{allergen_note}

{substitution_instruction}

Respond with ONLY a valid JSON object — no markdown code fences, no explanation before or after it, just the raw JSON — with exactly these keys:
"explanation" (1-2 sentences on why this matches),
"substitution" ({substitution_key_desc}),
"healthier_alternative" (one realistic way to make this recipe healthier — e.g. reduce sugar/fat, add vegetables, a lighter cooking method — grounded in the actual ingredients/directions given, not invented),
"cooking_tip" (one practical tip specific to this recipe),
"summary" (a 2-3 sentence plain-language summary of the cooking directions)."""


# --- Multi-key rotation ---------------------------------------------------
_key_cycle = None

def _next_key_cycle():
    global _key_cycle
    if _key_cycle is None:
        _key_cycle = itertools.cycle(api_key.GEMINI_API_KEYS) if api_key.GEMINI_API_KEYS else None
    return _key_cycle


def _is_rate_limit_error(e: Exception) -> bool:
    status = getattr(e, "status_code", None)
    if status == 429:
        return True
    text = str(e).lower()
    return "429" in text or "resource_exhausted" in text or "rate limit" in text or "quota" in text


def call_openai(prompt: str) -> dict:
    """Returns the parsed JSON response, or a dict with an 'error' key if
    no key is configured or every key's call fails — never raises, so one
    failed recipe doesn't take down the whole batch.

    Rotates across api_key.GEMINI_API_KEYS: tries one key per attempt, and
    on a rate-limit error specifically, immediately retries with the next
    key (up to len(GEMINI_API_KEYS) attempts) before giving up.
    """
    keys = api_key.GEMINI_API_KEYS
    if not keys:
        return {
            "explanation": "Add your API key(s) in api_key.py to generate a live explanation.",
            "substitution": "",
            "healthier_alternative": "Add your API key(s) in api_key.py to generate a live suggestion.",
            "cooking_tip": "Add your API key(s) in api_key.py to generate a live tip.",
            "summary": "Add your API key(s) in api_key.py to generate a live summary.",
        }

    cycle = _next_key_cycle()
    last_error = None
    from openai import OpenAI

    for _ in range(len(keys)):
        current_key = next(cycle)
        try:
            client = OpenAI(api_key=current_key, base_url=config.API_BASE_URL)
            response = client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content
            return _parse_json_response(raw)
        except Exception as e:
            last_error = e
            if _is_rate_limit_error(e):
                continue  # this key is exhausted — try the next one
            break  # a different kind of error — retrying with another key won't help

    return {"explanation": f"(GenAI call failed: {last_error})", "substitution": "",
            "healthier_alternative": "", "cooking_tip": "", "summary": ""}


def _parse_json_response(raw: str) -> dict:
    """Tries a clean json.loads first; falls back to extracting the first
    {...} block if the model wrapped the JSON in markdown fences or added
    stray text before/after it."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"explanation": f"(Could not parse model response as JSON: {raw[:150]})",
                "substitution": "", "healthier_alternative": "", "cooking_tip": "", "summary": ""}


def enrich_with_genai(data, recommender: RecipeRecommender, recipe: dict, strict_grounding: bool = True) -> dict:
    """Takes one recipe dict from RecipeRecommender.recommend() and adds
    explanation/substitution/healthier_alternative/cooking_tip/summary/
    allergen_flags to it, grounded in real ingredient data.

    strict_grounding=False (intended for your top 1-2 results) lets the
    model deviate from the grounded candidate list if none of them are a
    sensible substitute — see build_prompt's docstring above for why.
    """
    canon_ings = recommender.get_recipe_ingredients(recipe)[:3]
    substitute_candidates = {ing: recommender.get_substitute_candidates(ing, k=config.N_SUBSTITUTE_CANDIDATES)
                              for ing in canon_ings}

    ner_text = recipe.get("NER_text", "")
    ingredients_text, directions_text = ner_text, ""
    allergen_source_list = recommender.get_recipe_ingredients(recipe)

    if data.have_directions:
        key = (recipe["title"], recipe["link"])
        if key in data.directions_df.index:
            row = data.directions_df.loc[key]
            if isinstance(row, type(data.directions_df)):  # duplicate (title, link) -> DataFrame, not Series; take first
                row = row.iloc[0]
            ingredients_text = row.get("ingredients_raw_text", ingredients_text)
            directions_text = row.get("directions_text", "")

    allergen_flags = flag_allergens(allergen_source_list)

    prompt = build_prompt(recipe["title"], ingredients_text, directions_text,
                           recipe.get("source", ""), substitute_candidates, allergen_flags,
                           strict_grounding=strict_grounding)
    genai_fields = call_openai(prompt)

    recipe.update(genai_fields)
    recipe["allergen_flags"] = allergen_flags
    return recipe

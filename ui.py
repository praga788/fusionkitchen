"""
FusionKitchen — UI Layer
==========================
Gradio Blocks layout and the "case file" card rendering. No recommendation
or GenAI logic lives here — this module only turns data into HTML/UI.
"""

import html
import functools

import gradio as gr

import config
from demo_data import DEMO_RECIPES
from recommender import RecipeRecommender
from genai_service import enrich_with_genai
from theme import CUSTOM_CSS


def _caution_tag(allergen_flags: list) -> str:
    if not allergen_flags:
        return ""
    flags = " • ".join(f.upper() for f in allergen_flags)
    return f'<span class="caution-tag">CAUTION: {html.escape(flags)}</span>'


def _match_label(pct: int) -> str:
    """Similarity-score interpretation, not just a raw number — this is
    the explainability feature your proposal calls out directly."""
    if pct >= 90:
        return "STRONG MATCH"
    if pct >= 75:
        return "GOOD MATCH"
    if pct >= 60:
        return "LOOSE MATCH"
    return "WEAK MATCH"


def _overlap_line(have: list, needs: list) -> str:
    """The "why this recipe" transparency feature: shows which of the
    user's ingredients this recipe actually uses, and what else it needs
    beyond that — so a high match % is never a mystery."""
    have_text = ", ".join(have) if have else "(none matched directly — similarity is profile-based, see THE PITCH)"
    parts = [f'<div class="overlap-line"><span class="overlap-label">YOU HAVE:</span> {html.escape(have_text)}</div>']
    if needs:
        parts.append(f'<div class="overlap-line"><span class="overlap-label">ALSO NEEDS:</span> {html.escape(", ".join(needs))}</div>')
    return "".join(parts)


def render_results(recipes: list, ingredients_text: str, is_real: bool) -> str:
    if not recipes:
        return render_empty_state()

    cards = []
    for r in recipes:
        match_pct = int(round(r["similarity_score"] * 100))
        cards.append(f"""
        <div class="recipe-card">
          <div class="match-badge">{match_pct}%<span>MATCH</span></div>
          <h3 class="card-title">{html.escape(r['title'])}</h3>
          <span class="match-label">{_match_label(match_pct)}</span>
          {_caution_tag(r['allergen_flags'])}
          {_overlap_line(r.get('have', []), r.get('needs', []))}
          <div class="field"><span class="field-label">THE PITCH — why this one</span>{html.escape(r['explanation'])}</div>
          <div class="field"><span class="field-label">THE SWITCH — a substitution</span>{html.escape(r['substitution'])}</div>
          <div class="field"><span class="field-label">THE LIGHTER CUT — healthier alternative</span>{html.escape(r.get('healthier_alternative', ''))}</div>
          <div class="field"><span class="field-label">THE TRICK — a cooking tip</span>{html.escape(r['cooking_tip'])}</div>
          <div class="field"><span class="field-label">THE RECAP — recipe summary</span>{html.escape(r.get('summary', ''))}</div>
        </div>""")

    mode_note = "" if is_real else (
        '<p class="chapter-sub" style="color:#E8B923">Running on DEMO DATA — '
        'real pipeline artifacts not found. See config.py.</p>'
    )
    return f"""
    <div class="chapter-header">
      <span class="eyebrow">CHAPTER TWO</span>
      <h2>THE LINEUP</h2>
      <div class="rule"></div>
      <p class="chapter-sub">{len(recipes)} suspects matched against: <em>{html.escape(ingredients_text)}</em></p>
      {mode_note}
    </div>
    <div class="results-grid">{''.join(cards)}</div>
    """


def render_empty_state(warning: str = None) -> str:
    sub = warning or "No suspects yet. List what you've got in Chapter One and pull the trigger."
    return f"""
    <div class="chapter-header">
      <span class="eyebrow">CHAPTER TWO</span>
      <h2>THE LINEUP</h2>
      <div class="rule"></div>
      <p class="chapter-sub">{html.escape(sub)}</p>
    </div>
    """


def _rerank_prioritizing_exact_matches(raw_results: list, matched_terms: set, recommender: RecipeRecommender, top_n: int) -> list:
    """Re-ranks the similarity-scored candidate pool so recipes that need
    the FEWEST extra ingredients beyond what the user typed come first —
    "prioritize dishes that only use the stated ingredients, and only pull
    in something broader if not enough exact-ish matches exist" — before
    falling back to plain similarity ranking as the tie-breaker.
    """
    scored = []
    for r in raw_results:
        recipe_ingredients = recommender.get_recipe_ingredients(r)
        extra_needed = [i for i in recipe_ingredients if i not in matched_terms]
        scored.append((len(extra_needed), -r["similarity_score"], r))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [r for _, _, r in scored[:top_n]]


def make_recommend_fn(data, recommender: RecipeRecommender):
    """Closure so the Gradio callback has access to the loaded data/recommender
    without relying on module-level globals."""

    def _compute(ingredients_text: str):
        if not ingredients_text.strip():
            return render_empty_state()

        user_ingredients = [i.strip() for i in ingredients_text.split(",") if i.strip()]

        if data.is_real:
            # Pull a larger pool by similarity first, THEN re-rank down to
            # TOP_N so "uses only what you have" can actually win out over
            # a slightly-higher-similarity recipe that needs more extras.
            raw_results, matched_terms = recommender.recommend(user_ingredients, top_n=config.RERANK_POOL_SIZE)
            if not raw_results:
                return render_empty_state(
                    warning="None of those ingredients were recognized. Try more common ingredient names."
                )
            raw_results = _rerank_prioritizing_exact_matches(raw_results, matched_terms, recommender, config.TOP_N)

            recipes = []
            for rank, r in enumerate(raw_results):
                recipe_ingredients = recommender.get_recipe_ingredients(r)  # multi-word-safe, e.g. keeps "lemon juice" intact
                # Top 2 results get loosened grounding (see genai_service's
                # build_prompt docstring) — everything else stays strict.
                strict_grounding = rank >= 2
                enriched = enrich_with_genai(data, recommender, r, strict_grounding=strict_grounding)
                have = [i for i in recipe_ingredients if i in matched_terms]
                needs = [i for i in recipe_ingredients if i not in matched_terms]
                recipes.append({
                    "title": enriched["title"],
                    "source": enriched.get("source", "Gathered"),
                    "similarity_score": float(enriched["similarity_score"]),
                    "have": have,
                    "needs": needs,
                    "explanation": enriched.get("explanation", ""),
                    "substitution": enriched.get("substitution", ""),
                    "healthier_alternative": enriched.get("healthier_alternative", ""),
                    "cooking_tip": enriched.get("cooking_tip", ""),
                    "summary": enriched.get("summary", ""),
                    "allergen_flags": enriched.get("allergen_flags", []),
                })
        else:
            recipes = [dict(r, have=r["ingredients"], needs=[]) for r in DEMO_RECIPES[:config.TOP_N]]

        return render_results(recipes, ingredients_text, data.is_real)

    # Inference-speed optimization: identical queries (a very likely case —
    # e.g. a user hitting "Find" twice, or a demo repeating the same
    # example) are served instantly from cache instead of re-running
    # recipe scoring AND re-calling the Gemini API a second time. This is
    # a real, measurable speedup, not just a benchmark number.
    return functools.lru_cache(maxsize=128)(_compute)


def build_app(data, recommender: RecipeRecommender) -> gr.Blocks:
    get_recommendations = make_recommend_fn(data, recommender)

    with gr.Blocks(title="FusionKitchen — Ingredient Recommender") as demo:
        gr.HTML("""
        <div class="hero-wrap">
          <h1 class="hero-title">FUSIONKITCHEN</h1>
          <div class="hero-sub">An Ingredient-Matching Picture &nbsp;•&nbsp; Two Chapters &nbsp;•&nbsp; No Down-Sampling</div>
          <div class="hero-rule"></div>
        </div>
        """)

        with gr.Row():
            with gr.Column():
                gr.HTML("""
                <div class="chapter-header">
                  <span class="eyebrow">CHAPTER ONE</span>
                  <h2>THE INGREDIENTS</h2>
                  <div class="rule"></div>
                  <p class="chapter-sub">Tell us what's in the kitchen. We'll find the closest matches from the archive.</p>
                </div>
                <span class="intake-label">WHAT'VE YOU GOT? (comma-separated)</span>
                """)
                ingredient_input = gr.Textbox(
                    placeholder="chicken, garlic, lemon, olive oil...",
                    show_label=False,
                    elem_id="ingredient-input",
                    lines=2,
                )
                find_btn = gr.Button("FIND MY RECIPES", elem_id="find-btn")

        results_html = gr.HTML(render_empty_state())

        find_btn.click(fn=get_recommendations, inputs=ingredient_input, outputs=results_html)
        ingredient_input.submit(fn=get_recommendations, inputs=ingredient_input, outputs=results_html)

        gr.HTML("""
        <div class="footer-note">
          ALL RECIPES DRAWN FROM THE RECIPENLG ARCHIVE — 2.2 MILLION FILES, MOSTLY UNVERIFIED.<br>
          MATCH SCORES ARE ESTIMATES, NOT GUARANTEES. COOK AT YOUR OWN RISK.
        </div>
        """)

    return demo

"""
FusionKitchen — Streamlit Application
=====================================
Pulp-cinema / grindhouse styled ingredient recommendation web app.
Connects to Hugging Face Datasets for the 5GB dataset storage.

Run locally:
    streamlit run streamlit_app.py
"""

import os
import html
import functools
import streamlit as st

import config
import api_key
from data_loader import PipelineData
from recommender import RecipeRecommender
from genai_service import enrich_with_genai
from demo_data import DEMO_RECIPES

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FusionKitchen — Ingredient Recommender",
    page_icon="🔪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

STREAMLIT_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Anton&family=Oswald:wght@400;500;600;700&family=Courier+Prime:wght@400;700&display=swap');

:root {
  --void: #0D0B0A;
  --smoke: #1C1815;
  --blood: #A6192E;
  --marquee: #E8B923;
  --bone: #F2EDE4;
  --rust: #6E2A1E;
}

/* Background & Global Fonts */
.stApp {
  background: radial-gradient(ellipse at top, #1a1512 0%, var(--void) 65%) !important;
  font-family: 'Oswald', sans-serif !important;
  color: var(--bone) !important;
}

/* Hide Streamlit Header & Footer elements */
header[data-testid="stHeader"] { background: transparent !important; }
footer { visibility: hidden !important; }

/* Hero Section */
.hero-wrap { text-align: center; padding: 2rem 1rem 1rem; }
.hero-title {
  font-family: 'Anton', sans-serif;
  font-size: clamp(2.2rem, 7.5vw, 4.8rem);
  color: var(--blood);
  text-shadow: 3px 3px 0 var(--marquee), 6px 6px 0 rgba(0,0,0,0.6);
  letter-spacing: 0.03em;
  transform: rotate(-1.5deg);
  margin: 0;
  line-height: 1;
}
.hero-sub {
  font-family: 'Oswald', sans-serif;
  font-weight: 600;
  letter-spacing: 0.28em;
  color: var(--bone);
  opacity: 0.85;
  font-size: 0.9rem;
  margin-top: 0.9rem;
  text-transform: uppercase;
}
.hero-rule { width: 150px; height: 3px; background: var(--marquee); margin: 1rem auto 1.5rem; }

/* Chapter Headers */
.chapter-header { padding: 0.5rem 0.2rem 1rem; }
.eyebrow {
  font-family: 'Courier Prime', monospace;
  font-weight: 700;
  color: var(--marquee);
  letter-spacing: 0.3em;
  font-size: 0.82rem;
}
.chapter-header h2 {
  font-family: 'Anton', sans-serif;
  color: var(--bone);
  font-size: clamp(1.7rem, 4vw, 2.4rem);
  margin: 0.2rem 0 0;
  letter-spacing: 0.02em;
}
.rule { width: 100%; height: 2px; background: linear-gradient(90deg, var(--rust), transparent); margin: 0.6rem 0; }
.chapter-sub { font-size: 0.95rem; color: var(--bone); opacity: 0.85; margin: 0; }
.chapter-sub em { color: var(--marquee); font-style: normal; }

/* Text Area & Input Styling */
.stTextInput input, .stTextArea textarea {
  background: var(--smoke) !important;
  border: 2px solid var(--rust) !important;
  color: var(--bone) !important;
  font-family: 'Courier Prime', monospace !important;
  font-size: 1.05rem !important;
  border-radius: 2px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--marquee) !important;
  box-shadow: 0 0 0 2px rgba(232,185,35,0.25) !important;
}

/* Find Button */
div.stButton > button:first-child {
  font-family: 'Anton', sans-serif !important;
  font-size: 1.2rem !important;
  letter-spacing: 0.05em !important;
  background: var(--blood) !important;
  color: var(--bone) !important;
  border: 2px solid var(--rust) !important;
  border-radius: 2px !important;
  box-shadow: 4px 4px 0 rgba(0,0,0,0.5) !important;
  padding: 0.55rem 1.8rem !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease !important;
  width: 100%;
}
div.stButton > button:first-child:hover {
  transform: translate(-2px, -2px) !important;
  box-shadow: 6px 6px 0 rgba(0,0,0,0.5) !important;
  background: #8f1527 !important;
}

/* Results Grid & Cards */
.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 1.5rem;
  margin-top: 1rem;
}
.recipe-card {
  position: relative;
  background: var(--smoke);
  border: 1px solid var(--rust);
  border-left: 5px solid var(--blood);
  padding: 1.4rem 1.4rem 1.2rem;
  border-radius: 2px;
  box-shadow: 5px 5px 0 rgba(0,0,0,0.4);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.recipe-card:hover {
  transform: rotate(-0.5deg) translateY(-4px);
  box-shadow: 8px 12px 0 rgba(0,0,0,0.55);
}
.card-title {
  font-family: 'Anton', sans-serif;
  color: var(--marquee);
  font-size: 1.4rem;
  margin: 0 3rem 0.6rem 0;
  line-height: 1.15;
  text-transform: uppercase;
}
.match-badge {
  position: absolute;
  top: -14px; right: -10px;
  background: var(--blood);
  color: var(--bone);
  border: 2px solid var(--void);
  border-radius: 50%;
  width: 65px; height: 65px;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  font-family: 'Anton', sans-serif;
  font-size: 1.1rem;
  transform: rotate(8deg);
  box-shadow: 2px 3px 0 rgba(0,0,0,0.5);
}
.match-badge span { font-family: 'Courier Prime', monospace; font-size: 0.52rem; letter-spacing: 0.05em; }

.caution-tag {
  display: inline-block;
  background: var(--blood);
  color: var(--bone);
  font-family: 'Courier Prime', monospace;
  font-weight: 700;
  font-size: 0.68rem;
  letter-spacing: 0.05em;
  padding: 0.25rem 0.55rem;
  border-radius: 2px;
  margin-left: 0.4rem;
  margin-bottom: 0.4rem;
}

.match-label {
  display: block;
  font-family: 'Courier Prime', monospace;
  font-weight: 700;
  font-size: 0.72rem;
  letter-spacing: 0.1em;
  color: var(--marquee);
  opacity: 0.9;
  margin: -0.3rem 0 0.5rem;
}

.overlap-line {
  font-size: 0.84rem;
  line-height: 1.4;
  color: var(--bone);
  opacity: 0.88;
  margin-bottom: 0.35rem;
}
.overlap-label {
  font-family: 'Courier Prime', monospace;
  font-weight: 700;
  font-size: 0.7rem;
  letter-spacing: 0.05em;
  color: var(--marquee);
  margin-right: 0.3rem;
}

.field { font-size: 0.88rem; margin-bottom: 0.6rem; line-height: 1.45; color: var(--bone); opacity: 0.94; }
.field-label {
  display: block;
  font-family: 'Courier Prime', monospace;
  font-weight: 700;
  color: var(--marquee);
  font-size: 0.7rem;
  letter-spacing: 0.08em;
  margin-bottom: 0.15rem;
}

.footer-note {
  text-align: center;
  font-family: 'Courier Prime', monospace;
  font-size: 0.75rem;
  opacity: 0.6;
  margin: 3rem 0 1.5rem;
  letter-spacing: 0.04em;
}
</style>
"""

st.markdown(STREAMLIT_CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Pipeline Loading with Resource Cache
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing pipeline & embeddings...")
def load_pipeline():
    # Sync Streamlit secrets to environment for API key and Dataset repo if present
    if hasattr(st, "secrets"):
        if "HF_DATASET_REPO" in st.secrets:
            os.environ["HF_DATASET_REPO"] = st.secrets["HF_DATASET_REPO"]
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
        if "GEMINI_API_KEYS" in st.secrets:
            os.environ["GEMINI_API_KEYS"] = st.secrets["GEMINI_API_KEYS"]

    data = PipelineData()
    recommender = RecipeRecommender(data)
    return data, recommender


# -----------------------------------------------------------------------------
# 3. Helper Rendering Functions
# -----------------------------------------------------------------------------
def _caution_tag(allergen_flags: list) -> str:
    if not allergen_flags:
        return ""
    flags = " • ".join(f.upper() for f in allergen_flags)
    return f'<span class="caution-tag">CAUTION: {html.escape(flags)}</span>'


def _match_label(pct: int) -> str:
    if pct >= 90:
        return "STRONG MATCH"
    if pct >= 75:
        return "GOOD MATCH"
    if pct >= 60:
        return "LOOSE MATCH"
    return "WEAK MATCH"


def _overlap_line(have: list, needs: list) -> str:
    have_text = ", ".join(have) if have else "(profile-based match)"
    parts = [f'<div class="overlap-line"><span class="overlap-label">YOU HAVE:</span> {html.escape(have_text)}</div>']
    if needs:
        parts.append(f'<div class="overlap-line"><span class="overlap-label">ALSO NEEDS:</span> {html.escape(", ".join(needs))}</div>')
    return "".join(parts)


def _rerank_prioritizing_exact_matches(raw_results: list, matched_terms: set, recommender: RecipeRecommender, top_n: int) -> list:
    scored = []
    for r in raw_results:
        recipe_ingredients = recommender.get_recipe_ingredients(r)
        extra_needed = [i for i in recipe_ingredients if i not in matched_terms]
        scored.append((len(extra_needed), -r["similarity_score"], r))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [r for _, _, r in scored[:top_n]]


# -----------------------------------------------------------------------------
# 4. Main App Layout
# -----------------------------------------------------------------------------
def main():
    # Hero Title
    st.markdown("""
    <div class="hero-wrap">
      <h1 class="hero-title">FUSIONKITCHEN</h1>
      <div class="hero-sub">An Ingredient-Matching Picture &nbsp;•&nbsp; Two Chapters &nbsp;•&nbsp; No Down-Sampling</div>
      <div class="hero-rule"></div>
    </div>
    """, unsafe_allow_html=True)

    data, recommender = load_pipeline()

    # Chapter 1: Ingredients Input
    st.markdown("""
    <div class="chapter-header">
      <span class="eyebrow">CHAPTER ONE</span>
      <h2>THE INGREDIENTS</h2>
      <div class="rule"></div>
      <p class="chapter-sub">Tell us what's in the kitchen. We'll find the closest matches from the archive.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        ingredients_input = st.text_area(
            label="WHAT'VE YOU GOT? (comma-separated)",
            value="chicken, garlic, lemon, olive oil",
            height=100,
            help="Type your ingredients separated by commas",
        )
    with col2:
        st.write("")
        st.write("")
        find_button = st.button("FIND MY RECIPES", use_container_width=True)

    # Status Note
    if not data.is_real:
        st.info("⚡ Running on DEMO DATA — upload your 5GB dataset to Hugging Face Dataset to activate the 2.2M recipe archive.")

    # Chapter 2: The Lineup
    if find_button or ingredients_input:
        user_ingredients = [i.strip() for i in ingredients_input.split(",") if i.strip()]

        if not user_ingredients:
            st.warning("Please enter at least one ingredient.")
            return

        with st.spinner("Scoring recipes across 2.2M corpus and generating grounded tips..."):
            if data.is_real:
                raw_results, matched_terms = recommender.recommend(user_ingredients, top_n=config.RERANK_POOL_SIZE)
                if not raw_results:
                    st.warning("None of those ingredients were recognized. Try more common ingredient names.")
                    return

                raw_results = _rerank_prioritizing_exact_matches(raw_results, matched_terms, recommender, config.TOP_N)
                recipes = []
                for rank, r in enumerate(raw_results):
                    recipe_ingredients = recommender.get_recipe_ingredients(r)
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

            # Render Cards
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

            results_html = f"""
            <div class="chapter-header">
              <span class="eyebrow">CHAPTER TWO</span>
              <h2>THE LINEUP</h2>
              <div class="rule"></div>
              <p class="chapter-sub">{len(recipes)} suspects matched against: <em>{html.escape(ingredients_input)}</em></p>
            </div>
            <div class="results-grid">{''.join(cards)}</div>
            """
            st.markdown(results_html, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer-note">
      ALL RECIPES DRAWN FROM THE RECIPENLG ARCHIVE — 2.2 MILLION FILES.<br>
      MATCH SCORES ARE ESTIMATES, NOT GUARANTEES. COOK AT YOUR OWN RISK.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

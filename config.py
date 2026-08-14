"""
FusionKitchen — Configuration
==============================
All file paths and tunable settings live here.

DATA LOCATION — works both locally AND on Hugging Face Spaces:
This checks your existing local Windows folders FIRST (so nothing changes
for local development). If those aren't found — which will always be true
on Hugging Face's Linux servers, since "C:\\Users\\hp" doesn't exist there —
it falls back to a `data/` folder living right next to this file. For
deployment, copy your recommender_output/, embeddings_output/, and
genai_output/ folders into a `data/` folder in this project (see
data/README.md) and they'll be picked up automatically, no code changes
needed.

For your OpenAI API key, edit api_key.py instead — keeping it in its own
file (and reading from an environment variable first) makes it safe to
commit this repo to GitHub and to use Hugging Face Spaces' secret storage.
"""

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


DATA_DIR = os.path.join(_THIS_DIR, "data")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "")


def _resolve_dir(local_windows_path: str, bundled_subfolder: str) -> str:
    """Prefer the local Windows path if it exists; otherwise fall back
    to a `data/<bundled_subfolder>` folder next to this file."""
    if os.path.isdir(local_windows_path):
        return local_windows_path
    return os.path.join(DATA_DIR, bundled_subfolder)


def refresh_paths():
    """Refreshes path variables after dataset download."""
    global RECOMMENDER_DIR, EMBEDDINGS_DIR, GENAI_DIR
    global RECIPE_VECTORS_W2V_PATH, RECIPE_VECTORS_SBERT_PATH, RECIPE_LOOKUP_PATH
    global W2V_VECTORS_PATH, SBERT_VECTORS_PATH, MAPPING_PATH, DIRECTIONS_LOOKUP_PATH

    RECOMMENDER_DIR = _resolve_dir(r"C:\Users\hp\recommender_output", "recommender_output")
    EMBEDDINGS_DIR = _resolve_dir(r"C:\Users\hp\embeddings_output", "embeddings_output")
    GENAI_DIR = _resolve_dir(r"C:\Users\hp\genai_output", "genai_output")

    RECIPE_VECTORS_W2V_PATH = os.path.join(RECOMMENDER_DIR, "recipe_vectors_word2vec.npy")
    RECIPE_VECTORS_SBERT_PATH = os.path.join(RECOMMENDER_DIR, "recipe_vectors_sbert.npy")
    RECIPE_LOOKUP_PATH = os.path.join(RECOMMENDER_DIR, "recipe_lookup_table.parquet")

    W2V_VECTORS_PATH = os.path.join(EMBEDDINGS_DIR, "word2vec_vectors_canonical.parquet")
    SBERT_VECTORS_PATH = os.path.join(EMBEDDINGS_DIR, "sbert_vectors_canonical.parquet")
    MAPPING_PATH = os.path.join(EMBEDDINGS_DIR, "ingredient_canonical_mapping.parquet")

    DIRECTIONS_LOOKUP_PATH = os.path.join(GENAI_DIR, "directions_lookup.parquet")


# Initialize paths on module import
refresh_paths()

# =============================================================================
# Recommender settings
# =============================================================================
ALPHA = 0.7    # Word2Vec/SBERT blend weight — 0.7 favors Word2Vec, per Step 3's
               # recall@5/@10/MRR evaluation showing it as the stronger of the two.
               # Automatically forced to 1.0 if SBERT artifacts aren't found.
TOP_N = 3      # number of recipes actually shown to the user
RERANK_POOL_SIZE = 30  # how many candidates to pull by similarity BEFORE
                        # re-ranking to prefer recipes that need the fewest
                        # extra ingredients beyond what the user typed —
                        # needs to be bigger than TOP_N so there's something
                        # real to re-rank among

# =============================================================================
# GenAI settings (Google Gemini — via its OpenAI-compatible endpoint)
# =============================================================================
# Google provides an official OpenAI-compatible endpoint for Gemini (confirmed
# directly from ai.google.dev/gemini-api/docs/openai) — so the same `openai`
# Python package works here too, same pattern as OpenRouter: point it at a
# different URL, use a different key.
API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# "gemini-flash-latest" is Google's own auto-updating alias — it always
# points at the current stable Flash release rather than one dated version,
# with a 2-week email notice before anything breaking changes underneath it
# (confirmed from ai.google.dev/gemini-api/docs/models). This is the same
# reasoning as OpenRouter's "openrouter/free" router: don't pin a specific
# dated model name that can be deprecated with little warning.
# As of testing, Gemini's FREE tier covers Flash and Flash-Lite models (NOT
# Pro, which requires billing) — "gemini-flash-latest" stays on the free
# side. Double-check current free-tier coverage at
# https://ai.google.dev/gemini-api/docs/pricing if you hit a billing error.
OPENAI_MODEL = "gemini-flash-latest"

N_SUBSTITUTE_CANDIDATES = 5    # how many nearest-neighbor candidates to offer per ingredient

ALLERGEN_KEYWORDS = {
    "dairy": ["milk", "butter", "cheese", "cream", "yogurt", "yoghurt", "buttermilk"],
    "nuts": ["almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut", "peanut", "macadamia"],
    "gluten": ["flour", "wheat", "bread", "pasta", "breadcrumb", "barley", "rye"],
    "shellfish": ["shrimp", "crab", "lobster", "prawn", "clam", "mussel", "oyster", "scallop"],
    "soy": ["soy", "soybean", "tofu", "edamame", "miso"],
    "egg": ["egg", "eggs", "egg white", "egg yolk", "mayonnaise"],
}

# =============================================================================
# Server
# =============================================================================
# Hugging Face Spaces sets SPACE_ID automatically — used here to bind to
# 0.0.0.0 (reachable from outside the container) when deployed, vs.
# 127.0.0.1 (localhost only) for local development.
IS_HF_SPACE = "SPACE_ID" in os.environ
SERVER_NAME = "0.0.0.0" if IS_HF_SPACE else "127.0.0.1"
SERVER_PORT = 7860

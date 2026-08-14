---
title: FusionKitchen
emoji: 🔪
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

# FusionKitchen — Ingredient Recommender App

A Gradio app: type the ingredients you have, get back the closest-matching
real recipes from the RecipeNLG corpus, with AI-generated explanations,
substitutions, and cooking tips.

This is Step 5 (Application Development & Deployment) of the FusionKitchen
project, built on top of Steps 1-4 (data pipeline, embeddings, recommender,
GenAI integration).

> The block of YAML at the very top of this file is required by Hugging
> Face Spaces to recognize this as a Gradio app — GitHub just renders it as
> a harmless metadata block, it doesn't affect anything there.

---

## 1. Local setup

```
pip install -r requirements.txt
```

Open `api_key.py` and paste your Gemini API key(s) directly into the
`GEMINI_API_KEYS` list (up to 5 supported — see that file's comments for
why: Gemini's free tier caps requests per key per day, and this rotates
across keys to multiply your effective quota, automatically falling back
to the next key if one hits its limit).

**Note on GitHub**: this version keeps things simple for local running —
`api_key.py` itself is NOT gitignored, so if you paste real keys directly
into it and push to GitHub, they'll be visible in your repo. That's fine
for local-only use; if you plan to push this to a public GitHub repo,
either keep this repo private, or ask for the gitignored-key-file version
of this app instead before pushing.

**This app uses [Google Gemini](https://aistudio.google.com/apikey) by
default**, not OpenAI directly — accessed through Gemini's official
OpenAI-compatible endpoint (confirmed from ai.google.dev/gemini-api/docs/
openai), so the same `openai` Python package works, just pointed at
Google's URL. `config.py`'s `OPENAI_MODEL` is set to `gemini-flash-latest`
— Google's own auto-updating alias for the current stable Flash model,
which stays on the free tier (Pro models require billing; Flash/Flash-Lite
don't). To use OpenAI or OpenRouter directly instead, edit `API_BASE_URL`
and `OPENAI_MODEL` in `config.py`.

Run it:
```
python app.py
```
Open the printed `http://127.0.0.1:7860` link. Watch the first lines it
prints — they tell you plainly whether it found your real pipeline data
(`REAL DATA mode`) or fell back to demo samples (`DEMO DATA mode`, with the
exact path it checked).

Your existing data locations (`C:\Users\hp\recommender_output`, etc.) are
checked automatically — nothing to configure for local use.

---

## 2. File structure

```
fusionkitchen_package/
├── app.py              Entry point — run this file
├── config.py            File paths and settings (portable — see note below)
├── api_key.py            Paste up to 5 Gemini keys here directly
├── data_loader.py         Loads Step 1-4 pipeline artifacts
├── recommender.py         Cosine-similarity recommendation engine (Step 3 logic)
├── genai_service.py       OpenAI integration: grounded prompts, allergen flagging (Step 4 logic)
├── ui.py                 Gradio Blocks layout and result-card rendering
├── theme.py               CSS styling, kept separate from logic
├── demo_data.py           Sample data used if real artifacts aren't found
├── requirements.txt
├── tests/                 Automated test suite (see Section 3)
├── .github/workflows/     CI: runs the test suite on every push (see Section 4)
└── .gitignore
```

Each file has one job — change how recipes are scored, and only
`recommender.py` needs touching; change the OpenAI prompt, and only
`genai_service.py` does; restyle the app, and only `theme.py` does.

**On path portability:** `config.py` checks your real local Windows folders
first (so nothing changes for local use), and automatically falls back to a
`data/` folder next to the code if those aren't found — which is what
happens on Hugging Face's Linux servers, where `C:\Users\hp\...` doesn't
exist. This is what makes the same codebase run both on your machine and
once deployed, with zero code changes.

---

## 3. Testing

A real, runnable test suite — not just manual spot-checks — covering the
recommendation engine, the GenAI service's error handling, and the
demo-mode fallback:

```
pip install pytest
pytest tests/ -v
```

The tests build a small **synthetic** copy of the pipeline artifacts in a
temp folder (see `tests/conftest.py`) rather than depending on your real
2-million-row dataset — so the suite runs anywhere, including CI, in under
two seconds, without needing your actual data files present.

What's covered:
- Text normalization (`clean_formatting`, `canonicalize_text`) — including a
  regression test for the "cold water"/"warm water" filler-word bug found
  and fixed during Step 2c
- The recommender's scoring, top-N ranking, and graceful handling of
  unrecognized ingredients
- Allergen flagging and prompt construction
- The OpenAI call's fallback behavior when no key is set, and when the API
  call itself fails (must never crash the app)
- `PipelineData`'s fallback to demo mode when real files aren't found

---

## 4. Managing this as a GitHub repository

```
git init
git add .
git commit -m "FusionKitchen: Steps 1-5 application"
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

`.gitignore` already excludes `__pycache__/`, `.pytest_cache/`, **`api_key_local.py`**
(the one file that ever holds a real key — `api_key.py` itself stays
trackable since it never contains a secret, only the logic to find one),
and — **on purpose** — the `data/` folder. Your pipeline's real output files
are large (the Word2Vec recipe-vector matrix alone is roughly n_recipes ×
100 × 4 bytes — hundreds of MB at your dataset's full size) and don't
belong in a normal Git repository's history. Section 5 below covers
getting them onto your deployed Space without putting them in this Git repo.

A GitHub Actions workflow (`.github/workflows/tests.yml`) is already set
up to run the full `pytest` suite automatically on every push — since the
tests use synthetic data, this works with no setup on GitHub's side, no
secrets required, and gives you a green/red check on every commit.

---

## 5. Deploying to Hugging Face Spaces (100% Free Forever)

Hugging Face Spaces provides **16 GB RAM and 2 vCPUs completely for free**. To host your 5GB dataset without paying for persistent disk or exceeding Space Git-LFS storage quotas, store the dataset in a **free Hugging Face Dataset repository** and let the Space auto-download it on startup.

### Step 1 — Upload your 5GB Data to a Free Hugging Face Dataset
1. Go to [huggingface.co/new-dataset](https://huggingface.co/new-dataset).
2. Choose a name (e.g., `fusionkitchen-data`), set Visibility to **Public** (or Private), and click **Create dataset**.
3. Generate a free Write Token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
4. Run the upload helper script in your terminal:
   ```bash
   python upload_dataset_to_hf.py --repo <your-username>/fusionkitchen-data
   ```
   *(This automatically uploads `recommender_output`, `embeddings_output`, and `genai_output` directly to your dataset repo.)*

### Step 2 — Create the Free Hugging Face Space
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Space Name: `fusionkitchen`
3. License: `mit` (or choose yours)
4. SDK: **Gradio**
5. Hardware: **CPU basic (2 vCPU • 16GB RAM • Free)**
6. Click **Create Space**.

### Step 3 — Set Space Secrets & Variables
In your Space page: **Settings → Variables and secrets**:
- **New variable**:
  - Name: `HF_DATASET_REPO`
  - Value: `<your-username>/fusionkitchen-data`
- **New secret**:
  - Name: `OPENAI_API_KEY` (or `GEMINI_API_KEYS`)
  - Value: Your Gemini API Key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- *(Optional if dataset is Private)* **New secret**:
  - Name: `HF_TOKEN`
  - Value: Your Hugging Face Read/Write Token

### Step 4 — Push Code to Hugging Face Space
```bash
git remote add space https://huggingface.co/spaces/<your-username>/fusionkitchen
git push space main
```

The Space will build, automatically download and cache your 5GB dataset, and launch in **REAL DATA mode**. You can now copy the link and share it directly with your professor!

---

## Troubleshooting

- **"DEMO DATA mode" when you expected real data** — the app prints exactly
  which file path it checked and didn't find. Locally, check that path
  against where your `recommender_output` folder actually is. On a Space,
  check that Step 4's upload actually completed (see the Space's Files tab).
- **Recommendations work but explanations say "Add your API key..."**
  — expected if no key is set. Not an error.
- **A recipe's explanation shows "(GenAI call failed: ...)"** — check the
  error message shown; for Gemini this is usually an invalid/expired key,
  or the configured model requiring billing (Gemini's Pro models do; Flash
  and Flash-Lite, including the default `gemini-flash-latest`, don't —
  check https://ai.google.dev/gemini-api/docs/pricing if unsure). For
  OpenAI directly, it's usually billing/credits (check
  platform.openai.com → Billing). Either way, the error message from the
  API is shown directly so you can see exactly what went wrong.
- **A recipe's explanation shows "(Could not parse model response as
  JSON...)"** — rare, but can happen if a model wraps its answer in extra
  text despite the prompt's instructions. The fallback parser catches most
  cases; if it happens often, it's worth checking the raw error text shown
  to see what the model actually returned.
- **Tests fail after you edit `recommender.py` or `genai_service.py`** —
  good, that's the point. Read the failure message; it's telling you
  what behavior changed.

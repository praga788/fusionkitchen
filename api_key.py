"""
FusionKitchen — API Keys
==========================
Paste up to 5 free Gemini API keys below. Google's free tier gives each
key a limited number of requests per day (the error you hit —
"RESOURCE_EXHAUSTED" / 429 — is exactly this daily quota). Providing
several keys lets the app rotate across them, multiplying your effective
daily quota, and automatically falls back to the next key if the current
one hits its limit mid-search rather than failing the whole request.

Get keys at: https://aistudio.google.com/apikey — each Google account can
generate its own key; using 5 different keys (e.g. 5 separate Google
accounts, or 5 keys from the same account if Google allows it) gives you
roughly 5x the daily requests.

You're running this locally, so no separate gitignored file needed here —
just paste your keys directly below.
"""

import os

GEMINI_API_KEYS = [
    "PASTE_KEY_1_HERE",
    # "PASTE_KEY_2_HERE",
    # "PASTE_KEY_3_HERE",
    # "PASTE_KEY_4_HERE",
    # "PASTE_KEY_5_HERE",
]

# Support environment variables from Hugging Face Space secrets or .env
_env_single = os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
_env_multi = os.environ.get("GEMINI_API_KEYS")

if _env_multi:
    # Comma-separated list in environment variable
    parsed_multi = [k.strip() for k in _env_multi.split(",") if k.strip()]
    GEMINI_API_KEYS = parsed_multi + list(GEMINI_API_KEYS)
elif _env_single:
    GEMINI_API_KEYS = [_env_single.strip()] + list(GEMINI_API_KEYS)

# Drop unfilled placeholders/blanks so the rotation logic only ever sees real keys.
GEMINI_API_KEYS = [k for k in GEMINI_API_KEYS if k and not k.startswith("PASTE_KEY")]

# Kept for backward compatibility with anything checking a single key.
OPENAI_API_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else None

"""
FusionKitchen — Main Entry Point
===================================
Run with:  python app.py

Setup (once, in cmd prompt):
    pip install -r requirements.txt

Then edit api_key.py to add your OpenAI key (optional — the app runs
fully without one, just without live-generated text).
"""

import config
from data_loader import PipelineData
from recommender import RecipeRecommender
from ui import build_app
from theme import CUSTOM_CSS
import api_key


def main():
    data = PipelineData()
    recommender = RecipeRecommender(data)
    demo = build_app(data, recommender)

    print(f"\n{'REAL DATA' if data.is_real else 'DEMO DATA'} mode.")
    if data.is_real and not api_key.OPENAI_API_KEY:
        print("No OpenAI API key set in api_key.py — recommendations work fully, "
              "but explanations/tips will show a placeholder instead of live GenAI text.")

    demo.launch(
        css=CUSTOM_CSS,
        server_name=config.SERVER_NAME,
        server_port=config.SERVER_PORT,
        share=not config.IS_HF_SPACE,
    )


if __name__ == "__main__":
    main()

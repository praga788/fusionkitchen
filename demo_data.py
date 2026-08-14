"""
FusionKitchen — Demo Data
==========================
Used automatically when the real pipeline artifacts (Steps 1-4 outputs)
aren't found at the paths in config.py — so the app always runs, even
before you've pointed it at real data. Includes every field the real
GenAI service produces, so demo mode shows the full feature set too.
"""

DEMO_RECIPES = [
    {
        "title": "Tex-Mex Crab Bites", "source": "Recipes1M", "similarity_score": 0.97,
        "ingredients": ["crabmeat", "lemon juice", "green onion", "parsley", "worcestershire sauce", "pepper"],
        "explanation": "Matches your ingredients closely — a light, citrus-forward appetizer built around the crab.",
        "substitution": "Swap crabmeat for imitation crab if the real thing isn't on hand.",
        "healthier_alternative": "Use light mayonnaise (or Greek yogurt) in the binder to cut the fat without losing texture.",
        "cooking_tip": "Chill the mixture 20 minutes before serving so the flavors settle.",
        "summary": "Crabmeat is mixed with lemon juice, green onion, parsley, and Worcestershire sauce, then chilled and served as a light appetizer.",
        "allergen_flags": ["shellfish"],
    },
    {
        "title": "Classic Cinnamon Rolls", "source": "Recipes1M", "similarity_score": 0.91,
        "ingredients": ["yeast", "warm milk", "sugar", "butter", "salt", "egg", "flour", "brown sugar", "cinnamon"],
        "explanation": "A close match on your pantry staples — butter, sugar, and flour do the heavy lifting here.",
        "substitution": "Try margarine instead of butter for a dairy-lighter dough.",
        "healthier_alternative": "Swap half the all-purpose flour for whole wheat flour for more fiber without changing the method.",
        "cooking_tip": "Let the dough rise somewhere warm and draft-free for the best texture.",
        "summary": "A yeasted dough is enriched with butter and sugar, rolled with a brown sugar-cinnamon filling, then baked into rolls.",
        "allergen_flags": ["dairy", "gluten", "egg"],
    },
    {
        "title": "Coq Au Vin", "source": "Gathered", "similarity_score": 0.86,
        "ingredients": ["chicken", "button mushroom", "pearl onion", "carrot", "celery stalk", "salt pork"],
        "explanation": "A hearty braise that leans on the same base of chicken, aromatics, and root vegetables.",
        "substitution": "Turkey works in place of chicken if that's what's in the fridge.",
        "healthier_alternative": "Remove the chicken skin before browning to cut a meaningful amount of saturated fat.",
        "cooking_tip": "Brown the chicken in batches — crowding the pan steams it instead of searing it.",
        "summary": "Chicken is browned with salt pork, then braised slowly with mushrooms, pearl onions, carrots, and celery until tender.",
        "allergen_flags": [],
    },
    {
        "title": "Lightened Broccoli Salad", "source": "Recipes1M", "similarity_score": 0.79,
        "ingredients": ["broccoli floret", "light mayonnaise", "sugar substitute", "rice wine vinegar", "bacon"],
        "explanation": "A cold, crunchy match — good if you're after something that isn't stovetop cooking.",
        "substitution": "Greek yogurt can stand in for the light mayonnaise.",
        "healthier_alternative": "Use turkey bacon in place of regular bacon to reduce saturated fat while keeping the smoky flavor.",
        "cooking_tip": "Salt the broccoli lightly and let it sit 10 minutes to draw out excess water first.",
        "summary": "Raw broccoli florets are tossed in a light mayonnaise dressing with a sugar substitute, vinegar, and crumbled bacon.",
        "allergen_flags": ["dairy"],
    },
    {
        "title": "Breakfast Lasagna", "source": "Recipes1M", "similarity_score": 0.74,
        "ingredients": ["chickpea flour", "olive oil", "ground cumin", "kosher salt", "eggs"],
        "explanation": "An unconventional pick, but it shares more of your core ingredients than it first looks.",
        "substitution": "Chickpea flour can be swapped for regular flour in a pinch.",
        "healthier_alternative": "Use egg whites for part of the eggs to lower cholesterol while keeping the protein.",
        "cooking_tip": "Rest the batter 5 minutes before cooking — it thickens and browns more evenly.",
        "summary": "A chickpea-flour batter seasoned with cumin and salt is combined with eggs and cooked in layers, savory-brunch style.",
        "allergen_flags": ["egg"],
    },
]

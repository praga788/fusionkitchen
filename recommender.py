"""
FusionKitchen — Recommendation Engine
========================================
Turns a user's typed ingredients into recipe matches via cosine similarity
against pre-built recipe vectors (Step 3's output). Word2Vec and SBERT
scores are blended, not merged at the vector level, since their
dimensions (100 vs 384) aren't compatible to mix directly.
"""

import re
import numpy as np

# Same conservative filler-word list as Step 2c's canonicalization —
# reused here so a user's typed ingredients get normalized exactly the
# same way the recipe corpus already was.
FILLER_WORDS = {"extra", "additional", "virgin", "pure"}


def clean_formatting(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("+", " ")
    text = re.sub(r"\s*-\s*", "-", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonicalize_text(text: str) -> str:
    cleaned = clean_formatting(text)
    tokens = [t for t in cleaned.split(" ") if t not in FILLER_WORDS]
    result = " ".join(tokens).strip()
    return result if result else cleaned


class RecipeRecommender:
    """Wraps a PipelineData instance with the actual scoring logic."""

    def __init__(self, data):
        self.data = data

    def get_recipe_ingredients(self, recipe: dict) -> list:
        """Returns this recipe's ingredients as a proper list, preserving
        multi-word ingredients like "lemon juice" as ONE entry.

        recipe_lookup_table.parquet (built in Step 3) only stores `NER_text`
        — ingredients joined with spaces, e.g. "lemon juice olive oil" — so
        naively splitting on " " breaks "lemon juice" into two separate,
        wrong tokens. When available, directions_lookup.parquet (Step 4a)
        has `NER_clean_str`, the semicolon-joined form Step 1 exported
        specifically to avoid this problem — we prefer that here. Falls
        back to the naive space-split only if directions data isn't loaded.
        """
        if self.data.have_directions:
            # Check on-demand streaming lookup first
            if getattr(self.data, "directions_lookup", None) is not None:
                row = self.data.directions_lookup.get(recipe.get("title", ""), recipe.get("link", ""))
                if row:
                    ner_clean_str = row.get("NER_clean_str")
                    if isinstance(ner_clean_str, str) and ner_clean_str.strip():
                        return [t.strip() for t in ner_clean_str.split(";") if t.strip()]
            elif getattr(self.data, "directions_df", None) is not None:
                key = (recipe["title"], recipe["link"])
                if key in self.data.directions_df.index:
                    row = self.data.directions_df.loc[key]
                    if isinstance(row, type(self.data.directions_df)):  # duplicate key -> DataFrame, take first
                        row = row.iloc[0]
                    ner_clean_str = row.get("NER_clean_str")
                    if isinstance(ner_clean_str, str) and ner_clean_str.strip():
                        return [t.strip() for t in ner_clean_str.split(";") if t.strip()]
        return [t for t in recipe.get("NER_text", "").split(" ") if t]

    def _canonicalize_query(self, user_ingredients):
        out = []
        for ing in user_ingredients:
            norm = canonicalize_text(ing)
            out.append(self.data.raw_to_canonical.get(norm, norm))
        return out

    def _query_vector(self, canonical_terms, term_idx, embedding_matrix):
        if not term_idx:
            return None
        vecs = [embedding_matrix[term_idx[t]] for t in canonical_terms if t in term_idx]
        if not vecs:
            return None
        v = np.mean(np.array(vecs, dtype=np.float32), axis=0, keepdims=True)
        return v / np.linalg.norm(v).clip(min=1e-8)

    def get_substitute_candidates(self, ingredient: str, k=5):
        """Real nearest-neighbor substitutes from Word2Vec — used to ground
        the GenAI substitution suggestion instead of letting it invent one."""
        if not self.data.w2v_term_idx or ingredient not in self.data.w2v_term_idx:
            return []
        idx = self.data.w2v_term_idx[ingredient]
        sims = self.data.w2v_matrix @ self.data.w2v_matrix[idx]
        top_idx = np.argsort(-sims)[1:k + 1]  # skip index 0 — the ingredient itself
        terms = list(self.data.w2v_term_idx.keys())
        return [terms[i] for i in top_idx]

    def recommend(self, user_ingredients: list, top_n=5):
        """Returns (list of recipe dicts with similarity_score, set of
        canonical terms that were actually recognized)."""
        canonical_terms = self._canonicalize_query(user_ingredients)

        combined_score = np.zeros(len(self.data.recipe_lookup_df), dtype=np.float32)
        any_scored = False
        matched_terms = set()

        qv = self._query_vector(canonical_terms, self.data.w2v_term_idx, self.data.w2v_matrix)
        if qv is not None:
            combined_score += self.data.effective_alpha * (self.data.recipe_vectors["word2vec"] @ qv.T).flatten()
            any_scored = True
            matched_terms |= {t for t in canonical_terms if t in self.data.w2v_term_idx}

        if self.data.have_sbert:
            qv_s = self._query_vector(canonical_terms, self.data.sbert_term_idx, self.data.sbert_matrix)
            if qv_s is not None:
                combined_score += (1 - self.data.effective_alpha) * (self.data.recipe_vectors["sbert"] @ qv_s.T).flatten()
                any_scored = True
                matched_terms |= {t for t in canonical_terms if t in self.data.sbert_term_idx}

        if not any_scored:
            return [], set(canonical_terms)

        top_idx = np.argsort(-combined_score)[:top_n]
        results = self.data.recipe_lookup_df.iloc[top_idx].copy()
        results["similarity_score"] = combined_score[top_idx]
        return results.to_dict("records"), matched_terms

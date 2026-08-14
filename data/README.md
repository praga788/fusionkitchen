# data/

Not needed for local use — `config.py` finds your real data automatically
at your existing Windows folders (`C:\Users\hp\recommender_output`, etc.).

This folder only matters if you later deploy somewhere that can't see
those Windows paths (e.g. a cloud host). In that case, copy your Step 1-4
output here, matching this structure:

```
data/
├── recommender_output/
│   ├── recipe_vectors_word2vec.npy
│   ├── recipe_vectors_sbert.npy        (optional)
│   └── recipe_lookup_table.parquet
├── embeddings_output/
│   ├── word2vec_vectors_canonical.parquet
│   ├── sbert_vectors_canonical.parquet   (optional)
│   └── ingredient_canonical_mapping.parquet
└── genai_output/
    └── directions_lookup.parquet         (optional)
```

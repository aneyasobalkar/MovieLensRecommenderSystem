"""
02_collaborative_filtering.py
==============================
Train item-item CF, sanity check similar items, inspect recommendations.

Run:
    python notebooks/02_collaborative_filtering.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_artifacts, build_sparse_matrix
from src.collaborative_filtering import ItemItemCF

# ── CELL 1: Load data ─────────────────────────────────────────────────────────
ratings, movies, dense_matrix, sparse_matrix = load_artifacts()
user_index = list(dense_matrix.index)
item_index = list(dense_matrix.columns)

movie_lookup = movies.set_index("movie_id")["title"].to_dict()

print(f"Loaded: {len(user_index)} users, {len(item_index)} items")

# ── CELL 2: Train CF ──────────────────────────────────────────────────────────
cf = ItemItemCF(k_neighbors=20)
cf.fit(sparse_matrix, user_index, item_index)

# ── CELL 3: Sanity check — similar items ─────────────────────────────────────
# Star Wars is movie_id=50 in MovieLens 100K
STAR_WARS_ID = 50

similar = cf.similar_items(STAR_WARS_ID, n=10)
similar["title"] = similar["movie_id"].map(movie_lookup)
print("\nTop 10 similar movies to Star Wars:")
print(similar[["title", "similarity"]].to_string(index=False))

# ── CELL 4: Recommend for a specific user ─────────────────────────────────────
USER_ID = 1

# What has this user rated highly?
user_ratings = ratings[ratings["user_id"] == USER_ID].merge(movies, on="movie_id")
top_rated = user_ratings.sort_values("rating", ascending=False).head(5)
print(f"\nUser {USER_ID}'s top-rated movies:")
print(top_rated[["title", "rating"]].to_string(index=False))

# CF recommendations
recs = cf.recommend(USER_ID, n=10)
recs["title"] = recs["movie_id"].map(movie_lookup)
print(f"\nItem-Item CF recommendations for user {USER_ID}:")
print(recs[["title", "score"]].to_string(index=False))

# ── CELL 5: Visualize item similarity matrix (sample) ─────────────────────────
import seaborn as sns

# Use top 30 most-rated items for a readable heatmap
top_items = (
    ratings.groupby("movie_id")
    .size()
    .sort_values(ascending=False)
    .head(30)
    .index.tolist()
)
top_item_cols = [cf._item_to_col[mid] for mid in top_items if mid in cf._item_to_col]
top_item_titles = [movie_lookup.get(mid, str(mid))[:20] for mid in top_items if mid in cf._item_to_col]

sim_sample = cf.similarity_matrix[np.ix_(top_item_cols, top_item_cols)]

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    sim_sample,
    xticklabels=top_item_titles,
    yticklabels=top_item_titles,
    cmap="Blues",
    vmin=0,
    vmax=1,
    ax=ax,
    linewidths=0.2,
)
ax.set_title("Item-item cosine similarity (top 30 most-rated movies)")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("data/item_similarity_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: data/item_similarity_heatmap.png")

"""
01_data_setup.py
================
Download MovieLens 100K, build the user-item matrix, explore sparsity.

Run:
    python notebooks/01_data_setup.py
Or paste cells into a Jupyter notebook.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import prepare_data, load_artifacts

# ── CELL 1: Download & build ──────────────────────────────────────────────────
ratings, movies, dense_matrix, sparse_matrix, user_index, item_index = prepare_data()

# ── CELL 2: Basic stats ───────────────────────────────────────────────────────
print("\n--- Ratings sample ---")
print(ratings.head(10).to_string())

print("\n--- Rating distribution ---")
print(ratings["rating"].value_counts().sort_index())

print("\n--- Movies sample ---")
print(movies.head(10).to_string())

# ── CELL 3: Sparsity visualization ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Rating distribution
axes[0].bar(
    ratings["rating"].value_counts().sort_index().index,
    ratings["rating"].value_counts().sort_index().values,
    color="#378ADD",
    edgecolor="white",
)
axes[0].set_title("Rating distribution")
axes[0].set_xlabel("Rating (1–5)")
axes[0].set_ylabel("Count")

# Ratings per user
ratings_per_user = ratings.groupby("user_id").size()
axes[1].hist(ratings_per_user, bins=40, color="#1D9E75", edgecolor="white")
axes[1].set_title("Ratings per user")
axes[1].set_xlabel("Number of ratings")
axes[1].set_ylabel("Users")
axes[1].axvline(ratings_per_user.median(), color="#D85A30", linestyle="--", label=f"Median: {ratings_per_user.median():.0f}")
axes[1].legend()

# Ratings per movie
ratings_per_movie = ratings.groupby("movie_id").size()
axes[2].hist(ratings_per_movie, bins=40, color="#7F77DD", edgecolor="white")
axes[2].set_title("Ratings per movie")
axes[2].set_xlabel("Number of ratings")
axes[2].set_ylabel("Movies")
axes[2].axvline(ratings_per_movie.median(), color="#D85A30", linestyle="--", label=f"Median: {ratings_per_movie.median():.0f}")
axes[2].legend()

plt.tight_layout()
plt.savefig("data/eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: data/eda_plots.png")

# ── CELL 4: Sparsity heatmap (sample) ────────────────────────────────────────
# Show a 50×50 sample of the matrix to visualize sparsity
sample_matrix = dense_matrix.iloc[:50, :50].fillna(0)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    sample_matrix,
    cmap="Blues",
    cbar_kws={"label": "Rating"},
    linewidths=0,
    ax=ax,
)
ax.set_title("User-item matrix (first 50 users × 50 movies)\nBlue = rated, white = unrated")
ax.set_xlabel("Movie ID")
ax.set_ylabel("User ID")
plt.tight_layout()
plt.savefig("data/sparsity_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: data/sparsity_heatmap.png")

# ── CELL 5: Sparsity stats ────────────────────────────────────────────────────
n_users, n_items = sparse_matrix.shape
sparsity = 1 - sparse_matrix.nnz / (n_users * n_items)
print(f"\n{'='*40}")
print(f"Users:          {n_users:,}")
print(f"Items:          {n_items:,}")
print(f"Ratings:        {sparse_matrix.nnz:,}")
print(f"Sparsity:       {sparsity:.1%}")
print(f"Matrix cells:   {n_users * n_items:,}")
print(f"{'='*40}")
print(f"\nThis is why CF struggles and why matrix factorization helps:")
print(f"  {sparsity:.1%} of the matrix is unknown — most users haven't rated most movies.")
print(f"  MF finds k={50} latent dimensions that explain the observed ratings,")
print(f"  then uses those factors to fill in the blanks.")

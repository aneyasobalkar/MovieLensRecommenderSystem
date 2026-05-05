"""
03_matrix_factorization.py
===========================
Train SVD and NMF matrix factorization models.
Inspect latent factors. Compare recommendations to CF.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.data_loader import load_artifacts
from src.matrix_factorization import MFRecommender, make_surprise_dataset
from surprise import SVD as SurpriseSVD
from surprise.model_selection import cross_validate


# ── LOAD DATA ────────────────────────────────────────────────────────────────
ratings, movies, _, _ = load_artifacts()

movie_lookup = movies.set_index("movie_id")["title"].to_dict()
all_movie_ids = movies["movie_id"].tolist()


# ── TRAIN MODELS ─────────────────────────────────────────────────────────────
print("Training SVD...")
svd = MFRecommender(algorithm="svd", n_factors=50, n_epochs=20)
svd.fit(ratings)

print("Training NMF...")
nmf = MFRecommender(algorithm="nmf", n_factors=50, n_epochs=50)
nmf.fit(ratings)


# ── RECOMMENDATIONS ──────────────────────────────────────────────────────────
USER_ID = 1
rated = set(ratings[ratings["user_id"] == USER_ID]["movie_id"])

svd_recs = svd.recommend(USER_ID, all_movie_ids, rated, n=10)
nmf_recs = nmf.recommend(USER_ID, all_movie_ids, rated, n=10)

svd_recs["title"] = svd_recs["movie_id"].map(movie_lookup)
nmf_recs["title"] = nmf_recs["movie_id"].map(movie_lookup)

print(f"\nSVD recommendations for user {USER_ID}:")
print(svd_recs[["title", "score"]].to_string(index=False))

print(f"\nNMF recommendations for user {USER_ID}:")
print(nmf_recs[["title", "score"]].to_string(index=False))


# ── LATENT SPACE VISUALIZATION ───────────────────────────────────────────────
print("\nBuilding latent factor visualization...")

item_factors = svd.get_item_factors()

pca = PCA(n_components=2, random_state=42)
item_2d = pca.fit_transform(item_factors)


# ── MAP MOVIES TO 2D SPACE SAFELY ────────────────────────────────────────────
trainset = svd._trainset

top_movie_ids = (
    ratings.groupby("movie_id")
    .size()
    .sort_values(ascending=False)
    .head(200)
    .index
    .tolist()
)

coords = []
labels = []

for mid in top_movie_ids:
    try:
        iid = trainset.to_inner_iid(mid)   # FIX: no str()
        coords.append(item_2d[iid])
        labels.append(movie_lookup.get(mid, str(mid))[:20])
    except ValueError:
        continue


coords = np.array(coords)

if coords.ndim != 2 or len(coords) == 0:
    raise ValueError(
        "No valid coordinates for plotting. "
        "Check movie_id alignment with Surprise trainset."
    )


# ── PLOT ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 10))
ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=20)

ax.set_title(
    "Item latent factors (SVD) projected to 2D via PCA\n"
    "Clusters reflect shared genres/moods"
)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")


# Highlight famous movies (optional)
highlight = {
    50: "Star Wars",
    100: "Fargo",
    127: "Godfather",
    172: "Empire Strikes Back",
    181: "Return of Jedi"
}

for mid, name in highlight.items():
    try:
        iid = trainset.to_inner_iid(mid)
        x, y = item_2d[iid]
        ax.scatter([x], [y], s=60)
        ax.annotate(name, (x, y), fontsize=8, xytext=(5, 5),
                    textcoords="offset points")
    except ValueError:
        pass


plt.tight_layout()
plt.show()


# ── HYPERPARAMETER SWEEP ────────────────────────────────────────────────────
print("\nSweeping SVD latent factors...")

data = make_surprise_dataset(ratings)

results = []
for k in [10, 20, 50, 100, 150]:
    model = SurpriseSVD(n_factors=k, n_epochs=20, random_state=42)
    cv = cross_validate(model, data, measures=["RMSE"], cv=5, verbose=False)

    rmse = cv["test_rmse"].mean()
    results.append((k, rmse))

    print(f"n_factors={k} → RMSE={rmse:.4f}")


df = pd.DataFrame(results, columns=["n_factors", "rmse"])

plt.figure(figsize=(7, 4))
plt.plot(df["n_factors"], df["rmse"], marker="o")
plt.title("SVD performance vs latent factors")
plt.xlabel("n_factors")
plt.ylabel("RMSE")
plt.grid(True, alpha=0.3)
plt.show()
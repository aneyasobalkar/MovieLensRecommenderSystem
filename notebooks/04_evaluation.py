"""
04_evaluation.py
=================
Proper train/test evaluation of all three models.
Outputs the results table that goes in your README.

Run:
    python notebooks/04_evaluation.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.data_loader import load_artifacts, build_user_item_matrix, build_sparse_matrix
from src.collaborative_filtering import ItemItemCF
from src.matrix_factorization import MFRecommender, split_dataset
from src.evaluation import (
    evaluate_cf, evaluate_mf, results_table,
    precision_at_k, recall_at_k, ndcg_at_k, build_ground_truth, rmse
)

# ── CELL 1: Load & split ──────────────────────────────────────────────────────
ratings, movies, dense_matrix, sparse_matrix = load_artifacts()
all_movie_ids = list(movies["movie_id"])
movie_lookup = movies.set_index("movie_id")["title"].to_dict()

# 80/20 split by rating row (stratified by user for better coverage)
train_ratings, test_ratings = train_test_split(
    ratings, test_size=0.2, random_state=42, stratify=ratings["user_id"]
    if ratings["user_id"].value_counts().min() >= 2
    else None
)

print(f"Train ratings: {len(train_ratings):,}")
print(f"Test ratings:  {len(test_ratings):,}")

# ── CELL 2: Train CF on train split ───────────────────────────────────────────
train_dense, train_user_idx, train_item_idx = build_user_item_matrix(train_ratings)
train_sparse = build_sparse_matrix(train_dense)

cf = ItemItemCF(k_neighbors=20)
cf.fit(train_sparse, train_user_idx, train_item_idx)

# ── CELL 3: Evaluate CF ───────────────────────────────────────────────────────
print("\nEvaluating Item-Item CF...")
cf_results = evaluate_cf(
    cf,
    test_ratings=test_ratings,
    train_ratings=train_ratings,
    k_values=[5, 10],
    relevance_threshold=4.0,
)
print("CF results:", cf_results)

# ── CELL 4: Train & evaluate SVD ─────────────────────────────────────────────
from src.matrix_factorization import make_surprise_dataset
from surprise.model_selection import train_test_split as surprise_split

data = make_surprise_dataset(ratings)
surprise_trainset, surprise_testset = surprise_split(data, test_size=0.2, random_state=42)

svd = MFRecommender(algorithm="svd", n_factors=50, n_epochs=20)
svd.fit_trainset(surprise_trainset)

# Convert surprise testset back to DataFrame for ranking metrics
test_df = pd.DataFrame(
    [(int(uid), int(iid), r) for uid, iid, r in surprise_testset],
    columns=["user_id", "movie_id", "rating"]
)
train_df = pd.DataFrame(
    [(int(surprise_trainset.to_raw_uid(u)),
      int(surprise_trainset.to_raw_iid(i)),
      surprise_trainset.ur[u][j][1])
     for u in surprise_trainset.all_users()
     for j, (i, _) in enumerate(surprise_trainset.ur[u])],
    columns=["user_id", "movie_id", "rating"]
)

print("\nEvaluating SVD...")
svd_results = evaluate_mf(
    svd,
    testset=surprise_testset,
    test_ratings=test_df,
    train_ratings=train_df,
    all_movie_ids=all_movie_ids,
    k_values=[5, 10],
    relevance_threshold=4.0,
)
print("SVD results:", svd_results)

# ── CELL 5: Train & evaluate NMF ─────────────────────────────────────────────
nmf = MFRecommender(algorithm="nmf", n_factors=50, n_epochs=50)
nmf.fit_trainset(surprise_trainset)

print("\nEvaluating NMF...")
nmf_results = evaluate_mf(
    nmf,
    testset=surprise_testset,
    test_ratings=test_df,
    train_ratings=train_df,
    all_movie_ids=all_movie_ids,
    k_values=[5, 10],
    relevance_threshold=4.0,
)
print("NMF results:", nmf_results)

# ── CELL 6: Results table ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS TABLE (copy this into your README)")
print("="*60)
table = results_table(cf_results, svd_results, nmf_results)
print(table.to_string())
print()

# Markdown version for README
print("\n--- Markdown version ---")
print("| Model | Precision@5 | NDCG@5 | Precision@10 | NDCG@10 | RMSE |")
print("|---|---|---|---|---|---|")
for name, res in [("Item-Item CF", cf_results), ("SVD", svd_results), ("NMF", nmf_results)]:
    p5 = res.get("precision@5", "-")
    n5 = res.get("ndcg@5", "-")
    p10 = res.get("precision@10", "-")
    n10 = res.get("ndcg@10", "-")
    r = res.get("rmse", "-")
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else v
    print(f"| {name} | {fmt(p5)} | {fmt(n5)} | {fmt(p10)} | {fmt(n10)} | {fmt(r)} |")

# ── CELL 7: Bar chart comparison ─────────────────────────────────────────────
metrics = ["precision@5", "ndcg@5", "precision@10", "ndcg@10"]
model_names = ["Item-Item CF", "SVD (MF)", "NMF (MF)"]
model_results = [cf_results, svd_results, nmf_results]

x = np.arange(len(metrics))
width = 0.25
colors = ["#378ADD", "#1D9E75", "#7F77DD"]

fig, ax = plt.subplots(figsize=(11, 5))
for i, (name, res, color) in enumerate(zip(model_names, model_results, colors)):
    vals = [res.get(m, 0) for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name, color=color, edgecolor="white")

ax.set_title("Model comparison: Precision@K and NDCG@K")
ax.set_xticks(x + width)
ax.set_xticklabels(["Precision@5", "NDCG@5", "Precision@10", "NDCG@10"])
ax.set_ylabel("Score")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
ax.set_ylim(0, max(
    max(res.get(m, 0) for m in metrics for res in model_results) * 1.2,
    0.1
))
plt.tight_layout()
plt.savefig("data/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: data/model_comparison.png")

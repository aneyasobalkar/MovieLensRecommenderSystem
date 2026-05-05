"""
evaluation.py
Evaluation metrics for recommendation systems.

Metrics implemented:
  - RMSE          : standard rating-prediction accuracy
  - Precision@K   : fraction of top-K recs that are "relevant" (rating >= threshold)
  - Recall@K      : fraction of relevant items captured in top-K recs
  - NDCG@K        : normalized discounted cumulative gain — rewards ranking
                    highly-rated items higher in the list

Why NDCG matters for interviews:
    Precision@K treats all relevant items equally — being rank 1 vs rank K
    is the same. NDCG@K penalizes relevant items buried lower in the list,
    which is closer to how users actually experience recommendations.
    Mentioning both shows you understand the difference between
    classification metrics and ranking metrics.
"""

import numpy as np
import pandas as pd
from surprise import accuracy
from typing import Callable
from tqdm import tqdm


# ── rating-level metric ───────────────────────────────────────────────────────

def rmse(predictions) -> float:
    """Compute RMSE from a list of surprise Prediction objects."""
    return accuracy.rmse(predictions, verbose=False)


def mae(predictions) -> float:
    """Compute MAE from a list of surprise Prediction objects."""
    return accuracy.mae(predictions, verbose=False)


# ── ranking metrics ───────────────────────────────────────────────────────────

def precision_at_k(
    recommendations: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int = 10,
) -> float:
    """
    Precision@K averaged across users.

    Args:
        recommendations: {user_id: [movie_id, ...]} — ordered top-K list per user.
        ground_truth: {user_id: set(movie_id)} — held-out relevant items per user.
        k: cutoff.

    Returns:
        Mean Precision@K across users who have ground-truth items.
    """
    scores = []
    for uid, recs in recommendations.items():
        if uid not in ground_truth or not ground_truth[uid]:
            continue
        top_k = recs[:k]
        hits = sum(1 for mid in top_k if mid in ground_truth[uid])
        scores.append(hits / k)
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(
    recommendations: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int = 10,
) -> float:
    """
    Recall@K averaged across users.

    Args:
        recommendations: {user_id: [movie_id, ...]}
        ground_truth: {user_id: set(movie_id)}
        k: cutoff.
    """
    scores = []
    for uid, recs in recommendations.items():
        if uid not in ground_truth or not ground_truth[uid]:
            continue
        top_k = set(recs[:k])
        hits = len(top_k & ground_truth[uid])
        scores.append(hits / len(ground_truth[uid]))
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(
    recommendations: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int = 10,
) -> float:
    """
    NDCG@K averaged across users.

    DCG@K = sum_{i=1}^{K}  rel_i / log2(i + 1)
    IDCG@K = DCG of perfect ranking (all relevant items first)
    NDCG@K = DCG@K / IDCG@K

    Args:
        recommendations: {user_id: [movie_id, ...]}
        ground_truth: {user_id: set(movie_id)}
        k: cutoff.
    """
    scores = []
    for uid, recs in recommendations.items():
        if uid not in ground_truth or not ground_truth[uid]:
            continue
        top_k = recs[:k]
        # DCG
        dcg = sum(
            1.0 / np.log2(i + 2)
            for i, mid in enumerate(top_k)
            if mid in ground_truth[uid]
        )
        # Ideal DCG — best possible ranking given the number of relevant items
        n_relevant = min(len(ground_truth[uid]), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


# ── evaluation pipeline ───────────────────────────────────────────────────────

def build_ground_truth(
    test_ratings: pd.DataFrame,
    relevance_threshold: float = 4.0,
) -> dict[int, set[int]]:
    """
    Build ground-truth relevant items from held-out test ratings.

    Args:
        test_ratings: DataFrame [user_id, movie_id, rating]
        relevance_threshold: minimum rating to count as "relevant"

    Returns:
        {user_id: set(relevant_movie_ids)}
    """
    relevant = test_ratings[test_ratings["rating"] >= relevance_threshold]
    return (
        relevant.groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )


def evaluate_cf(
    cf_model,
    test_ratings: pd.DataFrame,
    train_ratings: pd.DataFrame,
    k_values: list[int] = [5, 10],
    relevance_threshold: float = 4.0,
) -> dict:
    """
    Evaluate an ItemItemCF model on held-out test ratings.

    Args:
        cf_model: fitted ItemItemCF instance.
        test_ratings: held-out DataFrame [user_id, movie_id, rating].
        train_ratings: training DataFrame (to determine "already rated" items).
        k_values: list of K cutoffs to evaluate.
        relevance_threshold: min rating to count as relevant.

    Returns:
        dict of metric_name -> value.
    """
    ground_truth = build_ground_truth(test_ratings, relevance_threshold)
    test_users = list(ground_truth.keys())

    # Build per-user "already rated" sets from train
    train_rated = (
        train_ratings.groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    # Generate recommendations for each test user
    all_movie_ids = cf_model.item_index
    recommendations = {}
    print(f"Generating CF recommendations for {len(test_users)} users...")
    for uid in tqdm(test_users):
        if uid not in cf_model._user_to_row:
            continue
        recs_df = cf_model.recommend(
            uid,
            n=max(k_values),
            exclude_rated=True,
        )
        recommendations[uid] = recs_df["movie_id"].tolist()

    results = {}
    for k in k_values:
        results[f"precision@{k}"] = precision_at_k(recommendations, ground_truth, k)
        results[f"recall@{k}"] = recall_at_k(recommendations, ground_truth, k)
        results[f"ndcg@{k}"] = ndcg_at_k(recommendations, ground_truth, k)

    return results


def evaluate_mf(
    mf_model,
    testset,
    test_ratings: pd.DataFrame,
    train_ratings: pd.DataFrame,
    all_movie_ids: list,
    k_values: list[int] = [5, 10],
    relevance_threshold: float = 4.0,
) -> dict:
    """
    Evaluate an MFRecommender model.

    Args:
        mf_model: fitted MFRecommender instance.
        testset: surprise testset (list of (uid, iid, rating) tuples).
        test_ratings: test DataFrame for ranking metrics.
        train_ratings: training DataFrame.
        all_movie_ids: list of all candidate movie_ids.
        k_values: K cutoffs.
        relevance_threshold: min rating to count as relevant.
    """
    # Rating-level metrics
    predictions = mf_model.model.test(testset)
    results = {
        "rmse": rmse(predictions),
        "mae": mae(predictions),
    }

    # Ranking metrics
    ground_truth = build_ground_truth(test_ratings, relevance_threshold)
    test_users = list(ground_truth.keys())

    train_rated = (
        train_ratings.groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )

    recommendations = {}
    print(f"Generating MF recommendations for {len(test_users)} users...")
    for uid in tqdm(test_users):
        rated = train_rated.get(uid, set())
        recs_df = mf_model.recommend(
            uid,
            all_movie_ids=all_movie_ids,
            rated_movie_ids=rated,
            n=max(k_values),
            exclude_rated=True,
        )
        recommendations[uid] = recs_df["movie_id"].tolist()

    for k in k_values:
        results[f"precision@{k}"] = precision_at_k(recommendations, ground_truth, k)
        results[f"recall@{k}"] = recall_at_k(recommendations, ground_truth, k)
        results[f"ndcg@{k}"] = ndcg_at_k(recommendations, ground_truth, k)

    return results


def results_table(cf_results: dict, svd_results: dict, nmf_results: dict = None) -> pd.DataFrame:
    """
    Format evaluation results as a clean comparison DataFrame.
    Suitable for printing or dropping into a README.
    """
    rows = {"Item-Item CF": cf_results, "SVD (MF)": svd_results}
    if nmf_results:
        rows["NMF (MF)"] = nmf_results

    df = pd.DataFrame(rows).T
    # Round for display
    return df.round(4)

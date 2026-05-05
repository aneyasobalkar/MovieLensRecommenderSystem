"""
recommender.py
Loads both models (Item-Item CF and SVD) at startup and serves predictions.
New app users are deterministically mapped onto a MovieLens training user so
recommendations are meaningful from day one.
"""
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_loader import load_ratings, load_movies, build_user_item_matrix, build_sparse_matrix
from src.collaborative_filtering import ItemItemCF
from src.matrix_factorization import MFRecommender

ML_N_USERS = 943  # number of users in MovieLens 100K training set


class ModelStore:
    ratings: Optional[pd.DataFrame] = None
    movies: Optional[pd.DataFrame] = None
    movie_lookup: dict = {}        # movie_id -> title
    all_movie_ids: list = []
    user_rated: dict = {}          # ml_user_id -> set of rated movie_ids
    cf_model: Optional[ItemItemCF] = None
    svd_model: Optional[MFRecommender] = None


_store = ModelStore()


def assign_ml_user_id(app_user_id: int) -> int:
    """Map an app user to a MovieLens user (1-943) deterministically."""
    return ((app_user_id - 1) % ML_N_USERS) + 1


def train_and_load(data_dir: str) -> None:
    print("Loading MovieLens data …")
    _store.ratings = load_ratings(data_dir)
    _store.movies = load_movies(data_dir)
    _store.movie_lookup = dict(zip(_store.movies["movie_id"], _store.movies["title"]))
    _store.all_movie_ids = _store.movies["movie_id"].tolist()
    _store.user_rated = (
        _store.ratings.groupby("user_id")["movie_id"].apply(set).to_dict()
    )

    dense, user_index, item_index = build_user_item_matrix(_store.ratings)
    sparse = build_sparse_matrix(dense)

    print("Training Item-Item CF (variant A) …")
    _store.cf_model = ItemItemCF(k_neighbors=20)
    _store.cf_model.fit(sparse, user_index, item_index)

    print("Training SVD (variant B) …")
    _store.svd_model = MFRecommender(algorithm="svd", n_factors=50, n_epochs=20)
    _store.svd_model.fit(_store.ratings)

    print("Both models ready.")


def get_recommendations(ml_user_id: int, variant: str, n: int = 10) -> list[dict]:
    rated = _store.user_rated.get(ml_user_id, set())

    if variant == "A":
        if ml_user_id in _store.cf_model._user_to_row:
            df = _store.cf_model.recommend(ml_user_id, n=n, exclude_rated=True)
        else:
            df = _popular_fallback(rated, n)
    else:
        df = _store.svd_model.recommend(
            ml_user_id,
            all_movie_ids=_store.all_movie_ids,
            rated_movie_ids=rated,
            n=n,
            exclude_rated=True,
        )

    return [
        {
            "movie_id": int(row.movie_id),
            "title": _store.movie_lookup.get(int(row.movie_id), "Unknown"),
            "score": round(float(row.score), 4),
        }
        for _, row in df.iterrows()
    ]


def _popular_fallback(rated: set, n: int) -> pd.DataFrame:
    counts = (
        _store.ratings.groupby("movie_id")["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "score"})
    )
    counts = counts[~counts["movie_id"].isin(rated)]
    return counts.sort_values("score", ascending=False).head(n)


def get_store() -> ModelStore:
    return _store

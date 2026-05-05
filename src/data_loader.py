"""
data_loader.py
Download MovieLens 100K, parse ratings + movie titles,
build user-item matrices (dense + sparse), and persist to disk.
"""

import os
import io
import pickle
import zipfile
import requests
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ── download ──────────────────────────────────────────────────────────────────

def download_movielens(data_dir: str = DATA_DIR) -> str:
    """Download and extract MovieLens 100K. Returns path to extracted folder."""
    os.makedirs(data_dir, exist_ok=True)
    extract_path = os.path.join(data_dir, "ml-100k")

    if os.path.isdir(extract_path):
        print("MovieLens 100K already downloaded.")
        return extract_path

    print("Downloading MovieLens 100K...")
    r = requests.get(MOVIELENS_URL, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(data_dir)
    print(f"Extracted to {extract_path}")
    return extract_path


# ── load raw data ─────────────────────────────────────────────────────────────

def load_ratings(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Load u.data → DataFrame with columns:
        user_id (int), movie_id (int), rating (float), timestamp (int)
    """
    path = os.path.join(data_dir, "ml-100k", "u.data")
    df = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    df["rating"] = df["rating"].astype(float)
    return df


def load_movies(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Load u.item → DataFrame with columns:
        movie_id (int), title (str)
    """
    path = os.path.join(data_dir, "ml-100k", "u.item")
    df = pd.read_csv(
        path,
        sep="|",
        encoding="latin-1",
        usecols=[0, 1],
        names=["movie_id", "title"],
        header=None,
    )
    return df


# ── build matrices ────────────────────────────────────────────────────────────

def build_user_item_matrix(ratings: pd.DataFrame):
    """
    Build a dense user × item rating matrix (NaN = unrated, NOT filled).
    Returns:
        matrix (pd.DataFrame): shape (n_users, n_items)
        user_index (list[int]): user_ids in row order
        item_index (list[int]): movie_ids in column order
    """
    matrix = ratings.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
    )
    return matrix, list(matrix.index), list(matrix.columns)


def build_sparse_matrix(dense_matrix: pd.DataFrame) -> csr_matrix:
    """Convert the dense (NaN-filled) matrix to CSR sparse (0 where unrated)."""
    return csr_matrix(dense_matrix.fillna(0).values)


def sparsity_report(sparse: csr_matrix) -> dict:
    """Return a dict with basic sparsity stats — useful for README / notebook."""
    n_users, n_items = sparse.shape
    n_ratings = sparse.nnz
    sparsity = 1 - n_ratings / (n_users * n_items)
    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_ratings": n_ratings,
        "sparsity": sparsity,
        "avg_ratings_per_user": n_ratings / n_users,
        "avg_ratings_per_item": n_ratings / n_items,
    }


# ── persist ───────────────────────────────────────────────────────────────────

def save_artifacts(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    dense_matrix: pd.DataFrame,
    sparse_matrix: csr_matrix,
    data_dir: str = DATA_DIR,
) -> None:
    os.makedirs(data_dir, exist_ok=True)
    ratings.to_parquet(os.path.join(data_dir, "ratings.parquet"), index=False)
    movies.to_parquet(os.path.join(data_dir, "movies.parquet"), index=False)
    dense_matrix.to_parquet(os.path.join(data_dir, "user_item_dense.parquet"))
    with open(os.path.join(data_dir, "user_item_sparse.pkl"), "wb") as f:
        pickle.dump(sparse_matrix, f)
    print("Saved: ratings, movies, dense matrix, sparse matrix.")


def load_artifacts(data_dir: str = DATA_DIR):
    """Load pre-saved artifacts. Returns (ratings, movies, dense_matrix, sparse_matrix)."""
    ratings = pd.read_parquet(os.path.join(data_dir, "ratings.parquet"))
    movies = pd.read_parquet(os.path.join(data_dir, "movies.parquet"))
    dense_matrix = pd.read_parquet(os.path.join(data_dir, "user_item_dense.parquet"))
    with open(os.path.join(data_dir, "user_item_sparse.pkl"), "rb") as f:
        sparse_matrix = pickle.load(f)
    return ratings, movies, dense_matrix, sparse_matrix


# ── convenience entry-point ───────────────────────────────────────────────────

def prepare_data(data_dir: str = DATA_DIR):
    """Full pipeline: download → load → build matrices → save → return."""
    download_movielens(data_dir)
    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)
    dense_matrix, user_index, item_index = build_user_item_matrix(ratings)
    sparse_matrix = build_sparse_matrix(dense_matrix)

    stats = sparsity_report(sparse_matrix)
    print(f"\n--- Dataset stats ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v:,}")

    save_artifacts(ratings, movies, dense_matrix, sparse_matrix, data_dir)
    return ratings, movies, dense_matrix, sparse_matrix, user_index, item_index


if __name__ == "__main__":
    prepare_data()

"""
collaborative_filtering.py
Item-item collaborative filtering using cosine similarity.

Why item-item over user-user?
  - Items are denser than users on MovieLens (each item rated by more people
    than each person has rated items), giving more stable similarity estimates.
  - Item similarity is static; user similarity shifts as preferences evolve.
  - Scales better: n_items (1682) << n_users (943) for 100K, but this
    advantage grows massively on real-world datasets.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional


class ItemItemCF:
    """
    Item-item collaborative filtering recommender.

    Usage:
        cf = ItemItemCF(k_neighbors=20)
        cf.fit(user_item_sparse, user_index, item_index)
        recs = cf.recommend(user_id=42, n=10)
    """

    def __init__(self, k_neighbors: int = 20):
        """
        Args:
            k_neighbors: number of similar items to aggregate per rated item.
        """
        self.k_neighbors = k_neighbors
        self.similarity_matrix: Optional[np.ndarray] = None
        self.user_item_matrix: Optional[np.ndarray] = None
        self.user_index: Optional[list] = None
        self.item_index: Optional[list] = None
        self._user_to_row: dict = {}
        self._item_to_col: dict = {}
        self._col_to_item: dict = {}

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        sparse_matrix: csr_matrix,
        user_index: list,
        item_index: list,
    ) -> "ItemItemCF":
        """
        Compute item-item cosine similarity matrix.

        Args:
            sparse_matrix: CSR matrix, shape (n_users, n_items), 0 = unrated.
            user_index: list of user_ids matching row order.
            item_index: list of movie_ids matching column order.
        """
        self.user_item_matrix = sparse_matrix.toarray().astype(np.float32)
        self.user_index = user_index
        self.item_index = item_index
        self._user_to_row = {uid: i for i, uid in enumerate(user_index)}
        self._item_to_col = {iid: j for j, iid in enumerate(item_index)}
        self._col_to_item = {j: iid for j, iid in enumerate(item_index)}

        print("Computing item-item cosine similarity... ", end="", flush=True)
        # Transpose: similarity between items means comparing their user-rating vectors
        item_vectors = sparse_matrix.T  # shape (n_items, n_users)
        self.similarity_matrix = cosine_similarity(item_vectors, dense_output=True)
        np.fill_diagonal(self.similarity_matrix, 0)  # item not similar to itself
        print("done.")
        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict_score(self, user_id: int, movie_id: int) -> float:
        """
        Predict rating for (user, item) pair using weighted neighbor average.
        Returns 0.0 if user or item not in training data.
        """
        if user_id not in self._user_to_row or movie_id not in self._item_to_col:
            return 0.0

        row = self._user_to_row[user_id]
        col = self._item_to_col[movie_id]

        user_ratings = self.user_item_matrix[row]  # shape (n_items,)
        item_sims = self.similarity_matrix[col]     # shape (n_items,)

        # Only use items this user has actually rated
        rated_mask = user_ratings > 0
        if not rated_mask.any():
            return 0.0

        # Top-k neighbors among rated items
        neighbor_sims = item_sims * rated_mask
        top_k_idx = np.argpartition(neighbor_sims, -self.k_neighbors)[-self.k_neighbors:]

        sims = neighbor_sims[top_k_idx]
        ratings = user_ratings[top_k_idx]

        denom = np.abs(sims).sum()
        if denom == 0:
            return 0.0
        return float(np.dot(sims, ratings) / denom)

    # ── recommend ─────────────────────────────────────────────────────────────

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_rated: bool = True,
    ) -> pd.DataFrame:
        """
        Return top-n movie recommendations for a user.

        Args:
            user_id: the user to recommend for.
            n: number of recommendations.
            exclude_rated: if True, exclude movies the user already rated.

        Returns:
            DataFrame with columns [movie_id, score] sorted descending.
        """
        if user_id not in self._user_to_row:
            raise ValueError(f"user_id {user_id} not in training data.")

        row = self._user_to_row[user_id]
        user_ratings = self.user_item_matrix[row]  # shape (n_items,)

        # Score all unrated (or all) items
        scores = []
        for col, movie_id in self._col_to_item.items():
            if exclude_rated and user_ratings[col] > 0:
                continue
            score = self._fast_predict(row, col, user_ratings)
            scores.append((movie_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return pd.DataFrame(scores[:n], columns=["movie_id", "score"])

    def _fast_predict(self, row: int, col: int, user_ratings: np.ndarray) -> float:
        """Internal predict — reuses pre-loaded user_ratings array."""
        item_sims = self.similarity_matrix[col]
        rated_mask = user_ratings > 0
        if not rated_mask.any():
            return 0.0
        neighbor_sims = item_sims * rated_mask
        top_k_idx = np.argpartition(neighbor_sims, -self.k_neighbors)[-self.k_neighbors:]
        sims = neighbor_sims[top_k_idx]
        ratings = user_ratings[top_k_idx]
        denom = np.abs(sims).sum()
        if denom == 0:
            return 0.0
        return float(np.dot(sims, ratings) / denom)

    # ── utils ─────────────────────────────────────────────────────────────────

    def similar_items(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        """
        Return the n most similar movies to a given movie.
        Useful for sanity checks (Star Wars neighbors should look right).
        """
        if movie_id not in self._item_to_col:
            raise ValueError(f"movie_id {movie_id} not in training data.")
        col = self._item_to_col[movie_id]
        sims = self.similarity_matrix[col]
        top_idx = np.argsort(sims)[::-1][:n]
        return pd.DataFrame(
            [(self._col_to_item[i], sims[i]) for i in top_idx],
            columns=["movie_id", "similarity"],
        )

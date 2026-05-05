"""
tests/test_metrics.py
Unit tests for Precision@K and NDCG@K.
Run: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np

# Import directly from the module file to avoid triggering surprise import in __init__.py
import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location(
    "evaluation",
    pathlib.Path(__file__).parent.parent / "src" / "evaluation.py"
)
_eval = importlib.util.module_from_spec(_spec)

# Stub out surprise.accuracy so the module loads without the library
import types, unittest.mock
_surprise_stub = types.ModuleType("surprise")
_surprise_stub.accuracy = unittest.mock.MagicMock()
sys.modules.setdefault("surprise", _surprise_stub)

_spec.loader.exec_module(_eval)
precision_at_k = _eval.precision_at_k
recall_at_k = _eval.recall_at_k
ndcg_at_k = _eval.ndcg_at_k


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def perfect_recs():
    """All recommended items are relevant."""
    return {
        1: [10, 20, 30, 40, 50],
        2: [11, 21, 31, 41, 51],
    }

@pytest.fixture
def perfect_gt():
    return {
        1: {10, 20, 30, 40, 50},
        2: {11, 21, 31, 41, 51},
    }

@pytest.fixture
def zero_recs():
    """No recommended items are relevant."""
    return {
        1: [99, 98, 97, 96, 95],
    }

@pytest.fixture
def zero_gt():
    return {
        1: {10, 20, 30},
    }


# ── precision@K ───────────────────────────────────────────────────────────────

def test_precision_perfect(perfect_recs, perfect_gt):
    assert precision_at_k(perfect_recs, perfect_gt, k=5) == pytest.approx(1.0)

def test_precision_zero(zero_recs, zero_gt):
    assert precision_at_k(zero_recs, zero_gt, k=5) == pytest.approx(0.0)

def test_precision_partial():
    recs = {1: [10, 99, 20, 98, 97]}  # 2 out of 5 relevant
    gt = {1: {10, 20, 30}}
    assert precision_at_k(recs, gt, k=5) == pytest.approx(2 / 5)

def test_precision_k_cutoff():
    recs = {1: [99, 99, 99, 99, 10]}  # relevant item at position 5
    gt = {1: {10}}
    # k=4: item 10 not included → precision = 0
    assert precision_at_k(recs, gt, k=4) == pytest.approx(0.0)
    # k=5: item 10 included → precision = 1/5
    assert precision_at_k(recs, gt, k=5) == pytest.approx(1 / 5)

def test_precision_no_ground_truth():
    recs = {1: [10, 20], 2: [30, 40]}
    gt = {2: {30}}  # user 1 has no ground truth
    # Only user 2 should be scored: 1 hit out of k=2 recs → 0.5
    assert precision_at_k(recs, gt, k=2) == pytest.approx(0.5)


# ── ndcg@K ────────────────────────────────────────────────────────────────────

def test_ndcg_perfect(perfect_recs, perfect_gt):
    assert ndcg_at_k(perfect_recs, perfect_gt, k=5) == pytest.approx(1.0)

def test_ndcg_zero(zero_recs, zero_gt):
    assert ndcg_at_k(zero_recs, zero_gt, k=5) == pytest.approx(0.0)

def test_ndcg_ordering_matters():
    """NDCG should be higher when relevant items appear earlier."""
    gt = {1: {10, 20}}
    recs_good = {1: [10, 20, 99, 99, 99]}  # relevant items first
    recs_bad  = {1: [99, 99, 99, 10, 20]}  # relevant items last

    ndcg_good = ndcg_at_k(recs_good, gt, k=5)
    ndcg_bad  = ndcg_at_k(recs_bad, gt, k=5)
    assert ndcg_good > ndcg_bad, "NDCG should reward higher-ranked relevant items"

def test_ndcg_single_relevant():
    recs = {1: [10, 99, 99, 99, 99]}
    gt = {1: {10}}
    # DCG = 1/log2(2) = 1.0, IDCG = 1.0 → NDCG = 1.0
    assert ndcg_at_k(recs, gt, k=5) == pytest.approx(1.0)

def test_ndcg_between_zero_and_one():
    recs = {1: [99, 10, 99, 20, 99]}
    gt = {1: {10, 20, 30, 40}}
    score = ndcg_at_k(recs, gt, k=5)
    assert 0.0 <= score <= 1.0


# ── recall@K ──────────────────────────────────────────────────────────────────

def test_recall_perfect(perfect_recs, perfect_gt):
    assert recall_at_k(perfect_recs, perfect_gt, k=5) == pytest.approx(1.0)

def test_recall_zero(zero_recs, zero_gt):
    assert recall_at_k(zero_recs, zero_gt, k=5) == pytest.approx(0.0)

def test_recall_partial():
    recs = {1: [10, 99, 99, 99, 99]}
    gt = {1: {10, 20, 30}}  # 3 relevant, only 1 found
    assert recall_at_k(recs, gt, k=5) == pytest.approx(1 / 3)

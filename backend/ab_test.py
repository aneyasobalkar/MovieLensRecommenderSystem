"""
ab_test.py
Variant assignment and CTR metric aggregation.

Variant A  →  Item-Item CF  (control)
Variant B  →  SVD           (treatment)

Assignment is deterministic: even app user IDs get A, odd get B.
This ensures users always see the same model regardless of session.
"""
from sqlalchemy import func
from sqlalchemy.orm import Session

from .db import ClickEvent, Impression


def assign_variant(user_id: int) -> str:
    return "A" if user_id % 2 == 0 else "B"


VARIANT_LABELS = {
    "A": "Item-Item CF (control)",
    "B": "SVD Matrix Factorization (treatment)",
}


def get_ab_metrics(db: Session) -> dict:
    imp_rows = (
        db.query(Impression.variant, func.count(Impression.id).label("n"))
        .group_by(Impression.variant)
        .all()
    )
    click_rows = (
        db.query(ClickEvent.variant, func.count(ClickEvent.id).label("n"))
        .group_by(ClickEvent.variant)
        .all()
    )

    imp_map = {v: n for v, n in imp_rows}
    click_map = {v: n for v, n in click_rows}

    return {
        variant: {
            "model": VARIANT_LABELS[variant],
            "impressions": imp_map.get(variant, 0),
            "clicks": click_map.get(variant, 0),
            "ctr": round(click_map.get(variant, 0) / imp_map[variant], 4)
            if imp_map.get(variant, 0) > 0
            else 0.0,
        }
        for variant in ("A", "B")
    }

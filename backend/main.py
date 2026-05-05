"""
main.py — FastAPI application

Run with:
    uvicorn backend.main:app --reload

Routes
------
POST /auth/register          register + receive JWT
POST /auth/login             login + receive JWT
GET  /auth/me                current user info

GET  /recommendations        personalised recs (auth required)
GET  /movies                 browse movie catalogue
POST /ratings                rate a movie (invalidates rec cache)
GET  /ratings/me             list user's own ratings

POST /events/click           record a recommendation click (A/B tracking)
GET  /ab_test/metrics        CTR per variant
GET  /cache/stats            cache size (debug)
"""
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .ab_test import assign_variant, get_ab_metrics
from .auth import create_access_token, get_current_user, hash_password, verify_password
from .cache import rec_cache
from .db import Base, ClickEvent, Impression, User, UserRating, engine, get_db
from .recommender import assign_ml_user_id, get_recommendations, get_store, train_and_load

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    train_and_load(DATA_DIR)
    yield


app = FastAPI(title="Movie Recommender API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(min_length=2, max_length=40)
    password: str = Field(min_length=4)


class RatingRequest(BaseModel):
    movie_id: int
    rating: float = Field(ge=1.0, le=5.0)


class ClickRequest(BaseModel):
    movie_id: int


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.post("/auth/register", summary="Register a new user")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == req.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")

    user = User(
        username=req.username,
        hashed_password=hash_password(req.password),
        ab_variant="A",   # placeholder — updated after commit gives us the ID
        ml_user_id=1,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Now that we have the real ID, set variant and ML mapping
    user.ab_variant = assign_variant(user.id)
    user.ml_user_id = assign_ml_user_id(user.id)
    db.commit()

    token = create_access_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "variant": user.ab_variant,
        "username": user.username,
    }


@app.post("/auth/login", summary="Login and receive JWT")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    token = create_access_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "variant": user.ab_variant,
        "username": user.username,
    }


@app.get("/auth/me", summary="Current user info")
def me(user: User = Depends(get_current_user)):
    return {
        "id": user.id,
        "username": user.username,
        "variant": user.ab_variant,
        "ml_user_id": user.ml_user_id,
    }


# ── Recommendations ───────────────────────────────────────────────────────────

@app.get("/recommendations", summary="Get personalised recommendations")
def recommendations(
    n: int = 10,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cache_key = f"recs:{user.id}:{user.ab_variant}:{n}"
    cached = rec_cache.get(cache_key)

    if cached is not None:
        _record_impressions(db, user.id, user.ab_variant, [r["movie_id"] for r in cached])
        return {"recommendations": cached, "variant": user.ab_variant, "from_cache": True}

    recs = get_recommendations(user.ml_user_id, user.ab_variant, n=n)
    rec_cache.set(cache_key, recs)
    _record_impressions(db, user.id, user.ab_variant, [r["movie_id"] for r in recs])
    return {"recommendations": recs, "variant": user.ab_variant, "from_cache": False}


def _record_impressions(db: Session, user_id: int, variant: str, movie_ids: list[int]) -> None:
    for mid in movie_ids:
        db.add(Impression(user_id=user_id, movie_id=mid, variant=variant))
    db.commit()


# ── Movies ────────────────────────────────────────────────────────────────────

@app.get("/movies", summary="Browse the movie catalogue")
def list_movies(limit: int = 100, offset: int = 0):
    store = get_store()
    if store.movies is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    chunk = store.movies.iloc[offset : offset + limit]
    return [
        {"movie_id": int(r.movie_id), "title": r.title}
        for _, r in chunk.iterrows()
    ]


@app.get("/movies/{movie_id}", summary="Movie details")
def get_movie(movie_id: int):
    store = get_store()
    title = store.movie_lookup.get(movie_id)
    if title is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    return {"movie_id": movie_id, "title": title}


# ── Ratings ───────────────────────────────────────────────────────────────────

@app.post("/ratings", summary="Rate a movie")
def rate_movie(
    req: RatingRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    existing = (
        db.query(UserRating)
        .filter(UserRating.user_id == user.id, UserRating.movie_id == req.movie_id)
        .first()
    )
    if existing:
        existing.rating = req.rating
    else:
        db.add(UserRating(user_id=user.id, movie_id=req.movie_id, rating=req.rating))
    db.commit()

    # Bust the cache so the next recommendation call is fresh
    rec_cache.delete_prefix(f"recs:{user.id}:")
    return {"ok": True, "movie_id": req.movie_id, "rating": req.rating}


@app.get("/ratings/me", summary="My ratings")
def my_ratings(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(UserRating).filter(UserRating.user_id == user.id).all()
    return [{"movie_id": r.movie_id, "rating": r.rating} for r in rows]


# ── Events ────────────────────────────────────────────────────────────────────

@app.post("/events/click", summary="Record a recommendation click")
def record_click(
    req: ClickRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    db.add(ClickEvent(user_id=user.id, movie_id=req.movie_id, variant=user.ab_variant))
    db.commit()
    return {"ok": True}


# ── A/B Test Metrics ──────────────────────────────────────────────────────────

@app.get("/ab_test/metrics", summary="CTR by A/B variant")
def ab_metrics(db: Session = Depends(get_db)):
    return get_ab_metrics(db)


# ── Debug ─────────────────────────────────────────────────────────────────────

@app.get("/cache/stats", summary="Cache stats (debug)")
def cache_stats():
    return {"entries": rec_cache.size}

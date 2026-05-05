import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./movie_recommender.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    ab_variant = Column(String, nullable=False, default="A")
    # Maps this app user to a MovieLens training user for recommendations
    ml_user_id = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class UserRating(Base):
    __tablename__ = "user_ratings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    movie_id = Column(Integer, nullable=False)
    rating = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ClickEvent(Base):
    __tablename__ = "click_events"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    movie_id = Column(Integer, nullable=False)
    variant = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Impression(Base):
    __tablename__ = "impressions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    movie_id = Column(Integer, nullable=False)
    variant = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

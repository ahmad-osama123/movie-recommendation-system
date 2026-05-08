"""
Data Ingestion and Preprocessing Module
Loads and cleans MovieLens dataset (movies.csv + ratings.csv)
"""

import pandas as pd
import numpy as np


def load_data(movies_path="data/movies.csv", ratings_path="data/ratings.csv"):
    """Load movies and ratings CSV files."""
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings


def preprocess_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess movie data.
    - Fill missing genres
    - Extract year from title
    - Normalize genres list
    """
    movies = movies.copy()

    # Handle missing values
    movies["genres"] = movies["genres"].fillna("Unknown")
    movies["title"] = movies["title"].fillna("Unknown Title")

    # Extract year from title (e.g., "Toy Story (1995)" → 1995)
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$").astype(float)

    # Replace '(no genres listed)' with 'Unknown'
    movies["genres"] = movies["genres"].replace("(no genres listed)", "Unknown")

    # Create genre list column
    movies["genre_list"] = movies["genres"].apply(lambda x: x.split("|"))

    return movies


def preprocess_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess ratings data.
    - Drop duplicates
    - Convert timestamp to datetime
    - Remove extreme outlier ratings if any
    """
    ratings = ratings.copy()

    # Drop duplicates (keep latest rating per user-movie pair)
    ratings = ratings.sort_values("timestamp").drop_duplicates(
        subset=["userId", "movieId"], keep="last"
    )

    # Convert timestamp
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")

    # Sanity check: ratings should be in [0.5, 5.0]
    ratings = ratings[ratings["rating"].between(0.5, 5.0)]

    ratings = ratings.reset_index(drop=True)
    return ratings


def get_stats(movies: pd.DataFrame, ratings: pd.DataFrame) -> dict:
    """Return basic dataset statistics."""
    stats = {
        "num_movies": len(movies),
        "num_users": ratings["userId"].nunique(),
        "num_ratings": len(ratings),
        "avg_rating": round(ratings["rating"].mean(), 3),
        "rating_std": round(ratings["rating"].std(), 3),
        "sparsity": round(
            1 - len(ratings) / (ratings["userId"].nunique() * ratings["movieId"].nunique()),
            4,
        ),
    }
    return stats


if __name__ == "__main__":
    movies, ratings = load_data()
    movies = preprocess_movies(movies)
    ratings = preprocess_ratings(ratings)
    stats = get_stats(movies, ratings)

    print("=== Dataset Statistics ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("\nMovies sample:")
    print(movies[["movieId", "title", "genres", "year"]].head())
    print("\nRatings sample:")
    print(ratings[["userId", "movieId", "rating", "datetime"]].head())

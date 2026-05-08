"""
Hybrid Movie Recommendation System — Streamlit App
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import sys
import requests

# ── Make sure local modules are importable ─────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_processing import load_data, preprocess_movies, preprocess_ratings, get_stats
from models.content_based import ContentBasedFilter
from models.collaborative_filter import CollaborativeFilter
from models.hybrid_recommender import HybridRecommender
from utils.evaluation import full_evaluation_report

# ── TMDb Poster ───────────────────────────────────────────────────────────
TMDB_API_KEY = "17bb05940e431890b39c5ec1741771eb"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMG_URL  = "https://image.tmdb.org/t/p/w300"

import re
import base64
from io import BytesIO

@st.cache_data(show_spinner=False)
def _fetch_poster_path(title: str, year: str = None) -> str:
    """محاولات متعددة للحصول على poster_path من TMDb."""
    attempts = []

    # محاولة 1: العنوان + السنة
    if year:
        attempts.append({"api_key": TMDB_API_KEY, "query": title, "year": year, "language": "en-US"})

    # محاولة 2: العنوان بدون سنة
    attempts.append({"api_key": TMDB_API_KEY, "query": title, "language": "en-US"})

    # محاولة 3: العنوان بدون كلمة "The" أو "A" في الأول
    short = re.sub(r"^(The|A|An)\s+", "", title, flags=re.IGNORECASE).strip()
    if short != title:
        attempts.append({"api_key": TMDB_API_KEY, "query": short, "language": "en-US"})

    # محاولة 4: أول كلمتين من العنوان (للأفلام اللي ليها subtitle طويل)
    words = title.split()
    if len(words) > 2:
        attempts.append({"api_key": TMDB_API_KEY, "query": " ".join(words[:2]), "language": "en-US"})

    for params in attempts:
        try:
            r = requests.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=8)
            results = r.json().get("results", [])
            for res in results:
                if res.get("poster_path"):
                    return res["poster_path"]
        except:
            continue
    return ""

@st.cache_data(show_spinner=False)
def get_poster_b64(movie_title: str) -> str:
    """
    Returns poster as base64 string for display via st.markdown HTML.
    Works on all Streamlit versions.
    """
    try:
        year_match = re.search(r"[(](\d{4})[)]", movie_title)
        year = year_match.group(1) if year_match else None
        title = movie_title.split("(")[0].strip()

        poster_path = _fetch_poster_path(title, year)

        if poster_path:
            img_r = requests.get(TMDB_IMG_URL + poster_path, timeout=8)
            if img_r.status_code == 200:
                b64 = base64.b64encode(img_r.content).decode()
                return f"data:image/jpeg;base64,{b64}"

    except:
        pass
    return ""

def show_poster(title: str, caption: str, sub: str = ""):
    """Display a single movie poster with caption using HTML."""
    b64 = get_poster_b64(title)
    if b64:
        img_tag = f'<img src="{b64}" style="width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.3);">'
    else:
        # placeholder بيعرض أول حرفين من اسم الفيلم
        initials = "".join(w[0].upper() for w in title.split()[:2] if w)
        img_tag = f'''<div style="width:100%;aspect-ratio:2/3;background:linear-gradient(135deg,#1a1a2e,#2e75b6);
                    border-radius:8px;display:flex;flex-direction:column;align-items:center;
                    justify-content:center;color:#fff;font-size:28px;font-weight:bold;
                    box-shadow:0 2px 8px rgba(0,0,0,0.3);">
                    {initials}
                    <span style="font-size:10px;margin-top:8px;color:#aaa;text-align:center;padding:0 4px">{title[:30]}</span>
                    </div>'''
    html = f"""
    <div style="text-align:center;padding:4px">
        {img_tag}
        <p style="font-size:11px;margin:4px 0 0;font-weight:600;line-height:1.3">{caption}</p>
        <p style="font-size:11px;color:#888;margin:2px 0 0">{sub}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────
MODEL_CACHE = "models_cache.pkl"
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


# ── Model Loading / Caching ────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Training models (first run only)...")
def load_models():
    movies, ratings = load_data()
    movies = preprocess_movies(movies)
    ratings = preprocess_ratings(ratings)

    cb = ContentBasedFilter()
    cb.fit(movies)

    cf = CollaborativeFilter()
    cf.prepare_data(ratings, test_size=0.2)
    cf.fit()

    hybrid = HybridRecommender(cf_model=cf, cb_model=cb, movies=movies, alpha=0.7)

    stats = get_stats(movies, ratings)
    return movies, ratings, cb, cf, hybrid, stats


movies, ratings, cb, cf, hybrid, dataset_stats = load_models()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Film_strip.svg/240px-Film_strip.svg.png", width=80)
    st.title("🎬 Movie Recommender")
    st.markdown("---")

    mode = st.radio(
        "Choose Mode",
        ["👤 Existing User", "🆕 New User (Cold Start)", "🔍 Find Similar Movies", "📊 Evaluation Report"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Dataset Stats**")
    st.metric("Movies", f"{dataset_stats['num_movies']:,}")
    st.metric("Users", f"{dataset_stats['num_users']:,}")
    st.metric("Ratings", f"{dataset_stats['num_ratings']:,}")
    st.metric("Avg Rating", dataset_stats["avg_rating"])
    st.metric("Sparsity", f"{dataset_stats['sparsity']*100:.1f}%")

    st.markdown("---")
    alpha = st.slider("⚖️ Hybrid Weight (α)", 0.0, 1.0, 0.7, 0.05,
                      help="α=1 → pure CF | α=0 → pure CB")
    hybrid.alpha = alpha
    top_n = st.slider("🎯 Top N Recommendations", 5, 20, 10)

# ── Main Content ───────────────────────────────────────────────────────────
st.title("🎬 Hybrid Movie Recommendation System")
st.markdown("*Combining Collaborative Filtering (SVD) + Content-Based Filtering (TF-IDF)*")

# ─────────────────────────────────────────────────────────────────────────
if mode == "👤 Existing User":
    st.header("Recommendations for an Existing User")

    all_users = sorted(ratings["userId"].unique())
    col1, col2 = st.columns([1, 2])

    with col1:
        user_id = st.selectbox("Select User ID", all_users)

        user_ratings = ratings[ratings["userId"] == user_id]
        st.metric("Movies Rated", len(user_ratings))
        st.metric("Avg Rating Given", round(user_ratings["rating"].mean(), 2))

    with col2:
        st.subheader("User's Top Rated Movies")
        top_rated = (
            user_ratings.merge(movies[["movieId", "title", "genres"]], on="movieId")
            .sort_values("rating", ascending=False)
            .head(5)[["title", "genres", "rating"]]
        )
        st.dataframe(top_rated, use_container_width=True, hide_index=True)

    if st.button("🚀 Get Recommendations", type="primary"):
        with st.spinner("Generating hybrid recommendations..."):
            recs = hybrid.recommend(user_id=user_id, ratings_df=ratings, top_n=top_n)

        st.subheader(f"🎯 Top {top_n} Recommendations for User {user_id}")
        st.caption(f"Using α={alpha:.2f} (CF weight) | {1-alpha:.2f} (CB weight)")

        # Color-code rows by hybrid score
        styled = recs[["rank", "title", "genres", "cf_score", "cb_score", "hybrid_score"]].copy()
        styled.columns = ["#", "Title", "Genres", "CF Score", "CB Score", "Hybrid Score"]
        styled["Hybrid Score"] = styled["Hybrid Score"].apply(lambda x: f"{x:.4f}")
        styled["CF Score"] = styled["CF Score"].apply(lambda x: f"{x:.3f}")
        styled["CB Score"] = styled["CB Score"].apply(lambda x: f"{x:.4f}")

        cols = st.columns(5)
        for i, row in recs.iterrows():
            with cols[i % 5]:
                show_poster(row["title"], row["title"], f"⭐ {row['hybrid_score']:.3f}")

        # Genre distribution of recommendations
        genres_flat = []
        for g in recs["genres"]:
            genres_flat.extend(g.split("|"))
        genre_counts = pd.Series(genres_flat).value_counts().reset_index()
        genre_counts.columns = ["Genre", "Count"]

        st.subheader("📊 Genre Distribution of Recommendations")
        st.bar_chart(genre_counts.set_index("Genre"))

# ─────────────────────────────────────────────────────────────────────────
elif mode == "🆕 New User (Cold Start)":
    st.header("Cold Start — Recommend for a New User")
    st.info("No ratings history? No problem! Select your favourite genres and we'll suggest movies.")

    selected_genres = st.multiselect(
        "🎭 Select Your Favourite Genres",
        ALL_GENRES,
        default=["Action", "Adventure"],
    )

    if st.button("🚀 Get Recommendations", type="primary"):
        if not selected_genres:
            st.warning("Please select at least one genre.")
        else:
            with st.spinner("Finding the best movies for you..."):
                cold_recs = hybrid.recommend_for_new_user(
                    liked_genres=selected_genres, top_n=top_n
                )

            if cold_recs.empty:
                st.error("No recommendations found. Try selecting different genres.")
            else:
                st.subheader(f"🎬 Top {top_n} Movies for a {', '.join(selected_genres)} fan")
                cols = st.columns(5)
                for i, row in cold_recs.iterrows():
                    with cols[i % 5]:
                        show_poster(row["title"], row["title"], row["genres"])

# ─────────────────────────────────────────────────────────────────────────
elif mode == "🔍 Find Similar Movies":
    st.header("Find Movies Similar to a Given Movie")

    movie_titles = movies["title"].sort_values().tolist()
    selected_title = st.selectbox("🎞️ Select a Movie", movie_titles)

    selected_movie = movies[movies["title"] == selected_title].iloc[0]
    movie_id = selected_movie["movieId"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Movie ID", movie_id)
    with col2:
        st.metric("Genres", selected_movie["genres"])

    if st.button("🔍 Find Similar", type="primary"):
        with st.spinner("Computing similarity..."):
            similar = cb.get_similar_movies(movie_id=movie_id, top_n=top_n)

        st.subheader(f"🎬 Movies Similar to *{selected_title}*")
        c1, _ = st.columns([1, 3])
        with c1:
            show_poster(selected_title, selected_title, selected_movie["genres"])
        st.markdown("**Similar movies:**")
        cols = st.columns(5)
        for i, row in similar.iterrows():
            with cols[i % 5]:
                show_poster(row["title"], row["title"], f"🔗 {row['similarity_score']:.4f}")

# ─────────────────────────────────────────────────────────────────────────
elif mode == "📊 Evaluation Report":
    st.header("📊 Model Evaluation Report")
    st.info(
        "This runs Precision@K, Recall@K, F1@K over 50 sampled users. "
        "RMSE & MAE are computed on the 20% held-out test set."
    )

    k = st.slider("K (for Precision/Recall/F1)", 5, 20, 10)

    if st.button("▶️ Run Evaluation", type="primary"):
        with st.spinner("Evaluating all models... this may take ~1–2 minutes..."):
            report = full_evaluation_report(cf, cb, hybrid, ratings, movies, k=k)

        st.subheader("📋 Results")
        st.dataframe(report, use_container_width=True, hide_index=True)

        # Visual comparison
        st.subheader(f"📈 Precision / Recall / F1 @{k}")
        plot_data = report[report[f"F1@{k}"] != "N/A"].copy()
        plot_data = plot_data.set_index("Model")[[f"Precision@{k}", f"Recall@{k}", f"F1@{k}"]].astype(float)
        st.bar_chart(plot_data)

        st.subheader("💡 Insights")
        best_f1_model = report.loc[report[f"F1@{k}"].replace("N/A", np.nan).dropna().astype(float).idxmax(), "Model"]
        st.success(f"🏆 Best model by F1@{k}: **{best_f1_model}**")
        st.markdown("""
        - **Collaborative Filtering** is strong when user history is rich.
        - **Content-Based** excels for cold-start and niche genre fans.
        - **Hybrid** balances both strengths for overall best performance.
        """)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Hybrid Recommendation System | MovieLens 100K | SVD + TF-IDF Cosine Similarity")

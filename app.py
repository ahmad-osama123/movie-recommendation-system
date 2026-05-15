"""
git add .
git commit -m "Final fix for import paths and evaluation logic"
git push origin main
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

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 CinemaAI — Movie Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium Cinematic CSS Theme ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700;800&display=swap');

:root {
    --primary-dark: #0a0e27;
    --secondary-dark: #121829;
    --accent-gold: #d4af37;
    --accent-emerald: #2ecc71;
    --accent-coral: #ff6b6b;
    --text-primary: #0f52ba;
    --text-secondary: #e5e4e2;
    --border-color: #1e2a45;
}

* { font-family: 'Poppins', sans-serif; }

/* Main Container */
.main {
    background: linear-gradient(135deg, #0a0e27 0%, #121829 100%);
    color: var(--text-primary);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #121829 0%, #0f1425 100%);
    border-right: 2px solid var(--border-color);
}

/* Headers with gradient */
h1, h2, h3 {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    background: linear-gradient(135deg, #0f52ba 0%, #e5e4e2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
}

h1 { font-size: 3.5rem; margin-bottom: 8px; }
h2 { font-size: 2.2rem; margin-top: 32px; margin-bottom: 20px; }
h3 { font-size: 1.6rem; margin-top: 20px; }

/* Radio buttons - improved */
[data-testid="stRadio"] {
    background: transparent;
    gap: 16px;
}

[data-testid="stRadio"] label {
    background: linear-gradient(135deg, rgba(18, 24, 41, 0.6), rgba(30, 42, 69, 0.6));
    border: 2px solid #1e2a45;
    border-radius: 10px;
    padding: 12px 20px;
    cursor: pointer;
    transition: all 0.3s ease;
    color: var(--text-primary);
    font-weight: 500;
}

[data-testid="stRadio"] label:hover {
    border-color: #0f52ba;
    background: linear-gradient(135deg, rgba(30, 42, 69, 0.8), rgba(38, 50, 80, 0.8));
    transform: translateY(-2px);
}

/* Primary buttons */
.stButton > button {
    background: linear-gradient(135deg, #1a3a6e 0%, #2e75b6 60%, #c0c0c0 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    transition: all 0.3s ease !important;
    font-size: 1rem !important;
    text-transform: none !important;
    box-shadow: 0 4px 20px rgba(46, 117, 182, 0.4) !important;
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(46, 117, 182, 0.6) !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(18, 24, 41, 0.7), rgba(30, 42, 69, 0.7));
    border: 1px solid #1e2a45;
    border-radius: 12px;
    padding: 20px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

[data-testid="metric-container"]:hover {
    border-color: #0f52ba;
    background: linear-gradient(135deg, rgba(30, 42, 69, 0.8), rgba(38, 50, 80, 0.8));
}

[data-testid="metric-container"] label {
    color: #b8c5d6;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

[data-testid="metric-container"] > div:last-child {
    color: #0f52ba;
    font-size: 1.8rem;
    font-weight: 700;
}

/* Alerts */
.stAlert {
    border-radius: 10px;
    border-left: 4px solid #0f52ba;
    background: rgba(30, 42, 69, 0.5) !important;
}

/* Info boxes */
.stInfo {
    background: rgba(46, 204, 113, 0.1) !important;
    border-left-color: #2ecc71 !important;
}

/* Sliders */
.stSlider > div > div > div {
    background: linear-gradient(90deg, #2e75b6, #c0c0c0) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] {
    background-color: transparent !important;
}

[data-testid="stSelectbox"] > div > div {
    background: linear-gradient(135deg, rgba(18, 24, 41, 0.8), rgba(30, 42, 69, 0.8));
    border: 1px solid #1e2a45 !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* Multiselect */
[data-testid="stMultiSelect"] {
    background-color: transparent !important;
}

[data-testid="stMultiSelect"] > div > div {
    background: linear-gradient(135deg, rgba(18, 24, 41, 0.8), rgba(30, 42, 69, 0.8));
    border: 1px solid #1e2a45 !important;
    border-radius: 10px !important;
}

/* Data frames */
.stDataFrame {
    font-size: 0.95rem;
}

[data-testid="stDataFrameContainer"] {
    background: transparent !important;
}

/* Dividers */
hr {
    border-color: #1e2a45 !important;
    margin: 24px 0;
}

/* Expander */
.streamlit-expander {
    background: linear-gradient(135deg, rgba(18, 24, 41, 0.5), rgba(30, 42, 69, 0.5));
    border: 1px solid #1e2a45;
    border-radius: 10px;
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.main > :nth-child(1) {
    animation: fadeInUp 0.6s ease-out;
}

/* Movie poster styling */
.movie-poster-wrapper {
    transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.movie-poster-wrapper:hover {
    transform: translateY(-8px) scale(1.02);
    filter: drop-shadow(0 20px 30px rgba(15, 82, 186, 0.3));
}

</style>
""", unsafe_allow_html=True)

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
    """Display a single movie poster with enhanced styling."""
    b64 = get_poster_b64(title)
    if b64:
        img_tag = f'<img src="{b64}" style="width:100%;border-radius:12px;box-shadow:0 8px 24px rgba(15, 82, 186, 0.2);transition:all 0.4s ease;border: 2px solid rgba(15, 82, 186, 0.3);">'
    else:
        # Elegant placeholder
        initials = "".join(w[0].upper() for w in title.split()[:2] if w)
        img_tag = f'''<div style="width:100%;aspect-ratio:2/3;background:linear-gradient(135deg,#1e2a45,#2e75b6);
                    border-radius:12px;display:flex;flex-direction:column;align-items:center;
                    justify-content:center;color:#f5f7fa;font-size:32px;font-weight:700;
                    box-shadow:0 8px 24px rgba(15, 82, 186, 0.2);border: 2px solid rgba(15, 82, 186, 0.3);
                    font-family:'Playfair Display', serif;">
                    {initials}
                    <span style="font-size:11px;margin-top:12px;color:#e5e4e2;text-align:center;padding:0 4px;font-family:'Poppins',sans-serif;">{title[:28]}</span>
                    </div>'''
    html = f"""
    <div class="movie-poster-wrapper" style="text-align:center;padding:8px">
        {img_tag}
        <p style="font-size:12px;margin:12px 0 4px;font-weight:700;line-height:1.3;color:#0f52ba;">{caption}</p>
        <p style="font-size:11px;color:#e5e4e2;margin:2px 0 0">{sub}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

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
    st.markdown('''
    <div style="text-align:center;padding:18px 0 8px">
        <div style="font-size:32px;font-weight:900;letter-spacing:3px;
                    background:linear-gradient(135deg,#2e75b6,#c0c0c0);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;font-family:Playfair Display,serif;">
            🎬 Cinema<span style="-webkit-text-fill-color:#c0c0c0">AI</span>
        </div>
        <div style="font-size:10px;color:#555;letter-spacing:3px;text-transform:uppercase;margin-top:4px;">
            Movie Intelligence
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("""
    
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    mode = st.radio(
        "Choose Mode",
        ["👤 Existing User", "🆕 New User (Cold Start)", "🔍 Find Similar Movies", "📊 Evaluation Report"],
        index=0,
    )

    st.markdown("---")
    
    st.markdown("""
    <p style="font-size: 0.9rem; font-weight: 700; color: #c0c0c0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px;">
    📊 Dataset Statistics
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Movies", f"{dataset_stats['num_movies']:,}")
        st.metric("Users", f"{dataset_stats['num_users']:,}")
    with col2:
        st.metric("Ratings", f"{dataset_stats['num_ratings']:,}")
        st.metric("Avg Rating", f"{dataset_stats['avg_rating']:.1f}")
    
    st.metric("Sparsity", f"{dataset_stats['sparsity']*100:.1f}%")

    st.markdown("---")
    
    st.markdown("""
    <p style="font-size: 0.85rem; font-weight: 700; color: #b8c5d6; text-transform: uppercase; letter-spacing: 0.5px;">
    ⚙️ Tuning Parameters
    </p>
    """, unsafe_allow_html=True)
    
    alpha = st.slider("⚖️ Hybrid Weight (α)", 0.0, 1.0, 0.7, 0.05,
                      help="α=1 → pure CF | α=0 → pure CB")
    hybrid.alpha = alpha
    
    top_n = st.slider("🎯 Top N Recommendations", 5, 20, 10)

# ── Main Content ───────────────────────────────────────────────────────────
st.markdown('''<h1 style="font-size:3rem;font-weight:900;letter-spacing:2px;background:linear-gradient(135deg,#2e75b6,#c0c0c0);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:4px;">🎬 CinemaAI</h1><p style="font-size:1rem;color:#888;letter-spacing:2px;text-transform:uppercase;margin-bottom:24px;">Movie Intelligence System</p>''', unsafe_allow_html=True)
st.markdown("""
<p style="font-size: 1.1rem; color: #c0c0c0; margin-bottom: 32px;">
Combining <span style="color: #e8e8e8; font-weight: 700;">Collaborative Filtering (SVD)</span> + 
<span style="color: #a8a8b0; font-weight: 700;">Content-Based Filtering (TF-IDF)</span> for smarter recommendations
</p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
if mode == "👤 Existing User":
    st.header("👤 Recommendations for an Existing User")

    all_users = sorted(ratings["userId"].unique())
    col1, col2 = st.columns([1, 2])

    with col1:
        user_id = st.selectbox("Select User ID", all_users)

        user_ratings = ratings[ratings["userId"] == user_id]
        st.metric("Movies Rated", len(user_ratings))
        st.metric("Avg Rating Given", round(user_ratings["rating"].mean(), 2))

    with col2:
        st.markdown("##### 🏆 User's Top Rated Movies")
        top_rated = (
            user_ratings.merge(movies[["movieId", "title", "genres"]], on="movieId")
            .sort_values("rating", ascending=False)
            .head(5)[["title", "genres", "rating"]]
        )
        st.dataframe(top_rated, use_container_width=True, hide_index=True)

    if st.button("🚀 Get Recommendations", type="primary", use_container_width=False):
        with st.spinner("✨ Generating hybrid recommendations..."):
            recs = hybrid.recommend(user_id=user_id, ratings_df=ratings, top_n=top_n)

        st.markdown(f"""
        <h2 style="margin-top: 32px;">🎯 Top {top_n} Recommendations for User {user_id}</h2>
        <p style="color: #ffb84d; margin-bottom: 24px;">Using <span style="color: #e0e0e0;">α={alpha:.2f}</span> (CF weight) | 
        <span style="color: #a0a0b0;">{1-alpha:.2f}</span> (CB weight)</p>
        """, unsafe_allow_html=True)

        # Movie grid
        st.markdown("##### 🎬 Movie Grid")
        cols = st.columns(5)
        for i, row in recs.iterrows():
            with cols[i % 5]:
                show_poster(row["title"], row["title"], f"⭐ {row['hybrid_score']:.3f}")

        # Scores table
        st.markdown("##### 📊 Detailed Scores")
        styled = recs[["rank", "title", "cf_score", "cb_score", "hybrid_score"]].copy()
        styled.columns = ["#", "Title", "CF Score", "CB Score", "Hybrid Score"]
        styled["Hybrid Score"] = styled["Hybrid Score"].apply(lambda x: f"{x:.4f}")
        styled["CF Score"] = styled["CF Score"].apply(lambda x: f"{x:.3f}")
        styled["CB Score"] = styled["CB Score"].apply(lambda x: f"{x:.4f}")
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Genre distribution
        genres_flat = []
        for g in recs["genres"]:
            genres_flat.extend(g.split("|"))
        genre_counts = pd.Series(genres_flat).value_counts().reset_index()
        genre_counts.columns = ["Genre", "Count"]

        st.markdown("##### 📈 Genre Distribution")
        st.bar_chart(genre_counts.set_index("Genre"), color="#2e75b6")

# ─────────────────────────────────────────────────────────────────────────
elif mode == "🆕 New User (Cold Start)":
    st.header("🆕 Cold Start — Discover Movies for a New User")
    st.info("💡 No ratings history? No problem! Select your favorite genres and we'll suggest amazing movies.")

    selected_genres = st.multiselect(
        "🎭 Select Your Favourite Genres",
        ALL_GENRES,
        default=["Action", "Adventure"],
    )

    if st.button("🚀 Get Recommendations", type="primary", use_container_width=False):
        if not selected_genres:
            st.warning("⚠️ Please select at least one genre to get recommendations.")
        else:
            with st.spinner("🎬 Finding the best movies for you..."):
                cold_recs = hybrid.recommend_for_new_user(
                    liked_genres=selected_genres, top_n=top_n
                )

            if cold_recs.empty:
                st.error("❌ No recommendations found. Try selecting different genres.")
            else:
                st.markdown(f"""
                <h2 style="margin-top: 32px;">🎬 Top {top_n} Movies for a {', '.join(selected_genres)} fan</h2>
                """, unsafe_allow_html=True)
                
                cols = st.columns(5)
                for i, row in cold_recs.iterrows():
                    with cols[i % 5]:
                        show_poster(row["title"], row["title"], row["genres"])

# ─────────────────────────────────────────────────────────────────────────
elif mode == "🔍 Find Similar Movies":
    st.header("🔍 Find Movies Similar to a Given Movie")

    movie_titles = movies["title"].sort_values().tolist()
    selected_title = st.selectbox("🎞️ Select a Movie", movie_titles)

    selected_movie = movies[movies["title"] == selected_title].iloc[0]
    movie_id = selected_movie["movieId"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Movie ID", movie_id)
    with col2:
        st.metric("Genres", selected_movie["genres"])

    if st.button("🔍 Find Similar", type="primary", use_container_width=False):
        with st.spinner("📊 Computing similarity..."):
            similar = cb.get_similar_movies(movie_id=movie_id, top_n=top_n)

        st.markdown(f"""
        <h2 style="margin-top: 32px;">🎬 Movies Similar to <span style="color: #ff9500;">'{selected_title}'</span></h2>
        """, unsafe_allow_html=True)
        
        col_poster, col_info = st.columns([1, 3])
        with col_poster:
            show_poster(selected_title, selected_title, selected_movie["genres"])
        
        with col_info:
            st.markdown(f"""
            <p style="font-size: 1.05rem; color: #ffb84d; margin: 16px 0;">
            Based on <span style="color: #e0e0e0; font-weight: 700;">content similarity</span>, 
            here are the top {top_n} movies you might enjoy:
            </p>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("##### 🎬 Similar Movies")
        cols = st.columns(5)
        for i, row in similar.iterrows():
            with cols[i % 5]:
                show_poster(row["title"], row["title"], f"📊 {row['similarity_score']:.3f}")

# ─────────────────────────────────────────────────────────────────────────
elif mode == "📊 Evaluation Report":
    st.header("📊 Model Evaluation Report")
    st.info(
        "📈 This runs Precision@K, Recall@K, F1@K over 50 sampled users. "
        "RMSE & MAE are computed on the 20% held-out test set."
    )

    k = st.slider("K (for Precision/Recall/F1)", 5, 20, 10)

    if st.button("▶️ Run Evaluation", type="primary", use_container_width=False):
        with st.spinner("⚙️ Evaluating all models... this may take ~1–2 minutes..."):
            report = full_evaluation_report(cf, cb, hybrid, ratings, movies, k=k)

        st.markdown("### 📋 Evaluation Results")
        st.dataframe(report, use_container_width=True, hide_index=True)

        # Visual comparison
        st.markdown(f"### 📈 Precision / Recall / F1 @{k}")
        plot_data = report[report[f"F1@{k}"] != "N/A"].copy()
        plot_data = plot_data.set_index("Model")[[f"Precision@{k}", f"Recall@{k}", f"F1@{k}"]].astype(float)
        st.bar_chart(plot_data, color=["#2e75b6", "#c0c0c0", "#7ab3d6"])

        st.markdown("### 💡 Model Insights")
        best_f1_model = report.loc[report[f"F1@{k}"].replace("N/A", np.nan).dropna().astype(float).idxmax(), "Model"]
        st.success(f"🏆 Best model by F1@{k}: **{best_f1_model}**")
        st.markdown(f"""
        #### Key Takeaways:
        - 🤝 **Collaborative Filtering** excels when user history is rich and detailed
        - 🎯 **Content-Based** shines for cold-start scenarios and niche genre enthusiasts  
        - ⚡ **Hybrid Approach** optimally balances both strengths for superior overall performance
        
        #### What This Means:
        - Use **CF** when you have strong user preference signals
        - Use **CB** when discovering new users' tastes
        - Use **Hybrid** for production systems requiring reliability and adaptability
        """)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #c0c0c0; font-size: 0.85rem; margin-top: 40px; padding-top: 24px; border-top: 1px solid #1e2a45;">
    <p>✨ <strong>Hybrid Recommendation System</strong> | MovieLens 100K | SVD + TF-IDF Cosine Similarity</p>
    <p style="margin-top: 8px; color: #a0a0b0; font-size: 0.8rem;">Built with Streamlit • Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

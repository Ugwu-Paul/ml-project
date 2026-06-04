import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎬 Movie Recommender")
st.caption("Rate a few films and get personalised recommendations powered by a Two-Tower neural network.")

# ── Model definition ───────────────────────────────────────────────────────────
class UserTower(nn.Module):
    def _init_(self, U, D):
        super()._init_()
        self.emb = nn.Embedding(U, D)
        self.mlp = nn.Sequential(nn.Linear(D, D), nn.ReLU())
    def forward(self, u):
        return self.mlp(self.emb(u))

class ItemTower(nn.Module):
    def _init_(self, I, D):
        super()._init_()
        self.emb = nn.Embedding(I, D)
        self.mlp = nn.Sequential(nn.Linear(D, D), nn.ReLU())
    def forward(self, i):
        return self.mlp(self.emb(i))

class TwoTowerRegressor(nn.Module):
    def _init_(self, U, I, D=64):
        super()._init_()
        self.user_tower  = UserTower(U, D)
        self.item_tower  = ItemTower(I, D)
        self.user_bias   = nn.Embedding(U, 1)
        self.item_bias   = nn.Embedding(I, 1)
        self.global_bias = nn.Parameter(torch.tensor([0.0]))
    def forward(self, u, i):
        u_vec = self.user_tower(u)
        i_vec = self.item_tower(i)
        dot   = (u_vec * i_vec).sum(dim=1)
        bias  = self.user_bias(u).squeeze() + self.item_bias(i).squeeze() + self.global_bias
        return dot + bias

# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artefacts():
    bundle          = joblib.load("movie_model.pkl")
    model           = bundle["model"]
    item_categories = bundle["item_categories"]
    movies          = bundle["movies"]
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, item_categories, movies, device

try:
    model, item_categories, movies, device = load_artefacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Could not load model: {e}")
    st.info(
        "Make sure movie_model.pkl is in the same directory and was saved with:\n"
        "python\njoblib.dump({'model': model, 'item_categories': item_categories,"
        " 'movies': movies}, 'movie_model.pkl')\n"
    )

# ── Optimise user vector WITH live loss animation ──────────────────────────────
def get_new_user_vector_animated(rated_items, model, device, D=64, steps=300):
    """
    Same optimisation as before, but yields (step, loss) every N steps
    so the caller can animate a live chart.
    """
    model.eval()
    u_vec = nn.Parameter(torch.randn(1, D, device=device))
    opt   = optim.Adam([u_vec], lr=0.05)
    for p in model.parameters():
        p.requires_grad = False

    loss_history = []
    pred_history = {}   # title → list of predicted ratings across steps

    # pre-compute item vectors (static)
    item_vecs = {}
    for item_idx, _ in rated_items:
        it = torch.tensor([item_idx], device=device)
        item_vecs[item_idx] = model.item_tower(it).detach()

    RECORD_EVERY = max(1, steps // 60)   # ~60 data points

    for step in range(1, steps + 1):
        opt.zero_grad()
        loss = torch.tensor(0.0, device=device)
        for item_idx, rating in rated_items:
            i_vec = item_vecs[item_idx]
            pred  = (u_vec * i_vec).sum()
            loss += (pred - rating) ** 2
        loss.backward()
        opt.step()

        if step % RECORD_EVERY == 0 or step == steps:
            loss_history.append({"step": step, "loss": loss.item()})
            for item_idx, rating in rated_items:
                i_vec = item_vecs[item_idx]
                with torch.no_grad():
                    p = (u_vec * i_vec).sum().item()
                pred_history.setdefault(item_idx, []).append(
                    {"step": step, "pred": p, "target": rating}
                )
            yield u_vec.detach().clone(), loss_history, pred_history

    for p in model.parameters():
        p.requires_grad = True


def recommend_new_user(u_vec, model, device, item_categories, movies, k=10):
    model.eval()
    with torch.no_grad():
        all_i  = torch.arange(len(item_categories), device=device)
        i_vecs = model.item_tower(all_i)
        preds  = (u_vec * i_vecs).sum(dim=1).cpu().numpy()
    topk = np.argsort(preds)[-k:][::-1]
    results = []
    for idx in topk:
        raw_id = item_categories[idx]
        row    = movies.loc[movies.movieId == raw_id]
        if row.empty:
            continue
        results.append({
            "Title":            row["title"].values[0],
            "Genres":           row["genres"].values[0].replace("|", " · "),
            "Predicted rating": round(float(preds[idx]), 2),
        })
    return results

# ── UI ─────────────────────────────────────────────────────────────────────────
if model_loaded:
    num_items = len(item_categories)

    rng        = np.random.default_rng(0)
    sample_idx = rng.choice(num_items, size=min(200, num_items), replace=False)
    sample_titles = {}
    for idx in sample_idx:
        raw_id = item_categories[idx]
        row    = movies.loc[movies.movieId == raw_id, "title"]
        if not row.empty:
            sample_titles[row.values[0]] = int(idx)
    sorted_titles = sorted(sample_titles.keys())

    # reverse lookup: item_idx → title
    idx_to_title = {v: k for k, v in sample_titles.items()}

    st.subheader("Step 1 — Rate some movies")
    st.write("Pick up to *5 movies* you have seen and rate them (1 = terrible, 5 = loved it).")

    ratings_input = []
    for i in range(5):
        col1, col2 = st.columns([3, 1])
        with col1:
            title = st.selectbox(f"Movie {i+1}", ["(skip)"] + sorted_titles, key=f"title_{i}")
        with col2:
            stars = st.slider("Rating", 1, 5, 3, key=f"rating_{i}")
        if title != "(skip)":
            ratings_input.append((sample_titles[title], float(stars)))

    st.subheader("Step 2 — How many recommendations?")
    k = st.slider("Top-K results", min_value=5, max_value=20, value=10)

    st.divider()

    if st.button("🎯 Get Recommendations", type="primary", disabled=len(ratings_input) == 0):

        # ── Animation section ──────────────────────────────────────────────────
        st.subheader("🧠 Learning your taste…")

        col_loss, col_pred = st.columns(2)

        with col_loss:
            st.markdown("*Optimisation loss* (lower = better fit to your ratings)")
            loss_chart = st.empty()

        with col_pred:
            st.markdown("*Predicted vs target rating* (per movie you rated)")
            pred_chart = st.empty()

        progress_bar  = st.progress(0, text="Optimising user profile…")
        status_text   = st.empty()

        final_u_vec   = None
        loss_history  = []
        pred_history  = {}
        STEPS         = 300

        for u_vec, loss_history, pred_history in get_new_user_vector_animated(
            ratings_input, model, device, steps=STEPS
        ):
            final_u_vec  = u_vec
            current_step = loss_history[-1]["step"]
            current_loss = loss_history[-1]["loss"]
            pct          = int(current_step / STEPS * 100)

            progress_bar.progress(pct, text=f"Step {current_step}/{STEPS}  |  Loss: {current_loss:.4f}")

            # Loss curve
            df_loss = pd.DataFrame(loss_history).set_index("step")
            loss_chart.line_chart(df_loss, y="loss", color="#FF4B4B", height=220)

            # Predicted vs target per rated movie
            rows = []
            for item_idx, pdata in pred_history.items():
                for pt in pdata:
                    rows.append({
                        "step":   pt["step"],
                        "pred":   pt["pred"],
                        "target": pt["target"],
                        "movie":  idx_to_title.get(item_idx, f"item {item_idx}"),
                    })
            if rows:
                df_pred = pd.DataFrame(rows)
                # Show one line per movie (latest pred vs target as horizontal ref)
                latest = df_pred.sort_values("step").groupby("movie").last().reset_index()
                # Pivot so each movie is its own column for line_chart
                pivot = df_pred.pivot_table(index="step", columns="movie", values="pred")
                pred_chart.line_chart(pivot, height=220)

            time.sleep(0.01)   # tiny pause so Streamlit can render

        progress_bar.progress(100, text="✅ Done!")
        status_text.success(f"Converged! Final loss: {loss_history[-1]['loss']:.4f}")

        # ── Summary stats ──────────────────────────────────────────────────────
        st.markdown("#### How well did the model fit your ratings?")
        summary_rows = []
        for item_idx, rating in ratings_input:
            if item_idx in pred_history and pred_history[item_idx]:
                final_pred = pred_history[item_idx][-1]["pred"]
                summary_rows.append({
                    "Movie":           idx_to_title.get(item_idx, f"item {item_idx}"),
                    "Your rating":     rating,
                    "Model predicted": round(final_pred, 2),
                    "Error":           round(abs(final_pred - rating), 2),
                })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.divider()

        # ── Recommendations ────────────────────────────────────────────────────
        st.subheader(f"🍿 Your Top {k} Picks")
        with st.spinner("Generating recommendations…"):
            recs = recommend_new_user(final_u_vec, model, device, item_categories, movies, k=k)

        df_recs = pd.DataFrame(recs)
        df_recs.index = range(1, len(df_recs) + 1)
        st.dataframe(df_recs, use_container_width=True)

    if len(ratings_input) == 0:
        st.info("Select at least one movie above to enable recommendations.")
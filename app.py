import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }
section[data-testid="stSidebar"] { background: rgba(255,255,255,0.04); border-right: 1px solid rgba(255,255,255,0.08); }
.hero-title { font-family: 'Syne', sans-serif; font-size: 2.6rem; font-weight: 800; background: linear-gradient(90deg, #f7971e, #ffd200); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.1; }
.hero-sub { font-size: 1rem; color: rgba(255,255,255,0.5); margin-top: 0.3rem; letter-spacing: 0.05em; }
.metric-card { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; padding: 1.2rem 1.5rem; text-align: center; }
.metric-label { font-size: 0.75rem; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; color: #ffd200; }
.metric-unit { font-size: 0.75rem; color: rgba(255,255,255,0.4); }
.pred-box { background: linear-gradient(135deg, rgba(247,151,30,0.15), rgba(255,210,0,0.08)); border: 1px solid rgba(255,210,0,0.3); border-radius: 20px; padding: 2rem; text-align: center; margin-top: 1rem; }
.pred-label { font-size: 0.85rem; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 0.1em; }
.pred-price { font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800; color: #ffd200; margin: 0.3rem 0; }
.section-header { font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; color: rgba(255,255,255,0.9); border-left: 3px solid #f7971e; padding-left: 0.7rem; margin: 1.5rem 0 1rem 0; }
.badge-success { background: rgba(72,199,142,0.15); border: 1px solid rgba(72,199,142,0.4); color: #48c78e; padding: 0.25rem 0.8rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
.badge-warning { background: rgba(255,210,0,0.12); border: 1px solid rgba(255,210,0,0.3); color: #ffd200; padding: 0.25rem 0.8rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
.algo-badge { display: inline-block; background: rgba(247,151,30,0.15); border: 1px solid rgba(247,151,30,0.3); color: #f7971e; padding: 0.2rem 0.7rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; margin-left: 0.5rem; }
.stButton > button { background: linear-gradient(90deg, #f7971e, #ffd200); color: #1a1a2e; border: none; border-radius: 10px; font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.95rem; padding: 0.6rem 2rem; transition: transform 0.2s, box-shadow 0.2s; }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(247,151,30,0.35); }
.stTabs [data-baseweb="tab"] { color: rgba(255,255,255,0.5); font-family: 'DM Sans', sans-serif; }
.stTabs [aria-selected="true"] { color: #ffd200 !important; border-bottom: 2px solid #ffd200 !important; }
hr { border-color: rgba(255,255,255,0.07); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  ALGORITHM REGISTRY
# ─────────────────────────────────────────────
ALGORITHMS = {
    "Linear Regression":  {"model": LinearRegression(),                                            "desc": "Simple baseline. Fast & interpretable."},
    "Ridge Regression":   {"model": Ridge(alpha=1.0),                                              "desc": "Linear + L2 penalty. Good when features correlate."},
    "Lasso Regression":   {"model": Lasso(alpha=1.0),                                              "desc": "Linear + L1 penalty. Auto-removes weak features."},
    "Decision Tree":      {"model": DecisionTreeRegressor(max_depth=6, random_state=42),           "desc": "Tree-based. Easy to visualize, can overfit."},
    "Random Forest":      {"model": RandomForestRegressor(n_estimators=100, random_state=42),      "desc": "Many trees averaged. Best balance of speed & accuracy."},
    "Gradient Boosting":  {"model": GradientBoostingRegressor(n_estimators=100, random_state=42),  "desc": "Sequential trees. Highest accuracy, slower to train."},
}

# ─────────────────────────────────────────────
#  ENCODING
# ─────────────────────────────────────────────
BINARY_COLS      = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]
FURNISHING_MAP   = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking",
                    "mainroad", "guestroom", "basement", "hotwaterheating",
                    "airconditioning", "prefarea", "furnishingstatus"]

def encode_df(df):
    df = df.copy()
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower().map({"yes": 1, "no": 0})
    if "furnishingstatus" in df.columns:
        df["furnishingstatus"] = df["furnishingstatus"].str.strip().str.lower().map(FURNISHING_MAP)
    return df

# ─────────────────────────────────────────────
#  CORE FUNCTIONS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    df = encode_df(df)
    df = df.dropna()
    available = [c for c in NUMERIC_FEATURES if c in df.columns]
    if "price" not in df.columns:
        st.error("No 'price' column found.")
        return None, None
    return df[available], df["price"]

def train_model(X, y, algo_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = ALGORITHMS[algo_name]["model"]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "MAE":      round(float(mean_absolute_error(y_test, y_pred)), 2),
        "MSE":      round(float(mean_squared_error(y_test, y_pred)), 2),
        "RMSE":     round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "R2":       round(float(r2_score(y_test, y_pred)), 4),
        "algo":     algo_name,
        "features": list(X.columns),
    }
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": list(X.columns)}, f)
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    return model, metrics, X_test, y_test, y_pred

def load_saved():
    model, features, metrics = None, NUMERIC_FEATURES, None
    if os.path.exists("models/model.pkl"):
        with open("models/model.pkl", "rb") as f:
            saved = pickle.load(f)
            model    = saved["model"]    if isinstance(saved, dict) else saved
            features = saved["features"] if isinstance(saved, dict) else NUMERIC_FEATURES
    if os.path.exists("reports/metrics.json"):
        with open("reports/metrics.json", "r") as f:
            metrics = json.load(f)
    return model, features, metrics

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for k in ["model", "metrics", "df", "X_test", "y_test", "y_pred", "features"]:
    if k not in st.session_state:
        st.session_state[k] = None

if st.session_state.model is None:
    m, f, met = load_saved()
    st.session_state.model    = m
    st.session_state.features = f
    st.session_state.metrics  = met

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title">🏠 HouseLens</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">ML Price Prediction</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📂 Upload Dataset")
    uploaded = st.file_uploader("Upload Housing CSV", type=["csv"], label_visibility="collapsed")
    if uploaded:
        df = load_data(uploaded)
        if df is not None:
            st.session_state.df = df
            st.markdown(f'<span class="badge-success">✓ {len(df):,} rows loaded</span>', unsafe_allow_html=True)
            st.caption(f"Columns: {', '.join(df.columns)}")

    st.markdown("---")
    st.markdown("### 🤖 Choose Algorithm")
    algo_choice = st.selectbox("Algorithm", list(ALGORITHMS.keys()), index=4, label_visibility="collapsed")
    st.caption(ALGORITHMS[algo_choice]["desc"])

    st.markdown("---")
    st.markdown("### ⚙️ Model Status")
    if st.session_state.model is not None:
        algo_name = st.session_state.metrics.get("algo", "Unknown") if st.session_state.metrics else "Unknown"
        st.markdown(f'<span class="badge-success">✓ Model Ready</span> <span class="algo-badge">{algo_name}</span>', unsafe_allow_html=True)
        if st.session_state.metrics:
            st.caption(f"R² Score: {st.session_state.metrics.get('R2', 'N/A')}")
    else:
        st.markdown('<span class="badge-warning">⚠ Not Trained Yet</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Built by Aryansingh | Naresh IT Technologies")

# ─────────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏋️ Train Model", "🔮 Predict Price", "📊 Metrics", "🔁 Compare Algorithms", "🔍 Data Explorer"])

# ══════════════════════════════════════════════
#  TAB 1 — TRAIN
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Train Your Model</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("👈 Upload your Housing CSV from the sidebar to get started.")
    else:
        df = st.session_state.df
        encoded_preview = encode_df(df)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Dataset Preview (after encoding)**")
            st.dataframe(encoded_preview.head(10), use_container_width=True, height=280)
        with col2:
            st.markdown("**Dataset Info**")
            st.metric("Total Rows",    f"{len(df):,}")
            st.metric("Features Used", len([c for c in NUMERIC_FEATURES if c in df.columns]))
            st.metric("Missing Values", df.isnull().sum().sum())

        st.markdown("---")
        with st.expander("ℹ️ How are text columns encoded?"):
            st.markdown("""
| Column | Original | Encoded |
|---|---|---|
| mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea | yes / no | 1 / 0 |
| furnishingstatus | unfurnished / semi-furnished / furnished | 0 / 1 / 2 |

This is called **Label Encoding** — converting text to numbers so the ML model can understand them.
""")

        st.markdown(f"**Selected Algorithm:** `{algo_choice}` — {ALGORITHMS[algo_choice]['desc']}")

        if st.button(f"🚀 Train {algo_choice}", use_container_width=True):
            X, y = preprocess_data(df)
            if X is not None:
                with st.spinner(f"Training {algo_choice}..."):
                    model, metrics, X_test, y_test, y_pred = train_model(X, y, algo_choice)
                    st.session_state.model    = model
                    st.session_state.metrics  = metrics
                    st.session_state.X_test   = X_test
                    st.session_state.y_test   = y_test
                    st.session_state.y_pred   = y_pred
                    st.session_state.features = metrics["features"]

                st.success(f"✅ {algo_choice} trained and saved!")

                c1, c2, c3, c4 = st.columns(4)
                for col, label, val, unit in [
                    (c1, "MAE",      metrics["MAE"],  "₹"),
                    (c2, "MSE",      metrics["MSE"],  ""),
                    (c3, "RMSE",     metrics["RMSE"], "₹"),
                    (c4, "R² Score", metrics["R2"],   ""),
                ]:
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value">{val:,}</div>
                            <div class="metric-unit">{unit}</div>
                        </div>""", unsafe_allow_html=True)

                r2 = metrics["R2"]
                if r2 >= 0.80:
                    st.success(f"🎉 Excellent! R² = {r2} — model explains {r2*100:.1f}% of price variation.")
                elif r2 >= 0.65:
                    st.warning(f"👍 Good. R² = {r2} ")
                else:
                    st.error(f"⚠️ R² = {r2} — model is struggling. Try a different algorithm from the sidebar.")

# ══════════════════════════════════════════════
#  TAB 2 — PREDICT
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Predict House Price</div>', unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning("⚠️ Train a model first in the **Train Model** tab!")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("**Numeric Features**")
            area      = st.number_input("🏗️ Area (sq ft)", min_value=100, max_value=50000, value=1500, step=50)
            bedrooms  = st.slider("🛏️ Bedrooms",       1, 10, 3)
            bathrooms = st.slider("🚿 Bathrooms",       1,  8, 2)
            stories   = st.slider("🏢 Stories",         1,  4, 1)
            parking   = st.slider("🅿️ Parking spots",  0,  4, 1)

            st.markdown("**Amenities**")
            c1, c2, c3 = st.columns(3)
            mainroad  = c1.selectbox("Main Road",  ["yes", "no"])
            guestroom = c2.selectbox("Guest Room", ["yes", "no"])
            basement  = c3.selectbox("Basement",   ["yes", "no"])
            c4, c5, c6 = st.columns(3)
            hotwater  = c4.selectbox("Hot Water",  ["yes", "no"])
            aircon    = c5.selectbox("Air Con",    ["yes", "no"])
            prefarea  = c6.selectbox("Pref Area",  ["yes", "no"])
            furnishing = st.select_slider("Furnishing Status",
                options=["unfurnished", "semi-furnished", "furnished"], value="semi-furnished")

            predict_btn = st.button("🔮 Predict Price", use_container_width=True)

        with col2:
            if predict_btn:
                yn = lambda v: 1 if v == "yes" else 0
                fm = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
                raw = {
                    "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
                    "stories": stories, "parking": parking,
                    "mainroad": yn(mainroad), "guestroom": yn(guestroom),
                    "basement": yn(basement), "hotwaterheating": yn(hotwater),
                    "airconditioning": yn(aircon), "prefarea": yn(prefarea),
                    "furnishingstatus": fm[furnishing],
                }
                features  = st.session_state.features or NUMERIC_FEATURES
                input_df  = pd.DataFrame([[raw.get(f, 0) for f in features]], columns=features)
                prediction = st.session_state.model.predict(input_df)[0]

                st.markdown(f"""
                <div class="pred-box">
                    <div class="pred-label">Estimated Price</div>
                    <div class="pred-price">₹ {prediction:,.0f}</div>
                    <div style="color:rgba(255,255,255,0.4);font-size:0.8rem;margin-top:0.5rem;">
                        {area} sq ft · {bedrooms} bed · {bathrooms} bath · {furnishing}
                    </div>
                </div>""", unsafe_allow_html=True)

                st.markdown('<div class="section-header" style="font-size:1rem;">Your Input Summary</div>',
                             unsafe_allow_html=True)
                input_display = pd.DataFrame({"Feature": features, "Value": [raw.get(f, 0) for f in features]})
                fig = px.bar(input_display, x="Value", y="Feature", orientation="h",
                    color_discrete_sequence=["#f7971e"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                    font=dict(color="rgba(255,255,255,0.7)"),
                    margin=dict(l=0,r=0,t=10,b=0), height=320,
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("""
                <div style="height:300px;display:flex;align-items:center;justify-content:center;
                     color:rgba(255,255,255,0.2);font-size:1.1rem;text-align:center;">
                    Fill in details and click<br><b>Predict Price</b> 🔮
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
#  TAB 3 — METRICS
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Model Evaluation</div>', unsafe_allow_html=True)

    metrics = st.session_state.metrics
    if metrics is None:
        st.info("No metrics yet. Train the model first.")
    else:
        st.markdown(f"**Algorithm:** `{metrics.get('algo', 'Unknown')}`")

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, desc in [
            (c1, "MAE",  metrics["MAE"],  "Mean Absolute Error"),
            (c2, "MSE",  metrics["MSE"],  "Mean Squared Error"),
            (c3, "RMSE", metrics["RMSE"], "Root MSE"),
            (c4, "R²",   metrics["R2"],   "Coefficient of Determination"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{desc}</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-unit">{label}</div>
                </div>""", unsafe_allow_html=True)

        if st.session_state.y_test is not None:
            y_test = np.array(st.session_state.y_test)
            y_pred = np.array(st.session_state.y_pred)

            st.markdown('<div class="section-header">Actual vs Predicted</div>', unsafe_allow_html=True)
            mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                marker=dict(color="#ffd200", opacity=0.6, size=6), name="Predictions"))
            fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                line=dict(color="#f7971e", dash="dash"), name="Perfect Fit"))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(color="rgba(255,255,255,0.7)"),
                xaxis_title="Actual Price", yaxis_title="Predicted Price",
                margin=dict(l=0,r=0,t=10,b=0), height=380,
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-header">Residuals Distribution</div>', unsafe_allow_html=True)
            fig2 = px.histogram(y_test - y_pred, nbins=40, color_discrete_sequence=["#f7971e"],
                labels={"value": "Residual", "count": "Frequency"})
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(color="rgba(255,255,255,0.7)"),
                margin=dict(l=0,r=0,t=10,b=0), height=260,
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 4 — COMPARE ALGORITHMS
# ══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Compare All Algorithms</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("Upload a dataset first.")
    else:
        st.markdown("Trains **all 6 algorithms** on your data and ranks them by R² score.")
        if st.button("⚡ Run All Algorithms & Compare", use_container_width=True):
            X, y = preprocess_data(st.session_state.df)
            if X is not None:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                results = []
                prog = st.progress(0)
                for i, (name, info) in enumerate(ALGORITHMS.items()):
                    with st.spinner(f"Training {name}..."):
                        m = info["model"]
                        m.fit(X_train, y_train)
                        yp = m.predict(X_test)
                        results.append({
                            "Algorithm": name,
                            "R²":   round(r2_score(y_test, yp), 4),
                            "MAE":  round(mean_absolute_error(y_test, yp), 0),
                            "RMSE": round(np.sqrt(mean_squared_error(y_test, yp)), 0),
                        })
                    prog.progress((i + 1) / len(ALGORITHMS))

                results_df = pd.DataFrame(results).sort_values("R²", ascending=False).reset_index(drop=True)
                results_df.index += 1
                best = results_df.iloc[0]["Algorithm"]
                st.success(f"🏆 Best algorithm: **{best}** with R² = {results_df.iloc[0]['R²']}")

                fig = px.bar(results_df, x="Algorithm", y="R²",
                    color="R²", color_continuous_scale=["#302b63", "#f7971e", "#ffd200"], text="R²")
                fig.update_traces(textposition="outside", textfont_color="rgba(255,255,255,0.8)")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                    font=dict(color="rgba(255,255,255,0.7)"),
                    margin=dict(l=0,r=0,t=30,b=0), height=350,
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0, 1]),
                    coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(results_df, use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 5 — DATA EXPLORER
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Explore Your Dataset</div>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("Upload a CSV from the sidebar to explore it here.")
    else:
        df = st.session_state.df
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Statistical Summary**")
            st.dataframe(df.describe().round(2), use_container_width=True)
        with col2:
            st.markdown("**Missing Values**")
            missing_df = df.isnull().sum().reset_index()
            missing_df.columns = ["Column", "Missing"]
            missing_df = missing_df[missing_df["Missing"] > 0]
            if missing_df.empty:
                st.success("✅ No missing values found!")
            else:
                st.dataframe(missing_df, use_container_width=True)

        st.markdown("---")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Distribution Plot**")
            selected_col = st.selectbox("Choose column", num_cols)
            fig = px.histogram(df, x=selected_col, nbins=40, color_discrete_sequence=["#ffd200"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(color="rgba(255,255,255,0.7)"),
                margin=dict(l=0,r=0,t=10,b=0), height=280,
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Price vs Feature Scatter**")
            scatter_col = st.selectbox("Compare with price", [c for c in num_cols if c != "price"])
            if "price" in df.columns:
                fig2 = px.scatter(df, x=scatter_col, y="price",
                    color_discrete_sequence=["#f7971e"], opacity=0.6)
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
                    font=dict(color="rgba(255,255,255,0.7)"),
                    margin=dict(l=0,r=0,t=10,b=0), height=280,
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
                st.plotly_chart(fig2, use_container_width=True)

        if len(num_cols) >= 2:
            st.markdown("**Correlation Heatmap**")
            corr = df[num_cols].corr().round(2)
            fig3 = px.imshow(corr, text_auto=True,
                color_continuous_scale=[[0,"#1a1a2e"],[0.5,"#302b63"],[1,"#ffd200"]])
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="rgba(255,255,255,0.7)"),
                margin=dict(l=0,r=0,t=10,b=0), height=420)
            st.plotly_chart(fig3, use_container_width=True)

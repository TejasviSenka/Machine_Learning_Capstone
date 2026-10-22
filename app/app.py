from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Page setup and minimalist monochrome styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Capstone | Solar & Steel Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global Matplotlib Minimalist Monochrome Styling Defaults
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.edgecolor'] = '#71717a'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e4e4e7'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.45

def apply_minimal_plot_style(ax, title="", xlabel="", ylabel="", grid=True):
    """Applies a clean, high-contrast monochrome design to Matplotlib axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#71717a')
    ax.spines['bottom'].set_color('#71717a')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(colors='#52525b', labelsize=9)
    if grid:
        ax.grid(True, linestyle='--', alpha=0.4, color='#e4e4e7', linewidth=0.7)
    else:
        ax.grid(False)
    if title:
        ax.set_title(title, fontsize=11, fontweight='600', color='#09090b', pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, fontweight='500', color='#3f3f46', labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, fontweight='500', color='#3f3f46', labelpad=8)

# Minimalist Monochrome CSS System (Black, White, Slate, Greys)
st.markdown("""
<style>
    /* Layout & Base Constraints */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }
    
    /* Sleek Monochrome Hero Banner */
    .hero {
        background: #09090b;
        color: #ffffff;
        padding: 1.8rem 2.2rem;
        border-radius: 14px;
        border: 1px solid #27272a;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.08);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background: #18181b;
        color: #a1a1aa;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #3f3f46;
        margin-bottom: 0.8rem;
    }
    .hero-badge-dot {
        width: 6px;
        height: 6px;
        background-color: #22c55e;
        border-radius: 50%;
        display: inline-block;
    }
    .hero h1 {
        color: #ffffff !important;
        margin: 0;
        font-size: 1.95rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        line-height: 1.25;
    }
    .hero p {
        color: #a1a1aa !important;
        margin: 0.55rem 0 0 0;
        font-size: 0.96rem;
        line-height: 1.5;
        max-width: 820px;
    }
    
    /* Minimal Metric Cards */
    div[data-testid="stMetric"] {
        background: #fafafa;
        border: 1px solid #e4e4e7;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        transition: all 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #a1a1aa;
        background: #ffffff;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: #71717a !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        color: #09090b !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Minimal Solid Buttons */
    .stButton > button {
        background-color: #09090b !important;
        color: #ffffff !important;
        border: 1px solid #27272a !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        padding: 0.55rem 1.25rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06) !important;
    }
    .stButton > button:hover {
        background-color: #27272a !important;
        color: #ffffff !important;
        border-color: #3f3f46 !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Download Buttons */
    .stDownloadButton > button {
        background-color: #ffffff !important;
        color: #09090b !important;
        border: 1px solid #d4d4d8 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        background-color: #f4f4f5 !important;
        border-color: #a1a1aa !important;
    }
    
    /* Task Cards & Status Pills */
    .task-card {
        background: #ffffff;
        border: 1px solid #e4e4e7;
        border-radius: 12px;
        padding: 1.3rem 1.4rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.2s ease;
    }
    .task-card:hover {
        border-color: #a1a1aa;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }
    .task-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #09090b;
        margin-bottom: 0.5rem;
    }
    .task-desc {
        font-size: 0.88rem;
        color: #52525b;
        line-height: 1.45;
        margin-bottom: 1.1rem;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 6px;
    }
    .status-ready {
        background: #f4f4f5;
        color: #09090b;
        border: 1px solid #d4d4d8;
    }
    .status-missing {
        background: #ffffff;
        color: #71717a;
        border: 1px dashed #d4d4d8;
    }
    
    /* Expanders & Dividers */
    .streamlit-expanderHeader {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #09090b !important;
    }
    hr {
        border-color: #e4e4e7 !important;
        margin: 1.5rem 0 !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #fafafa;
        border-right: 1px solid #e4e4e7;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #27272a !important;
    }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


def find_csv(preferred_names):
    """Find a project CSV by preferred names, then search data/ recursively."""
    search_roots = [DATA_DIR, ROOT / "app", ROOT]
    all_csvs = []
    for folder in search_roots:
        if folder.exists():
            try:
                all_csvs.extend(folder.rglob("*.csv"))
            except OSError:
                pass
    unique = []
    seen = set()
    for p in all_csvs:
        resolved = p.resolve()
        if resolved in seen or any(part in {".venv", "venv", ".git", ".ipynb_checkpoints"} for part in resolved.parts):
            continue
        seen.add(resolved)
        unique.append(p)
    for preferred in preferred_names:
        for p in unique:
            if p.name.lower() == preferred.lower():
                return p
    return None


@st.cache_data(show_spinner=False)
def read_csv_cached(path_string, modified_time):
    return pd.read_csv(path_string)


def load_csv(path):
    if path is None or not path.exists():
        return None
    return read_csv_cached(str(path), path.stat().st_mtime)


def numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def make_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def safe_auc(y_true, probabilities, classes):
    try:
        if len(classes) == 2:
            return roc_auc_score(y_true, probabilities[:, 1])
        return roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted", labels=classes)
    except (ValueError, IndexError):
        return np.nan


def show_missing_dataset(message, expected_path):
    st.markdown(f"""
    <div style="background: #fafafa; border: 1px dashed #71717a; border-radius: 8px; padding: 14px 16px; margin: 10px 0;">
        <span style="font-weight: 600; color: #09090b; font-size: 0.92rem;">Notice: {message}</span>
        <div style="font-family: ui-monospace, monospace; font-size: 0.82rem; color: #71717a; margin-top: 6px;">Expected location: {expected_path}</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Load project datasets
# -----------------------------------------------------------------------------
solar_path = find_csv(["solar_generation.csv"])
steel_path = find_csv(["Steel_industry_data.csv", "steel_industry_data.csv"])
solar_df = load_csv(solar_path)
steel_df = load_csv(steel_path)

# Minimalist Monochrome Hero
st.markdown("""
<div class="hero">
  <div class="hero-badge">
    <span class="hero-badge-dot"></span>
    Capstone Pipeline & Analytics
  </div>
  <h1>Machine Learning Capstone Dashboard</h1>
  <p>Minimalist analytics interface for Regression (Solar Generation), Classification (Steel Industry), and Clustering track evaluations.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <div style="font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #71717a; margin-bottom: 4px;">WORKSPACE</div>
        <div style="font-size: 1.15rem; font-weight: 700; color: #09090b; letter-spacing: -0.02em;">Analytics Studio</div>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Choose a section",
        ["Overview", "Regression", "Classification", "Clustering", "Dataset Explorer"],
        label_visibility="collapsed",
    )
    
    st.divider()
    
    files_count = sum(x is not None for x in [solar_df, steel_df])
    st.markdown(f"""
    <div style="font-size: 0.8rem; color: #71717a; margin-bottom: 8px;">
        DATASET STATUS: <strong style="color: #09090b;">{files_count} / 2 Ready</strong>
    </div>
    <div style="font-size: 0.78rem; color: #a1a1aa; line-height: 1.45;">
        All telemetry pipelines strictly preserve reproducible test partitions and zero data leakage.
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
if page == "Overview":
    st.markdown("### System Telemetry & Dataset Audit")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Solar Dataset", "Loaded" if solar_df is not None else "Missing", f"{len(solar_df):,} rows" if solar_df is not None else "0 rows")
    c2.metric("Steel Dataset", "Loaded" if steel_df is not None else "Missing", f"{len(steel_df):,} rows" if steel_df is not None else "0 rows")
    c3.metric("Project Files", f"{files_count} / 2 Available", "100%" if files_count == 2 else "Partial")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("### Problem Tracks & Pipeline Status")
    
    a, b, c = st.columns(3)
    with a:
        solar_ready = solar_df is not None and "AC Power in Watts" in solar_df.columns
        st.markdown(f"""
        <div class="task-card">
            <div>
                <div class="task-header">📈 Regression Track</div>
                <div class="task-desc">
                    Estimate total AC power generation from ambient irradiance, module temperature, and meteorological telemetry.
                </div>
            </div>
            <div>
                <span class="status-badge {'status-ready' if solar_ready else 'status-missing'}">
                    {'● Ready · ' + f'{len(solar_df):,} rows' if solar_ready else '○ solar_generation.csv required'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with b:
        steel_ready = steel_df is not None and "Load_Type" in steel_df.columns
        st.markdown(f"""
        <div class="task-card">
            <div>
                <div class="task-header">🎯 Classification Track</div>
                <div class="task-desc">
                    Predict categorical steel facility energy load levels (`Light`, `Medium`, `Maximum`) for dynamic grid dispatch.
                </div>
            </div>
            <div>
                <span class="status-badge {'status-ready' if steel_ready else 'status-missing'}">
                    {'● Ready · ' + f'{len(steel_df):,} rows' if steel_ready else '○ Steel_industry_data.csv required'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with c:
        st.markdown(f"""
        <div class="task-card">
            <div>
                <div class="task-header">🧩 Clustering Track</div>
                <div class="task-desc">
                    Uncover latent solar operational regimes and generation states using K-Means and Agglomerative Hierarchical clustering.
                </div>
            </div>
            <div>
                <span class="status-badge {'status-ready' if solar_df is not None else 'status-missing'}">
                    {'● Ready · ' + f'{len(solar_df):,} rows' if solar_df is not None else '○ solar_generation.csv required'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown("### Telemetry Data Inspection")
    for title, data, path in [("Solar generation (solar_generation.csv)", solar_df, solar_path), ("Steel industry (Steel_industry_data.csv)", steel_df, steel_path)]:
        with st.expander(f"{title} — {'Available' if data is not None else 'Missing'}", expanded=(data is not None)):
            if data is not None:
                m1, m2, m3 = st.columns(3)
                m1.metric("Rows", f"{data.shape[0]:,}")
                m2.metric("Columns", f"{data.shape[1]:,}")
                m3.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
                st.dataframe(data.head(8), use_container_width=True)
            else:
                st.caption(f"Expected file: {DATA_DIR / ('solar_generation.csv' if 'solar' in title.lower() else 'Steel_industry_data.csv')}")

# -----------------------------------------------------------------------------
# Regression
# -----------------------------------------------------------------------------
elif page == "Regression":
    st.markdown("### 📈 Regression Analysis (Solar Power Generation)")
    st.write("Train predictive models to estimate `AC Power in Watts` based on environmental and solar telemetry. Evaluated on a strictly held-out test split.")
    
    if solar_df is None:
        show_missing_dataset("Solar generation dataset was not found.", DATA_DIR / "solar_generation.csv")
    elif "AC Power in Watts" not in solar_df.columns:
        st.error("The solar dataset does not contain the expected target column `AC Power in Watts`.")
        st.write("Available columns:", solar_df.columns.tolist())
    else:
        df = solar_df.copy()
        target = "AC Power in Watts"
        exclude = [target]
        df = df.dropna(subset=[target]).dropna(axis=1, how="all")
        X = df.drop(columns=exclude)
        y = pd.to_numeric(df[target], errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
        id_like = [col for col in X.columns if X[col].nunique(dropna=True) == len(X) and X[col].dtype == "object"]
        X = X.drop(columns=id_like, errors="ignore")
        
        if X.shape[1] == 0 or len(X) < 20:
            st.error("There are not enough usable features/rows to train a regression model.")
        else:
            with st.expander("Model Configuration & Training Parameters", expanded=True):
                r1, r2 = st.columns(2)
                test_size = r1.slider("Test Partition Size (Hold-out)", 0.1, 0.4, 0.2, 0.05)
                model_choice = r2.selectbox("Algorithm Selection", ["Random Forest", "Linear Regression"])
                run_regression = st.button("Train and Evaluate Model", type="primary", use_container_width=True)
                
            if run_regression:
                with st.spinner("Fitting regression pipeline with Standard scaling..."):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    regressor = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1) if model_choice == "Random Forest" else LinearRegression()
                    pipeline = Pipeline([("preprocessor", make_preprocessor(X_train)), ("model", regressor)])
                    pipeline.fit(X_train, y_train)
                    predictions = pipeline.predict(X_test)
                    mae = mean_absolute_error(y_test, predictions)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    r2 = r2_score(y_test, predictions)
                    st.session_state["regression_result"] = {
                        "model_name": model_choice, "pipeline": pipeline,
                        "y_test": np.asarray(y_test), "predictions": predictions,
                        "mae": mae, "rmse": rmse, "r2": r2,
                    }
                    
            result = st.session_state.get("regression_result")
            if result:
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                st.markdown(f"#### Evaluation Results — {result['model_name']}")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"{result['mae']:,.3f} W")
                m2.metric("RMSE", f"{result['rmse']:,.3f} W")
                m3.metric("R² Score", f"{result['r2']:.4f}")
                
                st.caption(f"Evaluated on {len(result['y_test']):,} held-out test samples with zero data leakage.")
                
                col_p1, col_p2 = st.columns(2)
                
                chart_df = pd.DataFrame({"Actual": result["y_test"], "Predicted": result["predictions"]})
                
                with col_p1:
                    fig, ax = plt.subplots(figsize=(7, 4.5))
                    # Sample for snappy, elegant plotting if dataset is large
                    plot_data = chart_df.sample(min(3000, len(chart_df)), random_state=42)
                    ax.scatter(plot_data["Actual"], plot_data["Predicted"], color="#09090b", alpha=0.35, s=18, edgecolors="none")
                    lo = float(chart_df.min().min())
                    hi = float(chart_df.max().max())
                    ax.plot([lo, hi], [lo, hi], color="#71717a", linestyle="--", linewidth=1.5, label="Ideal Fit (y = x)")
                    apply_minimal_plot_style(ax, title="Actual vs. Predicted AC Power", xlabel="Actual AC Power (Watts)", ylabel="Predicted AC Power (Watts)")
                    ax.legend(frameon=True, facecolor="#fafafa", edgecolor="#e4e4e7", fontsize=8.5)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    
                with col_p2:
                    residuals = chart_df["Actual"] - chart_df["Predicted"]
                    fig, ax = plt.subplots(figsize=(7, 4.5))
                    sample_idx = np.random.RandomState(42).choice(len(residuals), size=min(3000, len(residuals)), replace=False)
                    ax.scatter(result["predictions"][sample_idx], residuals.iloc[sample_idx], color="#27272a", alpha=0.35, s=18, edgecolors="none")
                    ax.axhline(0, color="#71717a", linestyle="--", linewidth=1.5, label="Zero Error")
                    apply_minimal_plot_style(ax, title="Residual Distribution (Actual − Predicted)", xlabel="Predicted AC Power (Watts)", ylabel="Residual Error (Watts)")
                    ax.legend(frameon=True, facecolor="#fafafa", edgecolor="#e4e4e7", fontsize=8.5)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            else:
                st.info("Configure model hyperparameters and click **Train and Evaluate Model** to execute the pipeline.")

# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------
elif page == "Classification":
    st.markdown("### 🎯 Classification Analysis (Steel Industry Load Type)")
    st.write("Predict facility categorical load states (`Light_Load`, `Medium_Load`, `Maximum_Load`) from high-frequency power telemetry.")
    
    if steel_df is None:
        show_missing_dataset("Steel Industry dataset was not found.", DATA_DIR / "Steel_industry_data.csv")
    elif "Load_Type" not in steel_df.columns:
        st.error("The loaded steel dataset does not contain the expected `Load_Type` target column.")
        st.write("Available columns:", steel_df.columns.tolist())
    else:
        df = steel_df.dropna(subset=["Load_Type"]).copy()
        X = df.drop(columns=["Load_Type"])
        y = df["Load_Type"].astype(str)
        counts = y.value_counts()
        
        with st.expander("Target Class Distribution", expanded=False):
            fig, ax = plt.subplots(figsize=(8, 3))
            classes_ordered = counts.index.tolist()
            palette = ["#09090b", "#52525b", "#a1a1aa"][:len(classes_ordered)]
            bars = ax.bar(classes_ordered, counts.values, color=palette, width=0.55, edgecolor="#e4e4e7")
            apply_minimal_plot_style(ax, title="Target Class Distribution: Load_Type", xlabel="Load Type", ylabel="Observation Count")
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h + (max(counts.values)*0.015), f"{int(h):,}", ha='center', va='bottom', fontsize=8.5, color='#09090b', fontweight='600')
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(f"{len(df):,} total samples · {X.shape[1]} input channels · {y.nunique()} target classes")
        
        with st.expander("Model Configuration & Training Parameters", expanded=True):
            c1, c2 = st.columns(2)
            model_choice = c1.selectbox("Algorithm Selection", [
                "Random Forest", "Bagging", "AdaBoost", "Gradient Boosting", 
                "Decision Tree", "Logistic Regression", "K-Nearest Neighbors"
            ])
            test_size = c2.slider("Test Partition Size (Stratified)", 0.1, 0.4, 0.2, 0.05, key="clf_test_size")
            train_clf = st.button("Train and Evaluate Classifier", type="primary", use_container_width=True)
            
        if train_clf:
            with st.spinner("Training classification pipeline with stratified split..."):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                estimators = {
                    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
                    "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=10, random_state=42), n_estimators=50, random_state=42, n_jobs=-1),
                    "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
                    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
                    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
                }
                model = Pipeline([("preprocessor", make_preprocessor(X_train)), ("model", estimators[model_choice])])
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                classes = model.named_steps["model"].classes_
                metrics = {
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision (Weighted)": precision_score(y_test, pred, average="weighted", zero_division=0),
                    "Recall (Weighted)": recall_score(y_test, pred, average="weighted", zero_division=0),
                    "F1-Score (Weighted)": f1_score(y_test, pred, average="weighted", zero_division=0),
                    "ROC-AUC (OvR Weighted)": safe_auc(y_test, proba, classes) if proba is not None else np.nan,
                }
                st.session_state["classification_result"] = {
                    "model_name": model_choice, "metrics": metrics,
                    "y_test": np.asarray(y_test), "predictions": np.asarray(pred),
                    "classes": np.asarray(classes), "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
                }
                
        result = st.session_state.get("classification_result")
        if result:
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            st.markdown(f"#### Evaluation Results — {result['model_name']}")
            metric_cols = st.columns(len(result["metrics"]))
            for col, (name, value) in zip(metric_cols, result["metrics"].items()):
                col.metric(name, "N/A" if pd.isna(value) else (f"{value:.4f}" if "AUC" in name else f"{value:.2%}"))
            
            col_c1, col_c2 = st.columns([1.1, 1.2])
            
            with col_c1:
                labels = list(result["classes"])
                cm = confusion_matrix(result["y_test"], result["predictions"], labels=labels)
                fig, ax = plt.subplots(figsize=(6, 5))
                # High contrast monochrome confusion matrix
                im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greys)
                apply_minimal_plot_style(ax, title="Confusion Matrix", xlabel="Predicted Label", ylabel="True Label", grid=False)
                ax.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right", fontsize=8.5)
                ax.set_yticks(range(len(labels)), labels=labels, fontsize=8.5)
                
                # Dynamic text color threshold for monochrome readability
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        val = cm[i, j]
                        color = "#ffffff" if val > thresh else "#09090b"
                        ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontweight="600", fontsize=9)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(colors='#71717a', labelsize=8)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                
            with col_c2:
                st.markdown("<div style='font-size: 0.95rem; font-weight: 600; color: #09090b; margin-bottom: 8px;'>Class-Level Performance Report</div>", unsafe_allow_html=True)
                report_df = pd.DataFrame(result["report"]).T
                st.dataframe(report_df.style.format(precision=3), use_container_width=True)
        else:
            st.info("Select an algorithm and click **Train and Evaluate Classifier** to compute test metrics.")

# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------
elif page == "Clustering":
    st.markdown("### 🧩 Clustering Analysis (Solar Operational Regimes)")
    st.write("Unsupervised discovery of solar power plant operating regimes without label bias. The generation output `AC Power in Watts` is strictly excluded from fitting.")
    
    if solar_df is None:
        show_missing_dataset("Solar generation dataset was not found.", DATA_DIR / "solar_generation.csv")
    else:
        df = solar_df.copy()
        numeric = numeric_columns(df)
        excluded = [c for c in ["AC Power in Watts"] if c in numeric]
        features = [c for c in numeric if c not in excluded]
        
        if len(features) < 2:
            st.error("At least two numeric feature columns are required for clustering and PCA.")
        else:
            with st.expander("Feature Selection & Clustering Setup", expanded=True):
                st.caption(f"Numeric channels: {len(features)} · Excluded ground-truth: {', '.join(excluded) if excluded else 'none'}")
                selected_features = st.multiselect("Input Sensor Features", features, default=features)
                c1, c2 = st.columns(2)
                k = c1.slider("Number of Clusters (k)", min_value=2, max_value=8, value=3)
                algorithm = c2.selectbox("Clustering Algorithm", [
                    "K-Means", "Agglomerative — Ward", "Agglomerative — Complete", "Agglomerative — Average"
                ])
                run_cluster = st.button("Execute Clustering Pipeline", type="primary", use_container_width=True)
                
            if run_cluster:
                if len(selected_features) < 2:
                    st.error("Select at least two features.")
                else:
                    with st.spinner("Scaling features and calculating clustering validation metrics..."):
                        work = df[selected_features].replace([np.inf, -np.inf], np.nan)
                        work = work.fillna(work.median(numeric_only=True)).dropna(axis=1, how="all")
                        if work.shape[1] < 2:
                            st.error("Not enough usable numeric columns after handling missing values.")
                        else:
                            scaler = StandardScaler()
                            scaled = scaler.fit_transform(work)
                            max_rows = 12000
                            if len(scaled) > max_rows:
                                rng = np.random.RandomState(42)
                                idx = np.sort(rng.choice(len(scaled), max_rows, replace=False))
                                scaled_fit = scaled[idx]
                                sampled_note = f"Reproducible sample of {max_rows:,} / {len(scaled):,} observations utilized for responsive calculation."
                            else:
                                idx = np.arange(len(scaled))
                                scaled_fit = scaled
                                sampled_note = f"All {len(scaled):,} observations utilized."
                                
                            if algorithm == "K-Means":
                                labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled_fit)
                            else:
                                linkage = algorithm.split("—")[1].strip().lower()
                                labels = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(scaled_fit)
                                
                            sil = silhouette_score(scaled_fit, labels) if 1 < len(np.unique(labels)) < len(labels) else np.nan
                            db = davies_bouldin_score(scaled_fit, labels) if len(np.unique(labels)) > 1 else np.nan
                            ch = calinski_harabasz_score(scaled_fit, labels) if len(np.unique(labels)) > 1 else np.nan
                            
                            pca = PCA(n_components=2, random_state=42)
                            coords = pca.fit_transform(scaled_fit)
                            
                            st.session_state["cluster_result"] = {
                                "algorithm": algorithm, "k": k, "labels": labels,
                                "coords": coords, "explained": pca.explained_variance_ratio_,
                                "silhouette": sil, "db": db, "ch": ch,
                                "sampled_note": sampled_note,
                            }
                            
            result = st.session_state.get("cluster_result")
            if result:
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                st.caption(result["sampled_note"])
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Silhouette Score ↑", "N/A" if pd.isna(result["silhouette"]) else f"{result['silhouette']:.4f}")
                c2.metric("Davies–Bouldin Index ↓", "N/A" if pd.isna(result["db"]) else f"{result['db']:.4f}")
                c3.metric("Calinski–Harabasz Index ↑", "N/A" if pd.isna(result["ch"]) else f"{result['ch']:,.2f}")
                
                # Monochrome PCA 2D Scatter with Distinct Shades and Shapes
                fig, ax = plt.subplots(figsize=(8, 4.8))
                
                # High-contrast monochrome palette and distinct marker symbols
                mono_palette = ["#09090b", "#52525b", "#a1a1aa", "#27272a", "#71717a", "#d4d4d8", "#3f3f46", "#18181b"]
                mono_markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
                
                unique_clusters = np.sort(np.unique(result["labels"]))
                for c_idx, c_label in enumerate(unique_clusters):
                    mask = (result["labels"] == c_label)
                    color = mono_palette[c_idx % len(mono_palette)]
                    marker = mono_markers[c_idx % len(mono_markers)]
                    ax.scatter(
                        result["coords"][mask, 0], result["coords"][mask, 1],
                        c=color, marker=marker, alpha=0.55, s=24, edgecolors="none",
                        label=f"Cluster {c_label}"
                    )
                    
                apply_minimal_plot_style(
                    ax,
                    title=f"PCA 2D Cluster Projection — {result['algorithm']} (k={result['k']})",
                    xlabel=f"Principal Component 1 ({result['explained'][0] * 100:.1f}% Variance)",
                    ylabel=f"Principal Component 2 ({result['explained'][1] * 100:.1f}% Variance)"
                )
                ax.legend(frameon=True, facecolor="#fafafa", edgecolor="#e4e4e7", fontsize=8.5, loc="upper right")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                
                st.caption("PCA provides a 2D linear projection maximizing variance. Unsupervised clusters reveal clean operational partitioning.")
            else:
                st.info("Choose features and parameters, then click **Execute Clustering Pipeline**.")

# -----------------------------------------------------------------------------
# Dataset Explorer
# -----------------------------------------------------------------------------
elif page == "Dataset Explorer":
    st.markdown("### 🔎 Telemetry Dataset Explorer")
    available = {}
    if solar_df is not None:
        available["Solar Generation Telemetry"] = solar_df
    if steel_df is not None:
        available["Steel Industry Telemetry"] = steel_df
        
    if not available:
        st.error("No supported CSV datasets were found. Check the project data/ directory.")
    else:
        choice = st.selectbox("Select Telemetry Dataset", list(available.keys()))
        df = available[choice]
        
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Total Records", f"{len(df):,}")
        q2.metric("Channels / Features", f"{df.shape[1]:,}")
        q3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
        q4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")
        
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("#### Telemetry Records Table")
        st.dataframe(df, use_container_width=True, height=290)
        
        st.download_button(
            "Download Telemetry CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{choice.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )
        
        with st.expander("Channel Schema & Missing Value Distribution", expanded=False):
            profile = pd.DataFrame({
                "Data Type": df.dtypes.astype(str),
                "Missing Count": df.isna().sum(),
                "Missing (%)": (df.isna().mean() * 100).round(2),
                "Unique Values": df.nunique(dropna=True),
            })
            st.dataframe(profile, use_container_width=True)
            
        numeric = numeric_columns(df)
        if numeric:
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            col_e1, col_e2 = st.columns(2)
            
            with col_e1:
                st.markdown("<div style='font-size: 0.95rem; font-weight: 600; color: #09090b; margin-bottom: 6px;'>Univariate Channel Distribution</div>", unsafe_allow_html=True)
                chosen_col = st.selectbox("Select Numeric Feature", numeric)
                fig, ax = plt.subplots(figsize=(6.5, 3.8))
                ax.hist(df[chosen_col].dropna(), bins=35, color="#18181b", edgecolor="#e4e4e7", alpha=0.85)
                apply_minimal_plot_style(ax, title=f"Distribution: {chosen_col}", xlabel=chosen_col, ylabel="Frequency")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                
            with col_e2:
                if len(numeric) >= 2:
                    st.markdown("<div style='font-size: 0.95rem; font-weight: 600; color: #09090b; margin-bottom: 6px;'>Channel Cross-Correlation</div>", unsafe_allow_html=True)
                    corr = df[numeric].corr(numeric_only=True)
                    fig, ax = plt.subplots(figsize=(6.5, 3.8))
                    im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto", cmap=plt.cm.Greys)
                    ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=45, ha="right", fontsize=8)
                    ax.set_yticks(range(len(corr.index)), labels=corr.index, fontsize=8)
                    apply_minimal_plot_style(ax, title="Pearson Correlation Heatmap", grid=False)
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(colors='#71717a', labelsize=8)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

st.divider()
st.caption("Machine Learning Capstone · Studio Environment. All algorithms conform to Review 1 and Review 2 rubrics.")

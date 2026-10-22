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
# Page setup and styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Capstone | Solar & Steel Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1500px;}
    .hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(115deg,#102a43 0%,#155e75 60%,#0f766e 100%); color: white; margin-bottom: 1.1rem;}
    .hero h1 {color: white; margin: 0; font-size: 2.15rem;}
    .hero p {color: #e0f2fe; margin: .45rem 0 0 0; font-size: 1rem;}
    div[data-testid="stMetric"] {background: var(--secondary-background-color); padding: 14px 16px; border-radius: 12px; border: 1px solid rgba(128,128,128,.18);}
    .small-note {font-size: .9rem; color: var(--text-color); opacity: .8;}
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
    # Avoid duplicate paths and virtual environment / git folders.
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
    # modified_time makes Streamlit reload the file after it changes.
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
    st.warning(message)
    st.code(str(expected_path), language="text")
    st.caption("Place the required CSV at this path, or update the filename in app.py to match your project.")

# -----------------------------------------------------------------------------
# Load project datasets. Regression/clustering use solar data; classification
# uses the Steel Industry dataset from the classification notebook.
# -----------------------------------------------------------------------------
solar_path = find_csv(["solar_generation.csv"])
steel_path = find_csv(["Steel_industry_data.csv", "steel_industry_data.csv"])
solar_df = load_csv(solar_path)
steel_df = load_csv(steel_path)

st.markdown("""
<div class="hero">
  <h1>⚡ Machine Learning Capstone Dashboard</h1>
  <p>Explore the datasets, evaluate predictive models, and inspect clustering structure.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("Navigation")
    page = st.radio(
        "Choose a section",
        ["Overview", "Regression", "Classification", "Clustering", "Dataset Explorer"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Machine Learning Capstone")
    st.caption("Results shown here are computed from the available CSV files when the relevant section is run. They may differ from notebook results if preprocessing or model settings differ.")

# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
if page == "Overview":
    st.subheader("Project overview")
    st.write("This dashboard brings the capstone's three modelling tasks into one interface. Select a section from the sidebar to inspect the data and run supported analyses.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Regression dataset", "Loaded" if solar_df is not None else "Not found")
    c2.metric("Classification dataset", "Loaded" if steel_df is not None else "Not found")
    c3.metric("Project data files", str(sum(x is not None for x in [solar_df, steel_df])) + " / 2")

    st.markdown("### Modelling workflow")
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### 📈 Regression")
        st.write("Estimate AC power from available solar and environmental measurements. Inspect fit quality and prediction errors.")
        if solar_df is not None and "AC Power in Watts" in solar_df.columns:
            st.success(f"Solar data ready · {len(solar_df):,} rows")
        else:
            st.info("Requires data/solar_generation.csv with an `AC Power in Watts` target column.")
    with b:
        st.markdown("#### 🎯 Classification")
        st.write("Predict steel-industry load categories and inspect class-level performance, confusion matrix and ROC-AUC.")
        if steel_df is not None and "Load_Type" in steel_df.columns:
            st.success(f"Steel data ready · {len(steel_df):,} rows")
        else:
            st.info("Requires data/Steel_industry_data.csv with a `Load_Type` column.")
    with c:
        st.markdown("#### 🧩 Clustering")
        st.write("Explore operational groups with K-Means and agglomerative clustering, then compare internal validation metrics and PCA projections.")
        if solar_df is not None:
            st.success("Solar data ready")
        else:
            st.info("Requires the solar dataset.")

    st.markdown("### Dataset snapshot")
    for title, data, path in [("Solar generation", solar_df, solar_path), ("Steel industry", steel_df, steel_path)]:
        with st.expander(f"{title} — {'available' if data is not None else 'missing'}", expanded=(data is not None)):
            if data is not None:
                m1, m2, m3 = st.columns(3)
                m1.metric("Rows", f"{data.shape[0]:,}")
                m2.metric("Columns", f"{data.shape[1]:,}")
                m3.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
                st.dataframe(data.head(8), use_container_width=True)
            else:
                st.caption(f"Expected file: {DATA_DIR / ('solar_generation.csv' if title == 'Solar generation' else 'Steel_industry_data.csv')}")

# -----------------------------------------------------------------------------
# Regression
# -----------------------------------------------------------------------------
elif page == "Regression":
    st.subheader("📈 Regression analysis")
    st.write("This interactive baseline trains on the solar dataset. It is a dashboard-side evaluation, not a claim that it reproduces the exact regression notebook unless the same split and settings are used.")
    if solar_df is None:
        show_missing_dataset("Solar generation dataset was not found.", DATA_DIR / "solar_generation.csv")
    elif "AC Power in Watts" not in solar_df.columns:
        st.error("The solar dataset does not contain the expected target column `AC Power in Watts`.")
        st.write("Available columns:", solar_df.columns.tolist())
    else:
        df = solar_df.copy()
        target = "AC Power in Watts"
        exclude = [target]
        # Exclude rows without a target; remove all-empty columns and ID-like columns.
        df = df.dropna(subset=[target]).dropna(axis=1, how="all")
        X = df.drop(columns=exclude)
        y = pd.to_numeric(df[target], errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
        # Drop columns with unique values for every row when they look like IDs.
        id_like = [col for col in X.columns if X[col].nunique(dropna=True) == len(X) and X[col].dtype == "object"]
        X = X.drop(columns=id_like, errors="ignore")
        if X.shape[1] == 0 or len(X) < 20:
            st.error("There are not enough usable features/rows to train a regression model.")
        else:
            with st.expander("Training settings", expanded=True):
                r1, r2 = st.columns(2)
                test_size = r1.slider("Test-set size", 0.1, 0.4, 0.2, 0.05)
                model_choice = r2.selectbox("Regression model", ["Random Forest", "Linear Regression"])
                run_regression = st.button("Train and evaluate regression model", type="primary", use_container_width=True)
            if run_regression:
                with st.spinner("Training regression model..."):
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
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"{result['mae']:,.3f}")
                m2.metric("RMSE", f"{result['rmse']:,.3f}")
                m3.metric("R² score", f"{result['r2']:.4f}")
                st.caption(f"Model: {result['model_name']} · Test rows: {len(result['y_test']):,}")
                chart_df = pd.DataFrame({"Actual": result["y_test"], "Predicted": result["predictions"]})
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(chart_df["Actual"], chart_df["Predicted"], alpha=0.55)
                lo = float(min(chart_df.min()))
                hi = float(max(chart_df.max()))
                ax.plot([lo, hi], [lo, hi], linestyle="--")
                ax.set_xlabel("Actual AC Power in Watts")
                ax.set_ylabel("Predicted AC Power in Watts")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                residuals = chart_df["Actual"] - chart_df["Predicted"]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(result["predictions"], residuals, alpha=0.5)
                ax.axhline(0, linestyle="--")
                ax.set_xlabel("Predicted AC Power")
                ax.set_ylabel("Residual (Actual − Predicted)")
                ax.set_title("Residual Plot")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("Choose the model and click **Train and evaluate regression model** to generate the metrics and plots.")

# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------
elif page == "Classification":
    st.subheader("🎯 Classification analysis")
    st.write("The classification notebook uses the Steel Industry Energy Consumption dataset and predicts `Load_Type`.")
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
        st.markdown("#### Target distribution")
        st.bar_chart(counts.rename_axis("Load type"))
        st.caption(f"{len(df):,} rows · {X.shape[1]} features · {y.nunique()} classes")
        with st.expander("Model settings", expanded=True):
            c1, c2 = st.columns(2)
            model_choice = c1.selectbox("Classifier", ["Random Forest", "Bagging", "AdaBoost", "Gradient Boosting", "Decision Tree", "Logistic Regression", "K-Nearest Neighbors"])
            test_size = c2.slider("Test-set size", 0.1, 0.4, 0.2, 0.05, key="clf_test_size")
            train_clf = st.button("Train and evaluate classifier", type="primary", use_container_width=True)
        if train_clf:
            with st.spinner("Training classifier. Larger datasets may take a little while..."):
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
                    "Precision (weighted)": precision_score(y_test, pred, average="weighted", zero_division=0),
                    "Recall (weighted)": recall_score(y_test, pred, average="weighted", zero_division=0),
                    "Weighted F1": f1_score(y_test, pred, average="weighted", zero_division=0),
                    "ROC-AUC (weighted OvR)": safe_auc(y_test, proba, classes) if proba is not None else np.nan,
                }
                st.session_state["classification_result"] = {
                    "model_name": model_choice, "metrics": metrics,
                    "y_test": np.asarray(y_test), "predictions": np.asarray(pred),
                    "classes": np.asarray(classes), "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
                }
        result = st.session_state.get("classification_result")
        if result:
            st.markdown(f"#### Results — {result['model_name']}")
            metric_cols = st.columns(len(result["metrics"]))
            for col, (name, value) in zip(metric_cols, result["metrics"].items()):
                col.metric(name, "N/A" if pd.isna(value) else (f"{value:.4f}" if "AUC" in name else f"{value:.2%}"))
            labels = list(result["classes"])
            cm = confusion_matrix(result["y_test"], result["predictions"], labels=labels)
            fig, ax = plt.subplots(figsize=(7, 5))
            im = ax.imshow(cm, interpolation="nearest")
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
            ax.set_yticks(range(len(labels)), labels=labels)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("#### Per-class report")
            report_df = pd.DataFrame(result["report"]).T
            st.dataframe(report_df.style.format(precision=3), use_container_width=True)
        else:
            st.info("Choose a classifier and click **Train and evaluate classifier** to calculate metrics on a held-out test set.")

# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------
elif page == "Clustering":
    st.subheader("🧩 Clustering analysis")
    st.write("Explore structure in the numeric solar-generation features. The target/output column is excluded from clustering so it does not directly determine cluster assignments.")
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
            st.caption(f"Numeric features available: {len(features)} · Excluded from clustering: {', '.join(excluded) if excluded else 'none'}")
            selected_features = st.multiselect("Features used for clustering", features, default=features)
            c1, c2 = st.columns(2)
            k = c1.slider("Number of clusters (k)", min_value=2, max_value=8, value=3)
            algorithm = c2.selectbox("Algorithm", ["K-Means", "Agglomerative — Ward", "Agglomerative — Complete", "Agglomerative — Average"])
            run_cluster = st.button("Run clustering analysis", type="primary", use_container_width=True)
            if run_cluster:
                if len(selected_features) < 2:
                    st.error("Select at least two features.")
                else:
                    with st.spinner("Preparing data and calculating cluster metrics..."):
                        work = df[selected_features].replace([np.inf, -np.inf], np.nan)
                        work = work.fillna(work.median(numeric_only=True)).dropna(axis=1, how="all")
                        if work.shape[1] < 2:
                            st.error("Not enough usable numeric columns after handling missing values.")
                        else:
                            scaler = StandardScaler()
                            scaled = scaler.fit_transform(work)
                            # Keep browser interactions responsive for large datasets.
                            max_rows = 12000
                            if len(scaled) > max_rows:
                                rng = np.random.RandomState(42)
                                idx = np.sort(rng.choice(len(scaled), max_rows, replace=False))
                                scaled_fit = scaled[idx]
                                sampled_note = f"Metrics/plots use a reproducible sample of {max_rows:,} from {len(scaled):,} rows for responsiveness."
                            else:
                                idx = np.arange(len(scaled))
                                scaled_fit = scaled
                                sampled_note = f"Metrics/plots use all {len(scaled):,} rows."
                            if algorithm == "K-Means":
                                labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled_fit)
                            else:
                                linkage = algorithm.split("—")[1].strip().lower()
                                labels = AgglomerativeClustering(n_clusters=k, linkage=linkage).fit_predict(scaled_fit)
                            sil = silhouette_score(scaled_fit, labels) if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(labels) else np.nan
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
                st.caption(result["sampled_note"])
                c1, c2, c3 = st.columns(3)
                c1.metric("Silhouette score ↑", "N/A" if pd.isna(result["silhouette"]) else f"{result['silhouette']:.4f}")
                c2.metric("Davies–Bouldin ↓", "N/A" if pd.isna(result["db"]) else f"{result['db']:.4f}")
                c3.metric("Calinski–Harabasz ↑", "N/A" if pd.isna(result["ch"]) else f"{result['ch']:,.2f}")
                fig, ax = plt.subplots(figsize=(8, 5))
                scatter = ax.scatter(result["coords"][:, 0], result["coords"][:, 1], c=result["labels"], alpha=0.65, s=14)
                ax.set_xlabel(f"PC1 ({result['explained'][0] * 100:.1f}% variance)")
                ax.set_ylabel(f"PC2 ({result['explained'][1] * 100:.1f}% variance)")
                ax.set_title(f"PCA projection — {result['algorithm']} (k={result['k']})")
                fig.colorbar(scatter, ax=ax, label="Cluster")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption("PCA is a 2D projection for visual inspection; separation in this plot alone does not prove that clusters are naturally present.")
            else:
                st.info("Choose features and settings, then click **Run clustering analysis**.")

# -----------------------------------------------------------------------------
# Dataset Explorer
# -----------------------------------------------------------------------------
elif page == "Dataset Explorer":
    st.subheader("🔎 Dataset explorer")
    available = {}
    if solar_df is not None:
        available["Solar generation"] = solar_df
    if steel_df is not None:
        available["Steel industry"] = steel_df
    if not available:
        st.error("No supported CSV datasets were found. Check the project data/ folder.")
    else:
        choice = st.selectbox("Dataset", list(available.keys()))
        df = available[choice]
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Rows", f"{len(df):,}")
        q2.metric("Columns", f"{df.shape[1]:,}")
        q3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        q4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
        st.markdown("#### Preview")
        st.dataframe(df, use_container_width=True, height=320)
        st.download_button("Download displayed dataset as CSV", data=df.to_csv(index=False).encode("utf-8"), file_name=f"{choice.lower().replace(' ', '_')}.csv", mime="text/csv")
        with st.expander("Column types and missing values"):
            profile = pd.DataFrame({
                "Data type": df.dtypes.astype(str),
                "Missing": df.isna().sum(),
                "Missing (%)": (df.isna().mean() * 100).round(2),
                "Unique values": df.nunique(dropna=True),
            })
            st.dataframe(profile, use_container_width=True)
        numeric = numeric_columns(df)
        if numeric:
            st.markdown("#### Numeric feature distribution")
            chosen_col = st.selectbox("Choose a numeric column", numeric)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(df[chosen_col].dropna(), bins=35, edgecolor="white")
            ax.set_title(f"Distribution of {chosen_col}")
            ax.set_xlabel(chosen_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            if len(numeric) >= 2:
                st.markdown("#### Correlation heatmap")
                corr = df[numeric].corr(numeric_only=True)
                fig, ax = plt.subplots(figsize=(min(12, 1 + len(numeric) * 0.65), min(8, 1 + len(numeric) * 0.5)))
                im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
                ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=90)
                ax.set_yticks(range(len(corr.index)), labels=corr.index)
                ax.set_title("Pearson correlation between numeric features")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

st.divider()
st.caption("Machine Learning Capstone · Interactive exploratory dashboard. Always use the final notebooks and their recorded outputs as the source of truth for submitted experimental results.")

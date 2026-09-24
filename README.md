# Machine Learning Capstone

This repository contains the Machine Learning Capstone project, structured into two core tracks: **Regression** (Solar Power Generation) and **Classification** (Steel Industry Energy Consumption).

## 📌 Review 1 Deliverables
Both tracks have been fully implemented up to the requirements for **Review 1**, including robust Exploratory Data Analysis (EDA), data preprocessing, feature engineering, and baseline model training.

### 1. Regression Track (Solar Power)
- **Dataset:** `solar_generation.csv`
- **Notebook:** `regression.ipynb`
- **Status:** Complete for Review 1 & 2.
- **Highlights:** 
  - Analyzed the impact of Irradiance, Ambient Temperature, and Module Temperature on AC Power output.
  - Engineered features like `Temp_Diff` and `Total_AC_Current`.
  - Trained and evaluated 10 regression algorithms (Linear, Ridge, Lasso, ElasticNet, KNN, LinearSVR, Decision Tree, Random Forest, Gradient Boosting, Extra Trees).
  - Implemented hyperparameter tuning and comprehensive visualizations (Residual Plots, Feature Importance).

### 2. Classification Track (Steel Industry)
- **Dataset:** `Steel_industry_data.csv`
- **Notebook:** `classification.ipynb`
- **Status:** Complete for Review 1 (Part A).
- **Highlights:**
  - Predicting categorical `Load_Type` (Light, Medium, Maximum).
  - Handled class imbalance, engineered `Power_Factor_Diff`, and encoded categorical temporal variables (`WeekStatus`, `Day_of_week`).
  - Trained the 5 required baseline classification algorithms (Logistic Regression, KNN, Naive Bayes, Decision Tree, SVC).
  - Evaluated using Accuracy, Weighted F1-scores, and Seaborn confusion matrices.

---

## 📁 Repository Structure

```text
/ (root)
├── README.md                     # Project overview, setup, and results
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
│
├── data/                         # Raw dataset files
│   ├── solar_generation.csv      # Solar dataset
│   └── Steel_industry_data.csv   # Steel industry dataset
│
├── notebooks/                    # Jupyter notebooks for analysis and modelling
│   ├── regression.ipynb          # 📈 Solar Regression Track
│   └── classification.ipynb      # 🏭 Steel Classification Track
│
├── models/                       # Saved model files (.pkl via joblib) - [Optional]
│   └── .gitkeep
│
├── app/                          # GUI / deployment code (if attempting bonus) - [Optional]
│   └── .gitkeep
│
└── archive/                      # Deprecated scratch files and early EDA notebooks
```

---

## ⚙️ Setup Instructions

To run this project locally and view the results, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Machine_Learning_Capstone
   ```

2. **Set up a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # venv\Scripts\activate   # On Windows
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   *Open `regression.ipynb` or `classification.ipynb`, click **Kernel**, and select **Restart & Run All** to execute the pipelines end-to-end.*

---

## ⚖️ Rubric Compliance
Both notebooks strictly adhere to the capstone guidelines:
- **Reproducibility:** `random_state=42` is used universally.
- **Data Leakage Prevention:** All scalers (`StandardScaler`) and encoders are fitted **strictly on the training set**.
- **Presentation:** All plots utilize `tight_layout()`, labeled axes, clear titles, and colorblind-friendly palettes (`Set2`).
- **Narrative:** Every major code block is accompanied by a Markdown cell explaining the underlying intuition and insights.
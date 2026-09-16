# Solar Power Generation: Machine Learning Capstone

This repository contains a Machine Learning Capstone project focused on analyzing and predicting solar power generation based on weather and environmental factors.

## Project Overview
The goal of this project is to explore how different features (like Irradiance, Ambient Temperature, Module Temperature, and Wind Speed) impact the AC Power output of solar panels. Currently, the project is in the Exploratory Data Analysis (EDA) phase.

## Repository Structure
- `solar_generation.csv`: The dataset containing solar power generation and weather metrics.
- `solar_eda_analysis.ipynb`: A comprehensive Jupyter Notebook containing the Exploratory Data Analysis. This includes data quality checks, outlier detection, distribution plotting, and feature engineering.
- `requirements.txt`: Python package dependencies required to run the notebooks.

## Setup Instructions
To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Machine_Learning_Capstone
   ```

2. **Set up a Virtual Environment (Optional but recommended):**
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
   Open `solar_eda_analysis.ipynb` to view the analysis.

## Key Findings (EDA Phase)
- **Data Quality:** The data requires filtering out nighttime hours (where Irradiance is near zero) for more accurate modelling of power generation.
- **Correlations:** Irradiance is the strongest predictor of AC Power.
- **Feature Engineering:** We engineered new metrics such as `Panel_Efficiency` and `Temp_Diff` (Module Temp - Ambient Temp) to capture non-linear relationships.
- **Temperature Effects:** High module temperatures can reduce the efficiency of the solar panels even when irradiance is high.
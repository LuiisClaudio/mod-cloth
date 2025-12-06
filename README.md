[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/E3bJQmxL)
# Integrated CA2: Data Visualization Techniques and Machine Learning Basics

This repository contains the coursework for Integrated CA2, completed by Luis Martins. The main file, `LuisMartins_CA2.ipynb`, demonstrates data visualization techniques and basic machine learning workflows using Python.

## Contents

- `LuisMartins_CA2.ipynb`: Jupyter Notebook with all code, visualizations, and analysis.
- **dashboard**: https://modclothdashboard.streamlit.app/

---

## 📊 Dashboard Features

- **KPI Cards**: Quick stats on reviews, products, users, and ratings.
- **Statistical Summary**: Data health check with summary tables.
- **Correlation Heatmap**: Visualize relationships between features.
- **Fit & Sizing**: Analyze fit distribution, body measurements, and sizing consistency.
- **Body Measurements**: Explore bra/shoe size distributions, body shape clustering, and more.
- **Ratings & Reviews**: Rating distributions, review length analysis, sentiment vs. rating.
- **Product & Category**: Category breakdowns, popularity analysis, treemaps.
- **Advanced Analytics**: Parallel categories, voice of customer, and more.

---

## Features

- Data loading and preprocessing
- Exploratory Data Analysis (EDA) with visualizations
- Implementation of basic machine learning models
- Evaluation and discussion of results

## Getting Started

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/your-username/integrated-ca2-dvt-and-mlb-LuiisClaudio.git
   ```

2. **Install dependencies:**
   - Make sure you have Python 3.x installed.
   - Install required packages (e.g., pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter):
     ```powershell
     pip install -r requirements.txt
     ```
     *(If `requirements.txt` is not present, install packages manually.)*

3. **Run the notebook:**
   ```powershell
   jupyter notebook LuisMartins_CA2.ipynb
   ```
### Prepare the dataset

- Place the ModCloth dataset at: `dataset/modcloth_final_data/modcloth_final_data.json`
- The dashboard expects the JSON Lines format.

### Run the dashboard

```powershell
streamlit run dashboard.py
```

---

## Usage

Open the notebook and run the cells sequentially to reproduce the analysis and results.

## 📁 Files

- `LuisMartins_CA2.ipynb` — Coursework notebook
- `dashboard.py` — Streamlit dashboard app
- `plotFunctions.py and LuisMartins_CA2.ipynb` — Custom plotting functions (required for dashboard)
- `dataset/modcloth_final_data/modcloth_final_data.json` — Data file (not included)

---

## 👤 Author

- **Luis Martins**

---

## 📄 License

For educational use only.

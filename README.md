# 👗 ModCloth Retail Analytics Dashboard


[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://modclothdashboard.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-green)](https://plotly.com/)
[![Status](https://img.shields.io/badge/Status-Active-success)]()

> **"Turning customer feedback into fit & sizing strategy."**

## 📖 Project Overview

In the e-commerce fashion industry, **fit and sizing issues** are the primary drivers of returns, costing retailers billions annually. This project is a comprehensive **Data Analytics Dashboard** built to analyze customer transactions and reviews from **ModCloth**. 

The goal is to provide **actionable intelligence** to product designers, merchandisers, and inventory managers. By analyzing over **[Total Rows]** data points, this dashboard answers critical business questions:
- *Are our sizing charts accurate across all categories?*
- *Which body shapes are we failing to serve?*
- *How does sentiment correlate with specific fit problems?*

## 🚀 Key Features & Business Insights

The dashboard is structured into 6 strategic sections, transforming raw data into decision-making tools:

### 1. 📊 Executive Overview
- **KPI Cards**: Instant view of Total Reviews, Unique Products, User count, and Avg Ratings.
- **Business Value**: "Pulse check" for stakeholders to gauge overall platform health in seconds.

### 2. 👗 Fit & Sizing Intelligence
- **Fit Distribution**: Identifies categories with high "Small" or "Large" feedback rates.
- **Actionable Insight**: "Swimwear runs small? Resize the next batch."
- **Length vs. Height**: Visualizes if "standard" lengths work for petite/tall customers.

### 3. 📏 Body Measurement Analysis
- **Clustering**: Categorizes customers into body shapes (Pear, Hourglass, Rectangle) using anthropometric ratios.
- **Strategic Value**: Move beyond "Size 10" to designing for "Size 10 Curvy" vs "Size 10 Straight".

### 4. ⭐ Voice of the Customer
- **Sentiment & Review Analysis**: Uses Text Analysis techniques to extract common adjectives from reviews.
- **Insight**: If "tight" appears with negative sentiment in "Dresses", it confirms a sizing defect, not a style preference.

### 5. 🏷️ Portfolio Performance
- **Pareto Analysis (80/20 Rule)**: Identifies the "Head" (bestsellers) vs. "Tail" products.
- **Quality/Popularity Matrix**: A BCG-style matrix to decide whether to *Invest, Maintain, or Divest* products based on rating and volume.

---

## 🛠️ Tech Stack & Methodology

**Languages & Frameworks:**
- **Python**: Core logic and data processing.
- **Streamlit**: Interactive web application framework.
- **Pandas**: Advanced data manipulation and cleaning (Regex for text standardization, imputation strategies for missing metrics).
- **Plotly Express/Graph Objects**: Interactive, high-performance visualizations.
- **Scikit-Learn / Stats**: Correlation analysis and statistical profiling.
- **TextBlob**: Sentiment polarity analysis (Positivity/Negativity scoring).
- **Text Analysis**: Custom keyword extraction algorithms for Voice of Customer insights.

**Data Pipeline:**
The `clean_datset()` function demonstrates robust data engineering:
1.  **Standardization**: Converts column names to snake_case.
2.  **Imputation**: Uses median strategies for critical measurements (waist, hips) to preserve distribution integrity.
3.  **Parsing**: Extracts complex string data (e.g., bra sizes) into analyzable components.
4.  **Segmentation**: Implements logic-based clustering to categorize customers into body shapes (Pear, Hourglass, Rectangle) based on WHR (Waist-to-Hip Ratio).

---

## 💻 Installation & Usage

1.  **Clone the repository**
    ```bash
    git clone https://github.com/LuiisClaudio/Mod_Cloth_Dashboard.git
    cd Mod_Cloth_Dashboard
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Dashboard**
    ```bash
    streamlit run streamlit_app.py
    ```

---

## 📈 Visual Preview

*(Placeholder for Screenshots - Recommended: meaningful captions for each)*

| **Fit Analysis** | **Body Shape Clustering** |
|:---:|:---:|
| *Identifying categories with high return risks* | *Segmenting customers by real body observations* |

---

## 👨‍💻 About the Analyst

**Luis Claudio**  
*Data Analyst*

I specialize in building tools that make data accessible and actionable. This project demonstrates my ability not just to code, but to **think like a product manager**—focusing on the "So What?" behind every number.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/your-profile) 

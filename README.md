# 🏠 Airbnb Listings EDA & Data Visualization - New York 2024

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)](https://jupyter.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green?style=flat&logo=pandas)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet?style=flat&logo=python)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Viz-blue?style=flat)](https://seaborn.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Comprehensive Exploratory Data Analysis of 20,765+ Airbnb listings in New York City (2024)** | Uncover patterns in pricing, availability, and host behavior using Python, Pandas, Matplotlib, and Seaborn.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Analysis Structure](#analysis-structure)
- [Key Findings](#key-findings)
- [How to Run](#how-to-run)
- [Recommendations](#recommendations)
- [Future Work](#future-work)
- [Contact](#contact)

---

## 🎯 Overview

This project performs a **comprehensive Exploratory Data Analysis (EDA)** on New York City Airbnb listing data to uncover meaningful patterns, trends, and insights. Using modern data visualization and statistical techniques, we analyze pricing dynamics, geographic distribution, and host behavior across NYC's five boroughs.

**Airbnb** is the world's leading short-term accommodation platform. This analysis provides **actionable insights** for both guests seeking value and hosts optimizing their listings.

---

## 🎪 Objectives

1. ✅ **Analyze pricing patterns** across room types and geographic locations
2. ✅ **Explore annual availability** and its relationship with listing characteristics
3. ✅ **Identify host behavior** through multi-listing analysis
4. ✅ **Detect and filter outliers** for precise statistical analysis
5. ✅ **Visualize geographic distributions** of listings by borough and room type
6. ✅ **Determine correlations** between key numerical variables
7. ✅ **Generate actionable recommendations** based on data insights

---

## 📊 Dataset

**20,765 Airbnb listings** across New York City with **22 features**:

| Column | Description | Type |
|--------|-------------|------|
| `id` | Unique listing identifier | Integer |
| `name` | Listing title | String |
| `host_id` | Unique host identifier | Integer |
| `host_name` | Host name | String |
| `neighbourhood_group` | Borough (Manhattan, Brooklyn, Queens, Bronx, Staten Island) | Categorical |
| `neighbourhood` | Specific neighborhood | String |
| `latitude` | Latitude coordinate | Float |
| `longitude` | Longitude coordinate | Float |
| `room_type` | Type of accommodation | Categorical |
| `price` | Nightly price (USD) | Integer |
| `minimum_nights` | Minimum night requirement | Integer |
| `number_of_reviews` | Total reviews received | Integer |
| `reviews_per_month` | Average monthly reviews | Float |
| `availability_365` | Days available per year | Integer |
| `beds` | Number of beds | Integer |
| **And more...** | | |

**Source**: Kaggle - NYC Airbnb Open Data (January 2024)

---

## 💻 Tech Stack

```python
# Data Processing & Analysis
pandas                    # Data manipulation and analysis
numpy                     # Numerical computing

# Data Visualization
matplotlib                # Static visualizations
seaborn                   # Statistical visualization library

# Environment
jupyter notebook          # Interactive notebook environment
python 3.8+              # Programming language
google colab             # Cloud-based notebook (optional)
```

---

## 📈 Analysis Structure

### 1️⃣ Data Cleaning & Preparation

**Key Operations:**
- Convert data types (id → object, price → numeric)
- Handle missing values (identification and strategy)
- Detect outliers (price > $1,500 filtered)
- Data validation and quality checks

**Result**: Cleaned dataset ready for analysis

---

### 2️⃣ Exploratory Data Analysis

#### **A. Price Distribution Analysis**

**Findings:**
- **Heavily right-skewed distribution**
- Majority of listings: $50-$300 per night
- **Outliers detected**: Listings > $1,500 (< 1% of data)
- **Before filtering**: Range 0-$100,000+ (highly distorted)
- **After filtering**: Range 0-$1,500 (interpretable)

**Insight**: Price distribution follows typical real estate market pattern with few luxury outliers.

---

#### **B. Availability Distribution (availability_365)**

**Key Pattern: Bimodal Distribution**

| Availability | Frequency | Interpretation |
|--------------|-----------|-----------------|
| **0 days** | High peak | Long-term rentals/blocked |
| **90-180 days** | Moderate | Seasonal availability |
| **365 days** | Highest peak | Year-round active |

**Insight**: Hosts adopt two primary strategies—either long-term rentals (blocked) or full-time short-term rentals.

---

#### **C. Borough Analysis (Neighbourhood Group)**

**Average Pricing by Borough:**

| Borough | Avg Price | Price/Bed | Market Position |
|---------|-----------|-----------|-----------------|
| 🟠 **Manhattan** | $204.15 | $138.71 | Premium |
| 🔵 **Brooklyn** | $155.14 | $99.79 | Mid-range |
| 🟢 **Queens** | ~$130 | ~$85 | Budget-friendly |
| 🔴 **Bronx** | $107.99 | ~$70 | Most affordable |
| 🟣 **Staten Island** | ~$115 | $67.73 | Lowest cost |

**Insight**: Clear price stratification—Manhattan commands 90% premium over Bronx; Staten Island offers budget alternatives.

---

#### **D. Geographic Distribution - Room Type**

**Spatial Patterns:**
- 🟠 **Entire homes/apt** (Orange): Concentrated in Manhattan; sparse in outer boroughs
- 🔵 **Private rooms** (Blue): Evenly distributed across all areas; high in Brooklyn/Queens
- 🟢 **Hotel rooms** (Green): Negligible presence; isolated points
- 🔴 **Shared rooms** (Red): Marginal; very few listings

**Insight**: Supply follows demand—expensive room types in expensive boroughs.

---

#### **E. Geographic Distribution - Borough**

**Density & Coverage:**
- 🟠 **Manhattan**: Ultra-dense cluster; central concentration
- 🔵 **Brooklyn**: High density; larger geographic footprint
- 🟢 **Queens**: Moderate density; widest geographic coverage
- 🟣 **Staten Island**: Sparse; isolated southern area
- 🔴 **Bronx**: Low-moderate density; northern concentration

**Insight**: Geographic supply mirrors population density and transportation accessibility.

---

#### **F. Price vs. Review Count Relationship**

**Critical Finding: Inverse Correlation**

- **Expensive listings** ($800-$1,500): Few reviews (<250)
  - Located primarily in Manhattan/Brooklyn
  - Target luxury/niche market
  
- **Budget listings** ($50-$200): Many reviews (>500)
  - Higher booking frequency
  - Indicates strong demand and guest satisfaction
  
- **Interpretation**: Lower prices attract higher volume → more reviews → higher visibility

**Insight**: Price and review volume are inversely related; lower-priced listings show greater market traction.

---

#### **G. Correlation Matrix Analysis**

**Significant Correlations:**

| Variables | Coefficient | Strength | Meaning |
|-----------|------------|----------|---------|
| `number_of_reviews` ↔ `reviews_per_month` | **+0.63** | Strong | Expected relationship |
| `price` ↔ `beds` | **+0.42** | Moderate | Key price predictor |
| `price` ↔ `longitude` | **-0.19** | Weak | Westward = more expensive |
| Most other pairs | ≈ 0 | Negligible | Independence |

**Insight**: Number of beds is the strongest numerical predictor of price; most variables operate independently.

---

### 3️⃣ Multivariate Analysis

**Pair Plot Patterns:**

1. **Marginal Distributions**:
   - Most variables: right-skewed
   - `availability_365`: bimodal (0 and 365)
   - `price`: highly concentrated in $50-$300 range

2. **Room Type Patterns**:
   - Entire homes systematically higher-priced
   - Clear separation between categories
   - Price differential 2-3x across types

3. **Outlier Clustering**:
   - Extreme values rare but consistently low-priced
   - High minimum night requirements cluster at low prices
   - Indicates niche long-term strategies

---

## 🔍 Key Findings

### 💰 Pricing Insights

1. **Borough Premium Effect**: Manhattan commands ~90% price premium over the Bronx
2. **Room Type Impact**: Entire homes 2-3x more expensive than private rooms
3. **Price Non-Linearity**: Price not random—strong location and type influence
4. **Outlier Rarity**: Listings >$1,500 are <1% of market (removed from analysis)

### 📍 Geographic Insights

1. **Dual-Core Distribution**: Manhattan and Brooklyn dominate (60%+ of listings)
2. **Value Alternatives**: Staten Island and Bronx offer 30-40% discounts
3. **Type-Location Coupling**: Luxury types cluster in premium zones
4. **Accessibility Correlation**: Supply concentrates near transit hubs

### 📊 Availability & Reviews

1. **Binary Strategy**: Hosts choose either blocked (0 days) or full availability (365 days)
2. **Occupancy Signal**: Low prices → many reviews → high booking frequency
3. **Professional Hosting**: Multi-property hosts show sophisticated pricing strategies
4. **Trust Proxy**: Review count signals quality and guest confidence

### 💡 Market Dynamics

1. **Segmented Markets**: Three distinct pricing tiers (luxury, mid-range, budget)
2. **Sustainable Arbitrage**: Price differences by borough are stable and predictable
3. **Demand Stability**: Consistent bimodal availability indicates reliable year-round demand
4. **Reputation Economics**: Reviews function as valuable asset for host credibility

---

## 🚀 How to Run

### Prerequisites

```bash
# Python 3.8+
python --version

# pip package manager
pip --version
```

### Local Setup

```bash
# 1. Clone repository
git clone https://github.com/davidstocco2024-cell/Python-Project-New-York-AirBnb-Listing-2024.git
cd Python-Project-New-York-AirBnb-Listing-2024

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install pandas numpy matplotlib seaborn jupyter

# 4. Launch Jupyter Notebook
jupyter notebook note.ipynb
```

### Cloud Alternative (Google Colab)

```python
# No installation required!
1. Go to: https://colab.research.google.com
2. File → Open Notebook → GitHub
3. Paste: https://github.com/davidstocco2024-cell/Python-Project-New-York-AirBnb-Listing-2024
4. Select note.ipynb
5. Run all cells!
```

---

## 💡 Recommendations

### 👥 For Guests

- 🔍 **Explore Brooklyn**: Best price-to-location ratio vs Manhattan
- ⭐ **Prioritize Reviews**: Higher review count signals reliability and quality
- 💰 **Consider Private Rooms**: 50% cheaper than entire homes; often equally comfortable
- 📅 **Check Year-Round Availability**: 365-day listings indicate active, responsive hosts
- 🗺️ **Venture to Outer Boroughs**: Queens and Bronx offer 30-40% savings vs Manhattan

### 🏠 For Hosts

- 📈 **Maximize Availability**: Listings with >300 available days generate highest income
- 💬 **Rapid Response**: Quick inquiry responses correlate with more bookings and reviews
- 🎯 **Strategic Positioning**: Manhattan = premium pricing; Other boroughs = volume strategy
- 🛏️ **Add Beds**: Each additional bed increases nightly price by ~$30-50
- 📊 **Competitive Benchmarking**: Monitor price/bed ratio in your specific neighborhood
- 👌 **Maintain Excellence**: High review volume indicates satisfied guests and strong market position

### 🎓 Market Insights

- **Price Arbitrage Opportunity**: Stable, predictable borough-based price differentials
- **Market Segmentation**: Three distinct, non-overlapping customer segments
- **Demand Stability**: Consistent demand year-round (evidenced by bimodal availability)
- **Reputation Capital**: Reviews are valuable, defensible asset for competitive advantage

---

## 🔮 Future Work

### Predictive Analytics
- [ ] **Price Prediction Model**: Regression (Linear, Ridge, Lasso, XGBoost)
- [ ] **Occupancy Forecasting**: Time series analysis (ARIMA, Prophet)
- [ ] **Demand Clustering**: K-means segmentation of listings by market type

### Advanced NLP & Analysis
- [ ] **Sentiment Analysis**: Parse review text to identify satisfaction drivers
- [ ] **Topic Modeling**: Discover common themes in guest feedback (LDA, BERTopic)
- [ ] **Review-Price Relationship**: Causal analysis of review sentiment impact on pricing power

### Interactive Visualizations
- [ ] **Streamlit Dashboard**: Real-time filters, dynamic charts, borough comparison tools
- [ ] **Folium Map**: Interactive geospatial visualization with hover details
- [ ] **Plotly Visualizations**: 3D price-availability-reviews scatter plots

### Scalability & Production
- [ ] **ETL Pipeline**: Automated data collection and refresh (Airflow, dbt)
- [ ] **REST API**: Expose analysis results via FastAPI/Flask
- [ ] **Database Integration**: PostgreSQL/MongoDB for persistent storage
- [ ] **Cloud Deployment**: Serverless analytics (AWS Lambda, Google Cloud Functions)

---

## 📁 Repository Structure

```
Python-Project-New-York-AirBnb-Listing-2024/
├── README.md                    # Project documentation
├── LICENSE                      # MIT License
├── .gitignore                   # Git configuration
│
├── note.ipynb                   # Main analysis notebook (Jupyter)
├── datasets.csv                 # Raw data (20,765 listings)
│
├── images/                      # Visualization outputs
│   ├── 01_price_distribution.png
│   ├── 02_availability_distribution.png
│   ├── 03_geographic_room_type.png
│   ├── 04_geographic_borough.png
│   ├── 05_price_vs_reviews.png
│   ├── 06_price_by_neighborhood.png
│   └── 07_correlation_matrix.png
│
└── .github/                     # GitHub configuration
    └── workflows/               # CI/CD pipelines (optional)
```

---

## 📊 Tools & Versions

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.8+ | Programming language |
| Pandas | 1.3+ | Data manipulation |
| NumPy | 1.21+ | Numerical computing |
| Matplotlib | 3.4+ | Static visualizations |
| Seaborn | 0.11+ | Statistical graphics |
| Jupyter | 1.0+ | Interactive notebooks |

---

## 📜 License

This project is licensed under the **MIT License**—see [LICENSE](LICENSE) file for details.

Free to use, modify, and distribute for personal and commercial purposes with proper attribution.

```
MIT License

Copyright (c) 2024 Ignacio David Stocco (davidstocco2024-cell)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the Software is furnished to do so...
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

### How to Contribute

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/YourFeature`
3. **Commit changes**: `git commit -m 'Add YourFeature'`
4. **Push to branch**: `git push origin feature/YourFeature`
5. **Open a Pull Request**

### Report Bugs

Found an issue? Please open a GitHub Issue with:
- Clear description of the problem
- Steps to reproduce
- Python version and library versions
- Expected vs actual behavior

---

## 💬 Contact & Connect

### 👨‍💻 Author

**Ignacio David Stocco** (Nacho)
- 📍 Las Heras, Mendoza, Argentina
- 💼 Full-Stack Developer | Data Analyst | Python Expert
- 🔧 Tech Stack: Python, Django, SQL Server, Tableau, VS Code

### 🔗 Get In Touch

- 🐙 **GitHub**: [@davidstocco2024-cell](https://github.com/davidstocco2024-cell)
- 💼 **LinkedIn**: [Connect on LinkedIn](https://linkedin.com/in/nacho-stocco)
- 📧 **Email**: [your.email@example.com](mailto:your.email@example.com)
- 🎥 **YouTube**: [Data Analytics Channel](https://youtube.com/@your-channel)

---

## 📚 References & Resources

- **Kaggle Dataset**: [NYC Airbnb Open Data](https://www.kaggle.com/datasets/vrindakallu/new-york-dataset)
- **Pandas Documentation**: [Official Docs](https://pandas.pydata.org/docs/)
- **Seaborn Tutorial**: [Statistical Data Visualization](https://seaborn.pydata.org/tutorial.html)
- **Matplotlib Guide**: [Visualization with Matplotlib](https://matplotlib.org/stable/users/index.html)
- **EDA Best Practices**: [Towards Data Science](https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15)

---

## 🙏 Acknowledgments

- **Kaggle** for hosting the Airbnb dataset
- **Open Source Community**: Pandas, Matplotlib, Seaborn developers
- **Inspiration**: [Reference Repository](https://github.com/najirh/Python-Project-P2-New-York-AirBnb-Listing-2024)
- **NYC Data**: 20,765 Airbnb listings analyzed

---

<div align="center">

### ⭐ If this project helped you, consider leaving a star! ⭐

Made with ❤️ by [Ignacio Stocco](https://github.com/davidstocco2024-cell)

![Last Updated](https://img.shields.io/badge/Last%20Updated-August%202024-blue)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-green)
![Data Points](https://img.shields.io/badge/Data%20Points-20765-orange)
![Analysis Status](https://img.shields.io/badge/Status-Completed-success)

---

**Questions?** Open an issue on GitHub or reach out via LinkedIn.

</div>

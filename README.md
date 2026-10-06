# kc-house-valuation-app

# 🏡 ValuEdge™ AI — King County Real Estate Valuation & Investment Intelligence

An end-to-end Automated Valuation Model (AVM) and interactive PropTech dashboard built with **Scikit-Learn**, **Streamlit**, **PyDeck**, and **Plotly**. 

The application utilizes an instance-based **K-Nearest Neighbors (KNN)** regression engine optimized with domain-specific feature weighting to estimate property appraisals, assess investment risk, and simulate remodeling returns across King County, WA.

---

## 🎯 Key Highlights

- **Custom-Weighted Machine Learning Pipeline:** Engineered a specialized `ColumnTransformer` with a $3.0\times$ geospatial coordinate multiplier and $2.0\times$ construction grade penalty, boosting $R^2$ from **0.8086 to 0.8582** and cutting Mean Absolute Error (MAE) by over **$10,500**.
- **Interactive 3D Geospatial Visuals:** Integrated **PyDeck** to render extruded 3D valuation columns and localized coordinate pins directly across Seattle, Bellevue, Redmond, and Renton micro-markets.
- **Dynamic CapEx / Renovation ROI Engine:** Built a "What-If" simulator that estimates property appreciation and calculates net ROI against user-defined remodeling budgets.
- **Institutional Confidence Bounds:** Employs empirical test MAPE ($\pm15\%$) to provide conservative floor and aggressive ceiling pricing for deal underwriting.
- **Client Dossier Export:** Supports single-click CSV export of full property specifications and appraisal estimates.

---

## 📊 Benchmark Results

| Model Architecture | $R^2$ Score | MAE | Strategy |
|---|:---:|:---:|---|
| **Baseline KNN ($K=7$, Uniform)** | `0.8086` | `$84,462.16` | Standard isotropic Euclidean scaling |
| **Feature-Weighted KNN ($K=7$, Distance)** | **`0.8582`** | **`$73,906.82`** | **$3\times$ Geo + $2\times$ Grade Weighting + Inverse Distance** |

---

## 💻 Tech Stack

- **Language:** Python 3.10+
- **Machine Learning:** Scikit-Learn, Joblib, NumPy, Pandas
- **Visualization:** PyDeck (3D maps), Plotly Graph Objects (Financial charts)
- **Deployment & UI:** Streamlit Cloud

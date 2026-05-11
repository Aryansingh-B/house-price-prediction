# 🏠 House Price Prediction — ML Web App

An end-to-end machine learning web application that predicts house prices using multiple regression algorithms, built with Python and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4%2B-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## What This Project Does

- Trains and compares **6 regression algorithms** on a housing dataset
- Ranks models by R² score so you always know which performs best
- Provides a **live price prediction form** with 12 house features
- Includes an **interactive data explorer** — distribution plots, scatter plots, and a correlation heatmap
- Saves trained models and evaluation metrics to disk for reuse

---

## App Tabs

| Tab | What it does |
|-----|-------------|
| **Predict** | Enter house features → get a predicted price instantly |
| **Train** | Choose an algorithm, train on your uploaded data |
| **Evaluate** | View MAE, RMSE, R² + actual vs predicted chart + residuals |
| **Compare** | Run all 6 algorithms at once and rank them |
| **Explore** | Statistical summary, missing values, distributions, heatmap |

---

## Algorithms Included

| Algorithm | Strengths |
|-----------|-----------|
| Linear Regression | Fast, interpretable baseline |
| Ridge Regression | L2 regularisation — good when features correlate |
| Lasso Regression | L1 regularisation — auto-removes weak features |
| Decision Tree | Easy to visualise, captures non-linearity |
| Random Forest | Ensemble — best balance of speed and accuracy |
| Gradient Boosting | Highest accuracy, sequential tree boosting |

---

## Project Structure

```
house-price-prediction/
│
├── app.py               # Streamlit web app (main entry point)
├── data_loader.py       # CSV loading utility
├── preprocess.py        # Feature selection and encoding
├── train.py             # Model training and saving
├── evaluate.py          # Metrics calculation and reporting
├── predict.py           # CLI-based prediction script
│
├── models/
│   └── model.pkl        # Saved trained model (auto-created after training)
│
├── reports/
│   └── metrics.json     # Saved evaluation metrics (auto-created after training)
│
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Aryansingh-B/house-price-prediction.git
cd House-price-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

### 4. Upload your dataset

Use the sidebar to upload a CSV file. The app expects these columns:

```
area, bedrooms, bathrooms, stories, parking,
mainroad, guestroom, basement, hotwaterheating,
airconditioning, prefarea, furnishingstatus, price
```

Binary columns (`mainroad`, `guestroom`, etc.) should have values `yes` / `no`.  
`furnishingstatus` should be `unfurnished`, `semi-furnished`, or `furnished`.

---

## Dataset

This project was built using the [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) from Kaggle.  
Download `Housing.csv` and upload it directly in the app sidebar.

---

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| **MAE** | Mean Absolute Error — average prediction error in price units |
| **RMSE** | Root Mean Squared Error — penalises large errors more heavily |
| **R²** | Coefficient of determination — how much variance the model explains (1.0 = perfect) |

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive web UI
- **Scikit-learn** — ML algorithms and metrics
- **Pandas / NumPy** — data manipulation
- **Plotly** — interactive charts

---

## Future Improvements

- [ ] Add XGBoost and LightGBM to the algorithm comparison
- [ ] Feature importance chart for tree-based models
- [ ] Cross-validation scores alongside train/test split metrics
- [ ] Export predictions as a downloadable CSV

---

## Author

**Your Name**  
[LinkedIn](www.linkedin.com/in/aryansinghbais8) · [GitHub](https://github.com/Aryansingh-B)

---

## License

MIT License — feel free to use and adapt for your own projects.

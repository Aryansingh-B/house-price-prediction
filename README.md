# 🏠 House Lens — ML Price Prediction App

An end-to-end Machine Learning web application that predicts house prices using multiple regression algorithms, built with Python and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4%2B-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> **📌 Note:** This app is currently built and optimised specifically for the House Price Prediction dataset. I'm actively working on v2 — a fully flexible ML platform that supports any regression or classification dataset. Stay tuned!

---

## 🔴 Live Demo

🔗 [house-price-prediction-zceizn5nygxynx4c3ij9ky.streamlit.app](https://house-price-prediction-zceizn5nygxynx4c3ij9ky.streamlit.app/)

---

## 📸 App Preview

### 🔮 Predict Price
Enter house features — area, bedrooms, bathrooms, amenities — and get an instant predicted price with an input summary chart.

![Predict Tab](screenshots/Predict_tab1.png)

---

### 🚀 Train Model
Upload your dataset, preview encoded data, choose an algorithm, and train with one click. Metrics appear immediately after training.

![Train Tab - Upload and Preview](screenshots/Train_tab1.png)

![Train Tab - Encoding Info](screenshots/Train_tab2.png)

![Train Tab - After Training](screenshots/Train_tab3.png)

---

### 📊 Metrics (Evaluate)
View MAE, MSE, RMSE, and R² score with an Actual vs Predicted scatter plot and Residuals Distribution histogram.

![Evaluate Tab - Metrics](screenshots/Evaluate_tab1.png)

![Evaluate Tab - Residuals](screenshots/Evaluate_tab2.png)

---

### ⚡ Compare Algorithms
Run all 6 algorithms at once and rank them by R² score. The best model is highlighted automatically.

![Compare Tab](screenshots/Compare_tab1.png)

---

### 🔍 Data Explorer
Interactive statistical summary, missing value checker, distribution plots, price vs feature scatter, and a correlation heatmap.

![Explore Tab - Stats and Distributions](screenshots/Explore_tab1.png)

![Explore Tab - Correlation Heatmap](screenshots/Explore_tab2.png)

---

## ✅ Features

- 🏆 **6 regression algorithms** — Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting
- 🥇 **Auto-ranks models** by R² score so you always know the best performer
- 💰 **Live price prediction** with 12 house features and an input summary chart
- 📈 **Interactive Plotly charts** — Actual vs Predicted, Residuals Distribution, Correlation Heatmap
- 🔍 **Data Explorer** — statistical summary, missing values, distribution plots, scatter plots
- 💾 **Model persistence** — saves trained model (.pkl) and metrics (.json) to disk

---

## 🧮 Algorithms Included

| Algorithm | Strengths |
|-----------|-----------|
| Linear Regression | Fast, interpretable baseline |
| Ridge Regression | L2 regularisation — handles correlated features |
| Lasso Regression | L1 regularisation — auto-removes weak features |
| Decision Tree | Captures non-linearity, easy to explain |
| Random Forest | Best balance of speed and accuracy |
| Gradient Boosting | Highest accuracy, sequential boosting |

---

## 📁 Project Structure

```
house-price-prediction/
│
├── screenshots/
│   ├── Predict_tab1.png
│   ├── Train_tab1.png
│   ├── Train_tab2.png
│   ├── Train_tab3.png
│   ├── Evaluate_tab1.png
│   ├── Evaluate_tab2.png
│   ├── Compare_tab1.png
│   ├── Explore_tab1.png
│   └── Explore_tab2.png
│
├── src/
│   ├── __init__.py          # Makes src a Python package
│   ├── data_loader.py       # CSV loading utility
│   ├── preprocess.py        # Feature selection and encoding
│   ├── train.py             # Model training and saving
│   ├── evaluate.py          # Metrics calculation and reporting
│   └── predict.py           # CLI-based prediction script
│
├── models/
│   └── model.pkl            # Auto-created after training
│
├── reports/
│   └── metrics.json         # Auto-created after training
│
├── app.py                   # Streamlit web app — main entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Opens in your browser at `http://localhost:8501`

### 4. Upload your dataset

Use the sidebar to upload `Housing.csv`. The app expects these columns:

```
area, bedrooms, bathrooms, stories, parking,
mainroad, guestroom, basement, hotwaterheating,
airconditioning, prefarea, furnishingstatus, price
```

> Download the dataset from [Kaggle — Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

---

## 📏 Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| **MAE** | Mean Absolute Error — average prediction error in price units |
| **MSE** | Mean Squared Error — penalises large errors more heavily |
| **RMSE** | Root Mean Squared Error — same unit as price, easier to interpret |
| **R²** | How much variance the model explains (1.0 = perfect fit) |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive web UI
- **Scikit-learn** — ML algorithms and metrics
- **Pandas / NumPy** — data manipulation
- **Plotly** — interactive charts

---

## 🗺️ Roadmap

- [x] 6 regression algorithms with comparison
- [x] Live prediction with input summary
- [x] Interactive data explorer
- [x] Model persistence
- [ ] XGBoost and LightGBM support
- [ ] Cross-validation scores
- [ ] Export predictions as CSV
- [ ] **v2: Support for any regression or classification dataset**

---

## 👨‍💻 Author

**Aryansingh**
Built during training at **Naresh IT Technologies**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/your-username)

---

## 📄 License

MIT License — feel free to use and adapt for your own projects.
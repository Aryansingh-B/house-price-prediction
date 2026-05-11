from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import os
from evaluate import evaluate_model, save_metrics

def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate
    mae, mse, r2 = evaluate_model(y_test, y_pred)
    save_metrics(mae, mse, r2)

    # Save model
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("\n✅ Model trained, evaluated & saved")
    return model
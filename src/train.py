from sklearn.linear_model import LinearRegression
import pickle

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)

    # Save model
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained & saved")
    return model
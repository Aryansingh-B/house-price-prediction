import pickle
import pandas as pd

def load_model():
    with open("models/model.pkl", "rb") as f:
        return pickle.load(f)

def predict_price():
    model = load_model()

    print("\nEnter house details:")

    area = float(input("Area: "))
    bedrooms = int(input("Bedrooms: "))
    bathrooms = int(input("Bathrooms: "))

    # Use DataFrame instead of numpy array ✅
    features = pd.DataFrame([[area, bedrooms, bathrooms]],
                            columns=['area', 'bedrooms', 'bathrooms'])

    prediction = model.predict(features)

    print(f"\n💰 Predicted Price: {prediction[0]:.2f}")
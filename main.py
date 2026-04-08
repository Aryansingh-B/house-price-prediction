from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train import train_model
from src.predict import predict_price

def main():
    print("🏠 House Price Prediction System")

    while True:
        print("\n1. Train Model")
        print("2. Predict Price")
        print("3. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            df = load_data("data/housing.csv")
            if df is not None:
                X, y = preprocess_data(df)
                train_model(X, y)

        elif choice == "2":
            predict_price()

        elif choice == "3":
            print("👋 Exiting...")
            break

        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
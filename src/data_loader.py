import pandas as pd

def load_data(path):
    try:
        df = pd.read_csv(r"A:\DataScience_AI\Datasets\Housing.csv")
        print("✅ Data loaded successfully")
        return df
    except Exception as e:
        print("❌ Error loading data:", e)
        return None
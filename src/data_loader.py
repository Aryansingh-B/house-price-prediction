import pandas as pd
 
def load_data(path):
    """
    Load a CSV dataset from the given file path.
 
    Args:
        path (str): Path to the CSV file.
 
    Returns:
        pd.DataFrame or None: Loaded dataframe, or None if loading fails.
    """
    try:
        df = pd.read_csv(path)
        print("✅ Data loaded successfully")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
 
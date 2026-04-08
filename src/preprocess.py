def preprocess_data(df):
    # Drop missing values
    df = df.dropna()

    # Select features
    X = df[['area', 'bedrooms', 'bathrooms']]
    y = df['price']

    return X, y
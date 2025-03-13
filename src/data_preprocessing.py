import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """Load the Pokemon dataset."""
    return pd.read_csv(file_path)


def preprocess_data(df):
    """Preprocess the data for modeling."""
    # Handle missing values
    df['type2'].fillna('None', inplace=True)
    df['height_m'].fillna(df['height_m'].median(), inplace=True)
    df['weight_kg'].fillna(df['weight_kg'].median(), inplace=True)
    df['percentage_male'].fillna(df['percentage_male'].median(), inplace=True)

    # Select features for the model
    features = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'base_total']
    X = df[features]
    y = df['is_legendary']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features
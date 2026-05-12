"""
Phase 1 - California Housing : pipeline de chargement et normalisation.

Choix de pipeline : (b) split puis scaler.fit(X_train).
Justification : fitter le scaler sur X entier ferait fuiter la distribution
du test set dans le preprocessing (data leakage) - les stats globales
"voient" déjà le test avant l'évaluation. La règle ML : le test set doit
rester invisible jusqu'à evaluate().
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "housing.csv")

FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


def load_california():
    """Charge California Housing depuis le CSV local, reproduit le format
    de sklearn.datasets.fetch_california_housing (8 features, cible /1e5)."""
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["total_bedrooms"]).copy()
    df["AveRooms"] = df["total_rooms"] / df["households"]
    df["AveBedrms"] = df["total_bedrooms"] / df["households"]
    df["AveOccup"] = df["population"] / df["households"]
    df = df.rename(columns={
        "median_income": "MedInc",
        "housing_median_age": "HouseAge",
        "population": "Population",
        "latitude": "Latitude",
        "longitude": "Longitude",
    })
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = (df["median_house_value"].to_numpy(dtype=np.float64)) / 100000.0
    return X, y, FEATURE_NAMES


def make_splits(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Split train/val/test : 64% / 16% / 20%."""
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X_train):
    """Fit StandardScaler sur X_train UNIQUEMENT (pas de leakage)."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def main():
    X, y, feature_names = load_california()
    print(f"Dataset : {X.shape[0]} exemples, {X.shape[1]} features")
    print(f"Feature names ({len(feature_names)}) : {list(feature_names)}")
    assert len(feature_names) == 8, "California Housing doit avoir 8 features"

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    scaler = fit_scaler(X_train)
    X_train_norm = scaler.transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)

    print(f"\nX_train shape : {X_train_norm.shape}")
    print(f"X_val   shape : {X_val_norm.shape}")
    print(f"X_test  shape : {X_test_norm.shape}")

    print("\nX_train_norm mean (par feature) :")
    print(np.round(X_train_norm.mean(axis=0), 4))
    print("X_train_norm std  (par feature) :")
    print(np.round(X_train_norm.std(axis=0), 4))

    # Adversarial : valeurs aberrantes
    X_extreme = np.array([[99999, -99999, 0, 0, 0, 0, 37.0, -120.0]])
    X_extreme_norm = scaler.transform(X_extreme)
    print(f"\nValeur extreme normalisee (MedInc=99999) : {X_extreme_norm[0, 0]:.2f}")
    print("=> Les outliers produisent des valeurs normalisees enormes ;")
    print("   en prod, clipper les inputs hors plage entrainement protege le modele.")

    return X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test, scaler


if __name__ == "__main__":
    main()

"""
Phase 2 - Baseline PMC regression sur California Housing.

Pourquoi pas de sigmoid sur la couche de sortie :
- La cible (prix median / 1e5) est continue, valeurs typiquement entre 0.15 et 5.0.
- sigmoid borne la sortie a (0, 1) : le modele ne pourra JAMAIS predire un prix > 1.0
  (= 100 000 $). La loss MSE descendrait quand meme (le modele saturerait juste a 1.0
  pour toutes les maisons cheres), mais le modele est inutilisable. Sortie linear obligatoire.
"""
from tensorflow import keras
from tensorflow.keras import layers

from phase1_pipeline_california import load_california, make_splits, fit_scaler


def build_regression_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),  # pas d'activation : regression
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main():
    X, y, _ = load_california()
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)
    scaler = fit_scaler(X_train)
    X_train_n = scaler.transform(X_train)
    X_val_n = scaler.transform(X_val)
    X_test_n = scaler.transform(X_test)

    model = build_regression_model(input_dim=X_train_n.shape[1])
    model.summary()

    history = model.fit(
        X_train_n, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_n, y_val),
        verbose=2,
    )

    test_loss, test_mae = model.evaluate(X_test_n, y_test, verbose=0)
    print(f"\nMSE test : {test_loss:.4f}")
    print(f"MAE test : {test_mae:.4f} (en centaines de milliers de $)")
    print(f"=> environ {test_mae * 100000:.0f} $ d'erreur moyenne")

    return model, history


if __name__ == "__main__":
    main()

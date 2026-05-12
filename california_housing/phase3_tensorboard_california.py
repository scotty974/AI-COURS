"""
Phase 3 - Diagnostic TensorBoard sur California Housing.

Compare deux runs : donnees normalisees vs donnees brutes.
Hypothese : run "norm" a train et val qui descendent ensemble (situation a),
           run "raw" voit la loss exploser ou stagner haut.

Lancer ensuite :
    tensorboard --logdir=california_housing/logs/fit
puis ouvrir http://localhost:6006 et cocher les deux runs dans l'onglet Scalars.
"""
import datetime
import os
from tensorflow import keras

from phase1_pipeline_california import load_california, make_splits, fit_scaler
from phase2_baseline_regression import build_regression_model

LOG_ROOT = os.path.join(os.path.dirname(__file__), "logs", "fit")


def train_with_tensorboard(X_train, y_train, X_val, y_val, run_name, epochs=100):
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    log_dir = os.path.join(LOG_ROOT, f"{run_name}_{timestamp}")
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = build_regression_model(input_dim=X_train.shape[1])
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[tb_callback],
        verbose=0,
    )
    final_val = history.history["val_loss"][-1]
    print(f"Run '{run_name}' termine. Logs dans {log_dir} | final val_loss={final_val:.4f}")
    return model, history


def main():
    X, y, _ = load_california()
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)
    scaler = fit_scaler(X_train)
    X_train_n = scaler.transform(X_train)
    X_val_n = scaler.transform(X_val)

    # Run 1 : normalise (comportement sain attendu)
    train_with_tensorboard(X_train_n, y_train, X_val_n, y_val,
                           run_name="california_norm")
    # Run 2 : brut (gradients desequilibres, convergence degradee)
    train_with_tensorboard(X_train, y_train, X_val, y_val,
                           run_name="california_raw")

    print("\nLancer : tensorboard --logdir=california_housing/logs/fit")


# Diagnostic attendu :
# - california_norm : situation (a), train_loss et val_loss descendent ensemble.
# - california_raw  : Latitude (~37), Longitude (~-120), Population (>1000) ont des
#   echelles tres differentes ; les gradients explosent, la loss reste tres haute
#   ou diverge. Aucun overfit "interessant", juste un modele qui n'apprend pas.

if __name__ == "__main__":
    main()

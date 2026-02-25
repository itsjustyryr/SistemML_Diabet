#!/usr/bin/env python3
"""
Diabetes Risk Prediction – Training Script
===========================================
Compares two algorithms (Neural Network vs Random Forest) using Stratified
K-Fold Cross-Validation, then trains and saves the best-performing model.

Features:
  - Stratified K-Fold Cross-Validation (default 5 folds)
  - Side-by-side algorithm comparison (Neural Network vs Random Forest)
  - Full classification reports and confusion matrices
  - Saves the winning model for inference with predict.py

Usage:
    python train_model.py
    python train_model.py --data dataset/combined_all_sources_5050.csv
    python train_model.py --folds 10 --epochs 150 --batch 64
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── CLI arguments ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train diabetes risk model")
parser.add_argument(
    "--data",
    default="dataset/diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
    help="Path to the CSV dataset",
)
parser.add_argument("--epochs", type=int, default=100, help="Training epochs (NN)")
parser.add_argument("--batch", type=int, default=128, help="Batch size (NN)")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (NN)")
parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
parser.add_argument(
    "--out_dir", default="saved_model", help="Directory to save trained artefacts"
)
args = parser.parse_args()


# ── Model builders ──────────────────────────────────────────────────────────
def build_nn(input_dim: int, lr: float) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEED,
    )


def score(y_true, y_prob):
    """Compute all metrics from true labels and predicted probabilities."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


# ── 1. Load & inspect data ─────────────────────────────────────────────────
print("=" * 60)
print("DIABETES RISK PREDICTION – TRAINING")
print("=" * 60)

print("\nLoading dataset …")
df = pd.read_csv(args.data)
print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
print(f"  Class distribution:\n{df['Diabetes_binary'].value_counts().to_string()}\n")

FEATURE_COLS = [c for c in df.columns if c != "Diabetes_binary"]
X = df[FEATURE_COLS].values.astype("float32")
y = df["Diabetes_binary"].values.astype("float32")

# ── 2. Train / Test split ──────────────────────────────────────────────────
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
print(f"  Train: {len(X_train_full):,}  |  Test (hold-out): {len(X_test):,}\n")

METRIC_NAMES = ["accuracy", "auc", "precision", "recall", "f1"]

# ── 3. K-Fold Cross-Validation ─────────────────────────────────────────────
print("=" * 60)
print(f"CROSS-VALIDATION  ({args.folds}-Fold Stratified)")
print("=" * 60)

kfold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
nn_fold_scores = []
rf_fold_scores = []

for fold_i, (train_idx, val_idx) in enumerate(kfold.split(X_train_full, y_train_full)):
    print(f"\n── Fold {fold_i + 1}/{args.folds} ", "─" * 44)

    X_fold_train = X_train_full[train_idx]
    X_fold_val = X_train_full[val_idx]
    y_fold_train = y_train_full[train_idx]
    y_fold_val = y_train_full[val_idx]

    # Scale
    fold_scaler = StandardScaler()
    X_ft_sc = fold_scaler.fit_transform(X_fold_train)
    X_fv_sc = fold_scaler.transform(X_fold_val)

    # NN needs an internal val split for early stopping
    X_nn_t, X_nn_v, y_nn_t, y_nn_v = train_test_split(
        X_ft_sc, y_fold_train, test_size=0.15, random_state=SEED, stratify=y_fold_train
    )

    # Neural Network
    nn = build_nn(X_ft_sc.shape[1], args.lr)
    nn.fit(
        X_nn_t,
        y_nn_t,
        validation_data=(X_nn_v, y_nn_v),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=8, mode="max", restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0
            ),
        ],
        verbose=0,
    )
    nn_probs = nn.predict(X_fv_sc, verbose=0).ravel()
    nn_m = score(y_fold_val, nn_probs)
    nn_fold_scores.append(nn_m)

    del nn
    tf.keras.backend.clear_session()

    # Random Forest
    rf = build_rf()
    rf.fit(X_ft_sc, y_fold_train)
    rf_probs = rf.predict_proba(X_fv_sc)[:, 1]
    rf_m = score(y_fold_val, rf_probs)
    rf_fold_scores.append(rf_m)

    print(
        f"  NN :  AUC={nn_m['auc']:.4f}  Acc={nn_m['accuracy']:.4f}  F1={nn_m['f1']:.4f}"
    )
    print(
        f"  RF :  AUC={rf_m['auc']:.4f}  Acc={rf_m['accuracy']:.4f}  F1={rf_m['f1']:.4f}"
    )

# ── CV Summary ──
print("\n" + "=" * 60)
print("CROSS-VALIDATION RESULTS  (mean ± std)")
print("=" * 60)

print(
    f"\n  {'Metric':>12s}  {'Neural Network':>20s}  {'Random Forest':>20s}  {'Winner':>8s}"
)
print(f"  {'─' * 12}  {'─' * 20}  {'─' * 20}  {'─' * 8}")

nn_wins = 0
rf_wins = 0
cv_comparison = {}

for m in METRIC_NAMES:
    nn_vals = [s[m] for s in nn_fold_scores]
    rf_vals = [s[m] for s in rf_fold_scores]
    nn_mean, nn_std = np.mean(nn_vals), np.std(nn_vals)
    rf_mean, rf_std = np.mean(rf_vals), np.std(rf_vals)
    winner = "NN" if nn_mean >= rf_mean else "RF"
    if winner == "NN":
        nn_wins += 1
    else:
        rf_wins += 1
    print(
        f"  {m:>12s}  {nn_mean:.4f} ± {nn_std:.4f}      {rf_mean:.4f} ± {rf_std:.4f}      {winner:>8s}"
    )
    cv_comparison[m] = {
        "nn": {"mean": float(nn_mean), "std": float(nn_std)},
        "rf": {"mean": float(rf_mean), "std": float(rf_std)},
    }

cv_winner = "nn" if nn_wins >= rf_wins else "rf"
cv_winner_name = "Neural Network" if cv_winner == "nn" else "Random Forest"
print(f"\n  CV winner: {cv_winner_name}  ({nn_wins} vs {rf_wins} metrics)")

# ── 4. Final training on full training set ──────────────────────────────────
print("\n" + "=" * 60)
print("FINAL MODEL TRAINING  (full training set)")
print("=" * 60)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_full)
X_test_sc = scaler.transform(X_test)

# NN – needs internal val for early stopping
X_nn_train, X_nn_val, y_nn_train, y_nn_val = train_test_split(
    X_train_sc, y_train_full, test_size=0.15, random_state=SEED, stratify=y_train_full
)

print("\nTraining Neural Network …")
final_nn = build_nn(X_train_sc.shape[1], args.lr)
history = final_nn.fit(
    X_nn_train,
    y_nn_train,
    validation_data=(X_nn_val, y_nn_val),
    epochs=args.epochs,
    batch_size=args.batch,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=12, mode="max", restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
    ],
    verbose=1,
)

print("\nTraining Random Forest …")
final_rf = build_rf()
final_rf.fit(X_train_sc, y_train_full)
print("  Done.")

# ── 5. Test-set evaluation ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST SET EVALUATION  (hold-out 20%)")
print("=" * 60)

nn_test_probs = final_nn.predict(X_test_sc, verbose=0).ravel()
rf_test_probs = final_rf.predict_proba(X_test_sc)[:, 1]

nn_test = score(y_test, nn_test_probs)
rf_test = score(y_test, rf_test_probs)

print(
    f"\n  {'Metric':>12s}  {'Neural Network':>16s}  {'Random Forest':>16s}"
)
print(f"  {'─' * 12}  {'─' * 16}  {'─' * 16}")
for m in METRIC_NAMES:
    print(f"  {m:>12s}  {nn_test[m]:>16.4f}  {rf_test[m]:>16.4f}")

best_model = "nn" if nn_test["auc"] >= rf_test["auc"] else "rf"
best_name = "Neural Network" if best_model == "nn" else "Random Forest"
print(f"\n  Best on test set (by AUC): {best_name}")

# Detailed reports
for label, probs in [("Neural Network", nn_test_probs), ("Random Forest", rf_test_probs)]:
    preds = (probs >= 0.5).astype(int)
    print(f"\n── {label} ──")
    print(classification_report(y_test, preds, target_names=["No Diabetes", "Diabetes"]))
    print(f"  Confusion matrix:\n{confusion_matrix(y_test, preds)}\n")

# ── 6. Save artefacts ──────────────────────────────────────────────────────
print("=" * 60)
print("SAVING MODELS")
print("=" * 60)

os.makedirs(args.out_dir, exist_ok=True)

# Neural Network
nn_path = os.path.join(args.out_dir, "diabetes_risk_model.keras")
final_nn.save(nn_path)
print(f"  Neural Network → {nn_path}")

# Random Forest
rf_path = os.path.join(args.out_dir, "diabetes_risk_model_rf.joblib")
joblib.dump(final_rf, rf_path)
print(f"  Random Forest  → {rf_path}")

# Scaler
scaler_params = {
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "feature_names": FEATURE_COLS,
}
scaler_path = os.path.join(args.out_dir, "scaler_params.json")
with open(scaler_path, "w") as f:
    json.dump(scaler_params, f, indent=2)
print(f"  Scaler params  → {scaler_path}")

# Comparison results
comparison = {
    "best_model": best_model,
    "cv_folds": args.folds,
    "cv_results": cv_comparison,
    "test_results": {"nn": nn_test, "rf": rf_test},
}
comp_path = os.path.join(args.out_dir, "comparison_results.json")
with open(comp_path, "w") as f:
    json.dump(comparison, f, indent=2)
print(f"  Comparison     → {comp_path}")

# Training history (NN only)
hist_path = os.path.join(args.out_dir, "training_history.json")
with open(hist_path, "w") as f:
    json.dump(
        {k: [float(v) for v in vals] for k, vals in history.history.items()},
        f,
        indent=2,
    )
print(f"  NN history     → {hist_path}")

print(f"\n  Best model: {best_name}")
print("\nDone. Run  python predict.py  to make predictions.\n")

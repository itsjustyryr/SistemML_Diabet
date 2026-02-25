#!/usr/bin/env python3
"""
Diabetes Risk Prediction – Inference Script
============================================
Loads the best trained model (Neural Network or Random Forest) and scaler,
then scores one or more patient records with a 0-100% risk score.

Usage:
    python predict.py                        # run built-in demo examples
    python predict.py --interactive          # answer prompts for each feature
    python predict.py --csv patient.csv      # score every row in a CSV
    python predict.py --model rf             # force Random Forest model
    python predict.py --model nn             # force Neural Network model
"""

import argparse
import json
import os
import sys
import numpy as np

MODEL_DIR = "saved_model"
NN_PATH = os.path.join(MODEL_DIR, "diabetes_risk_model.keras")
RF_PATH = os.path.join(MODEL_DIR, "diabetes_risk_model_rf.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_params.json")
COMPARISON_PATH = os.path.join(MODEL_DIR, "comparison_results.json")

# ── Feature descriptions (used in interactive mode) ──────────────────────────
FEATURE_PROMPTS = {
    "HighBP": ("High blood pressure? (0=No, 1=Yes)", float),
    "HighChol": ("High cholesterol? (0=No, 1=Yes)", float),
    "CholCheck": ("Cholesterol check in last 5 years? (0=No, 1=Yes)", float),
    "BMI": ("Body Mass Index (e.g. 25)", float),
    "Smoker": ("Have you smoked >=100 cigarettes in life? (0=No, 1=Yes)", float),
    "Stroke": ("Ever had a stroke? (0=No, 1=Yes)", float),
    "HeartDiseaseorAttack": (
        "Coronary heart disease or heart attack? (0=No, 1=Yes)",
        float,
    ),
    "PhysActivity": ("Physical activity in past 30 days? (0=No, 1=Yes)", float),
    "Fruits": ("Consume fruit >=1 time/day? (0=No, 1=Yes)", float),
    "Veggies": ("Consume vegetables >=1 time/day? (0=No, 1=Yes)", float),
    "HvyAlcoholConsump": ("Heavy alcohol consumption? (0=No, 1=Yes)", float),
    "AnyHealthcare": ("Have any health-care coverage? (0=No, 1=Yes)", float),
    "NoDocbcCost": ("Could not see doctor because of cost? (0=No, 1=Yes)", float),
    "GenHlth": ("General health (1=Excellent ... 5=Poor)", float),
    "MentHlth": ("Days of poor mental health in past 30 (0-30)", float),
    "PhysHlth": ("Days of poor physical health in past 30 (0-30)", float),
    "DiffWalk": ("Serious difficulty walking? (0=No, 1=Yes)", float),
    "Sex": ("Sex (0=Female, 1=Male)", float),
    "Age": ("Age category (1=18-24 ... 13=80+)", float),
    "Education": ("Education level (1-6, 6=College grad)", float),
    "Income": ("Income level (1-8, 8=>=$75k)", float),
}


def detect_best_model():
    """Read comparison_results.json to find the best model, default to 'nn'."""
    if os.path.exists(COMPARISON_PATH):
        with open(COMPARISON_PATH) as f:
            comp = json.load(f)
        return comp.get("best_model", "nn")
    return "nn"


def load_artefacts(model_type=None):
    """Load the trained model and scaler parameters."""
    if model_type is None:
        model_type = detect_best_model()

    if model_type == "rf" and os.path.exists(RF_PATH):
        import joblib

        model = joblib.load(RF_PATH)
        print(f"  Loaded Random Forest model from {RF_PATH}")
    elif os.path.exists(NN_PATH):
        import tensorflow as tf

        model = tf.keras.models.load_model(NN_PATH)
        model_type = "nn"
        print(f"  Loaded Neural Network model from {NN_PATH}")
    else:
        sys.exit(f"No model found in {MODEL_DIR}/. Run train_model.py first.")

    with open(SCALER_PATH) as f:
        sp = json.load(f)
    mean = np.array(sp["mean"], dtype="float32")
    scale = np.array(sp["scale"], dtype="float32")
    feature_names = sp["feature_names"]
    return model, mean, scale, feature_names, model_type


def predict_risk(model, mean, scale, features: np.ndarray, model_type="nn") -> np.ndarray:
    """Return risk percentages (0-100) for an array of feature vectors."""
    scaled = (features - mean) / scale
    if model_type == "rf":
        probs = model.predict_proba(scaled)[:, 1]
    else:
        probs = model.predict(scaled, verbose=0).ravel()
    return np.round(probs * 100, 2)


def risk_label(pct: float) -> str:
    if pct < 25:
        return "LOW"
    if pct < 50:
        return "MODERATE"
    if pct < 75:
        return "HIGH"
    return "VERY HIGH"


def print_result(pct: float, idx: int | None = None):
    prefix = f"Patient {idx}: " if idx is not None else ""
    label = risk_label(pct)
    bar_len = int(pct / 2)
    bar = "#" * bar_len + "-" * (50 - bar_len)
    print(f"  {prefix}Risk = {pct:6.2f}%  [{bar}]  {label}")


# ── Interactive mode ─────────────────────────────────────────────────────────
def interactive(model, mean, scale, feature_names, model_type):
    print("\n── Interactive diabetes risk assessment ────────────")
    print("Answer each question below. Press Ctrl+C to quit.\n")
    while True:
        values = []
        try:
            for feat in feature_names:
                prompt_text, dtype = FEATURE_PROMPTS[feat]
                raw = input(f"  {prompt_text}: ")
                values.append(dtype(raw))
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        features = np.array([values], dtype="float32")
        pct = predict_risk(model, mean, scale, features, model_type)[0]
        print()
        print_result(pct)
        print()


# ── CSV scoring mode ─────────────────────────────────────────────────────────
def score_csv(model, mean, scale, feature_names, model_type, csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        sys.exit(f"CSV is missing columns: {missing}")

    features = df[feature_names].values.astype("float32")
    pcts = predict_risk(model, mean, scale, features, model_type)
    df["risk_pct"] = pcts
    df["risk_label"] = [risk_label(p) for p in pcts]

    out_path = csv_path.replace(".csv", "_scored.csv")
    df.to_csv(out_path, index=False)
    print(f"\nScored {len(df):,} records -> {out_path}")
    print(f"  Mean risk: {pcts.mean():.2f}%  |  Median: {np.median(pcts):.2f}%\n")
    for i, pct in enumerate(pcts[:10]):
        print_result(pct, idx=i + 1)
    if len(pcts) > 10:
        print(f"  ... and {len(pcts) - 10} more rows (see {out_path})")


# ── Demo mode ────────────────────────────────────────────────────────────────
def demo(model, mean, scale, model_type):
    print("\n── Demo predictions (hard-coded sample profiles) ──\n")

    profiles = {
        "Healthy 30-yr-old": [
            0, 0, 1, 23, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 4, 6, 7,
        ],
        "Obese smoker, 55-yr-old, high BP & cholesterol": [
            1, 1, 1, 38, 1, 0, 1, 0, 0, 0, 0, 1, 0, 4, 10, 15, 1, 1, 9, 4, 4,
        ],
        "Active retiree, slight overweight": [
            1, 0, 1, 27, 0, 0, 0, 1, 1, 1, 0, 1, 0, 2, 2, 3, 0, 1, 12, 5, 6,
        ],
        "Sedentary, poor diet, low income": [
            0, 1, 0, 31, 1, 0, 0, 0, 0, 0, 1, 0, 1, 4, 20, 10, 1, 0, 7, 3, 2,
        ],
    }

    for name, vals in profiles.items():
        features = np.array([vals], dtype="float32")
        pct = predict_risk(model, mean, scale, features, model_type)[0]
        print(f"  {name}")
        print_result(pct)
        print()


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict diabetes risk")
    parser.add_argument(
        "--interactive", action="store_true", help="Enter values interactively"
    )
    parser.add_argument("--csv", type=str, help="Score all rows in a CSV file")
    parser.add_argument("--model_dir", default=MODEL_DIR, help="Directory with saved model")
    parser.add_argument(
        "--model",
        choices=["nn", "rf", "auto"],
        default="auto",
        help="Which model to use (default: auto = best from training)",
    )
    cli = parser.parse_args()

    MODEL_DIR = cli.model_dir
    NN_PATH = os.path.join(MODEL_DIR, "diabetes_risk_model.keras")
    RF_PATH = os.path.join(MODEL_DIR, "diabetes_risk_model_rf.joblib")
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler_params.json")
    COMPARISON_PATH = os.path.join(MODEL_DIR, "comparison_results.json")

    chosen = None if cli.model == "auto" else cli.model
    model, mean, scale, feature_names, model_type = load_artefacts(chosen)

    model_label = "Neural Network" if model_type == "nn" else "Random Forest"
    print(f"  Using: {model_label}\n")

    if cli.interactive:
        interactive(model, mean, scale, feature_names, model_type)
    elif cli.csv:
        score_csv(model, mean, scale, feature_names, model_type, cli.csv)
    else:
        demo(model, mean, scale, model_type)

#!/usr/bin/env python3
"""
Dataset Preparation Script
==========================
Converts all datasets from 'dataset full/' into the 50-50 balanced binary
format compatible with train_model.py.

Datasets processed:
  1. BRFSS 2015 3-class (012)  → binary + 50-50 balance
  2. NHANES (demographic + examination + questionnaire + medications)
     → feature-engineered to match 21 BRFSS features + 50-50 balance
  3. Pima Indians Diabetes → partial mapping + median imputation + 50-50
  4. Combined (all sources merged) → 50-50 balance

Output files are saved to 'dataset/'.

Usage:
    python prepare_datasets.py
"""

import os
import pandas as pd
import numpy as np

SRC = "dataset full"
DST = "dataset"
SEED = 42
np.random.seed(SEED)

# Exact column order expected by train_model.py
COLUMNS = [
    "Diabetes_binary", "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth",
    "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]

# ─── Helpers ────────────────────────────────────────────────────────────────


def balance_5050(df, target="Diabetes_binary"):
    """Down-sample the majority class to create a 50-50 balance."""
    pos = df[df[target] == 1.0]
    neg = df[df[target] == 0.0]
    n = min(len(pos), len(neg))
    if n == 0:
        print("  WARNING: one class is empty – returning unbalanced data")
        return df
    balanced = pd.concat([
        pos.sample(n=n, random_state=SEED),
        neg.sample(n=n, random_state=SEED),
    ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return balanced


def age_years_to_brfss_category(age_series):
    """Convert continuous age (years) to BRFSS 13-level category.

    1=18-24, 2=25-29, 3=30-34, 4=35-39, 5=40-44, 6=45-49,
    7=50-54, 8=55-59, 9=60-64, 10=65-69, 11=70-74, 12=75-79, 13=80+
    """
    bins = [0, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 999]
    labels = list(range(1, 14))
    return pd.cut(age_series, bins=bins, labels=labels, right=True).astype(float)


def nhanes_income_to_brfss(income_code):
    """Map NHANES INDHHIN2 codes to BRFSS Income (1-8).

    NHANES: 1=$0-5k, 2=$5-10k, 3=$10-15k, 4=$15-20k, 5=$20-25k,
            6=$25-35k, 7=$35-45k, 8=$45-55k, 9=$55-65k, 10=$65-75k,
            14=$75-100k, 15=$100k+
    BRFSS:  1=<$10k, 2=$10-15k, 3=$15-20k, 4=$20-25k, 5=$25-35k,
            6=$35-50k, 7=$50-75k, 8=$75k+
    """
    mapping = {
        1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
        8: 7, 9: 7, 10: 7, 14: 8, 15: 8,
    }
    return income_code.map(mapping)


def nhanes_education_to_brfss(edu_code):
    """Map NHANES DMDEDUC2 codes to BRFSS Education (1-6).

    NHANES: 1=<9th grade, 2=9-11th, 3=HS grad/GED, 4=Some college/AA, 5=College grad+
    BRFSS:  1=None/kindergarten, 2=Elementary, 3=Some HS, 4=HS grad, 5=Some college, 6=College grad
    """
    mapping = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
    return edu_code.map(mapping)


def yn_to_binary(series, yes_val=1):
    """Convert Yes/No coded series (1=Yes, 2=No) to 1.0/0.0."""
    return (series == yes_val).astype(float)


def safe_col(df, col, default=np.nan):
    """Safely get a column, returning default Series if missing."""
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


# ─── 1. Convert BRFSS 3-class (012) to binary 50-50 ────────────────────────


def convert_brfss_012():
    """Convert 3-class BRFSS (0=no diabetes, 1=pre-diabetes, 2=diabetes)
    to binary (0 vs 1) and balance 50-50.
    Returns (balanced, full_unbalanced)."""
    print("\n" + "=" * 60)
    print("1. Converting BRFSS 2015 3-class → binary 50-50")
    print("=" * 60)

    path = os.path.join(SRC, "diabetes_012_health_indicators_BRFSS2015.csv")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"  Original class distribution:")
    print(f"    {df['Diabetes_012'].value_counts().to_dict()}")

    # Merge pre-diabetes (1) and diabetes (2) into positive class
    df["Diabetes_binary"] = (df["Diabetes_012"] >= 1).astype(float)
    df = df.drop(columns=["Diabetes_012"])

    # Ensure correct column order
    full = df[COLUMNS].copy()

    print(f"  After binary conversion: {full['Diabetes_binary'].value_counts().to_dict()}")

    balanced = balance_5050(full)
    print(f"  After 50-50 balancing: {len(balanced):,} rows")

    out = os.path.join(DST, "brfss2015_012_converted_5050.csv")
    balanced.to_csv(out, index=False)
    print(f"  Saved → {out}")
    return balanced, full


# ─── 2. Convert NHANES data to BRFSS format ────────────────────────────────


def convert_nhanes(reference_medians):
    """Join NHANES tables, engineer features to match BRFSS 21-feature format,
    and balance 50-50."""
    print("\n" + "=" * 60)
    print("2. Converting NHANES → BRFSS format 50-50")
    print("=" * 60)

    # Load tables
    demo = pd.read_csv(os.path.join(SRC, "demographic.csv"))
    exam = pd.read_csv(os.path.join(SRC, "examination.csv"))
    quest = pd.read_csv(os.path.join(SRC, "questionnaire.csv"))
    meds = pd.read_csv(os.path.join(SRC, "medications.csv"), encoding="latin-1")

    print(f"  Loaded: demo={len(demo):,}, exam={len(exam):,}, "
          f"quest={len(quest):,}, meds={len(meds):,}")

    # Join on SEQN
    df = demo.merge(exam, on="SEQN", how="inner")
    df = df.merge(quest, on="SEQN", how="inner")
    print(f"  After join: {len(df):,} participants")

    # Filter adults only (BRFSS is 18+)
    if "RIDAGEYR" in df.columns:
        df = df[df["RIDAGEYR"] >= 18].copy()
        print(f"  Adults (18+): {len(df):,}")

    # Identify participants on diabetes medications
    diabetes_med_seqns = set()
    for col in ["RXDRSD1", "RXDRSD2", "RXDRSD3"]:
        if col in meds.columns:
            mask = meds[col].astype(str).str.lower().str.contains("diabetes", na=False)
            diabetes_med_seqns |= set(meds.loc[mask, "SEQN"].unique())
    # Also check drug names for insulin
    if "RXDDRUG" in meds.columns:
        mask = meds["RXDDRUG"].astype(str).str.lower().str.contains("insulin|metformin|glipizide|glyburide", na=False)
        diabetes_med_seqns |= set(meds.loc[mask, "SEQN"].unique())
    print(f"  Participants on diabetes medications: {len(diabetes_med_seqns)}")

    # ── Build output DataFrame ──
    out = pd.DataFrame(index=df.index)

    # TARGET: Diabetes_binary
    # DIQ010: Doctor told you have diabetes? (1=Yes, 2=No, 3=Borderline)
    out["Diabetes_binary"] = yn_to_binary(safe_col(df, "DIQ010"))
    # Also count those on diabetes meds as positive
    out.loc[df["SEQN"].isin(diabetes_med_seqns), "Diabetes_binary"] = 1.0

    # HighBP: BPQ020 = Ever told you had high blood pressure? (1=Yes, 2=No)
    out["HighBP"] = yn_to_binary(safe_col(df, "BPQ020"))

    # HighChol: BPQ080 = Doctor told high cholesterol? (1=Yes, 2=No)
    out["HighChol"] = yn_to_binary(safe_col(df, "BPQ080"))

    # CholCheck: BPQ060 = Ever had blood cholesterol checked? (1=Yes, 2=No)
    out["CholCheck"] = yn_to_binary(safe_col(df, "BPQ060"))

    # BMI: BMXBMI from examination
    out["BMI"] = safe_col(df, "BMXBMI")

    # Smoker: SMQ020 = Smoked at least 100 cigarettes in life? (1=Yes, 2=No)
    out["Smoker"] = yn_to_binary(safe_col(df, "SMQ020"))

    # Stroke: MCQ160F = Ever told you had a stroke? (1=Yes, 2=No)
    out["Stroke"] = yn_to_binary(safe_col(df, "MCQ160F"))

    # HeartDiseaseorAttack: MCQ160C (coronary heart disease) or MCQ160D (heart attack)
    chd = yn_to_binary(safe_col(df, "MCQ160C"))
    ha = yn_to_binary(safe_col(df, "MCQ160D"))
    out["HeartDiseaseorAttack"] = ((chd == 1) | (ha == 1)).astype(float)

    # PhysActivity: any vigorous/moderate activity
    # PAQ605=vigorous work, PAQ620=moderate work, PAQ635=walk/bike, PAQ650=vigorous rec, PAQ665=moderate rec
    activity_cols = ["PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"]
    activity = pd.DataFrame()
    for col in activity_cols:
        activity[col] = yn_to_binary(safe_col(df, col))
    out["PhysActivity"] = (activity.sum(axis=1) > 0).astype(float)

    # Fruits: not directly available in NHANES questionnaire → impute
    out["Fruits"] = reference_medians.get("Fruits", 1.0)

    # Veggies: not directly available → impute
    out["Veggies"] = reference_medians.get("Veggies", 1.0)

    # HvyAlcoholConsump: derive from ALQ120Q, ALQ120U, ALQ130
    # ALQ130 = avg drinks/day, ALQ120Q = frequency, ALQ120U = unit (1=week, 2=month, 3=year)
    drinks_per_day = safe_col(df, "ALQ130").copy()
    freq_q = safe_col(df, "ALQ120Q").copy()
    freq_u = safe_col(df, "ALQ120U").copy()
    # Convert frequency to per-week
    freq_per_week = pd.Series(0.0, index=df.index)
    freq_per_week = np.where(freq_u == 1, freq_q, freq_per_week)
    freq_per_week = np.where(freq_u == 2, freq_q / 4.33, freq_per_week)
    freq_per_week = np.where(freq_u == 3, freq_q / 52.0, freq_per_week)
    drinks_per_week = pd.to_numeric(drinks_per_day, errors="coerce").fillna(0) * freq_per_week
    sex_series = safe_col(df, "RIAGENDR")  # 1=Male, 2=Female
    heavy_male = (sex_series == 1) & (drinks_per_week > 14)
    heavy_female = (sex_series == 2) & (drinks_per_week > 7)
    out["HvyAlcoholConsump"] = (heavy_male | heavy_female).astype(float)

    # AnyHealthcare: HIQ011 = Covered by health insurance? (1=Yes, 2=No)
    out["AnyHealthcare"] = yn_to_binary(safe_col(df, "HIQ011"))

    # NoDocbcCost: not directly available in NHANES → impute
    out["NoDocbcCost"] = reference_medians.get("NoDocbcCost", 0.0)

    # GenHlth: HSD010 = General health condition (1=Excellent to 5=Poor)
    # Matches BRFSS encoding directly
    gen_hlth = safe_col(df, "HSD010").copy()
    gen_hlth = gen_hlth.where(gen_hlth.between(1, 5))
    out["GenHlth"] = gen_hlth

    # MentHlth: approximate from PHQ-9 depression screener (DPQ010-DPQ090)
    # Each item scored 0-3, total 0-27. Scale to 0-30 days.
    phq_cols = [f"DPQ{i:03d}" for i in range(10, 100, 10)]
    phq_items = pd.DataFrame()
    for col in phq_cols:
        s = safe_col(df, col).copy()
        s = pd.to_numeric(s, errors="coerce")
        s = s.where(s.between(0, 3))
        phq_items[col] = s
    phq_total = phq_items.sum(axis=1, min_count=1)  # NaN if all items missing
    out["MentHlth"] = (phq_total / 27.0 * 30.0).round(0)

    # PhysHlth: approximate from HSQ571 (# days physical health not good, past 30d)
    # If not available, use general health as rough proxy
    phys = safe_col(df, "HSQ571").copy()
    phys = pd.to_numeric(phys, errors="coerce")
    phys = phys.where(phys.between(0, 30))
    out["PhysHlth"] = phys

    # DiffWalk: DLQ020 = Serious difficulty walking or climbing stairs? (1=Yes, 2=No)
    out["DiffWalk"] = yn_to_binary(safe_col(df, "DLQ020"))

    # Sex: RIAGENDR (1=Male, 2=Female) → BRFSS (0=Female, 1=Male)
    sex = safe_col(df, "RIAGENDR")
    out["Sex"] = sex.map({1: 1.0, 2: 0.0})

    # Age: RIDAGEYR → BRFSS age category
    out["Age"] = age_years_to_brfss_category(safe_col(df, "RIDAGEYR"))

    # Education: DMDEDUC2 → BRFSS scale
    out["Education"] = nhanes_education_to_brfss(safe_col(df, "DMDEDUC2"))

    # Income: INDHHIN2 → BRFSS scale
    out["Income"] = nhanes_income_to_brfss(safe_col(df, "INDHHIN2"))

    # Fill remaining NaN values with reference medians
    for col in COLUMNS:
        if col in out.columns and out[col].isna().any():
            fill_val = reference_medians.get(col, 0.0)
            n_fill = out[col].isna().sum()
            out[col] = out[col].fillna(fill_val)
            if n_fill > 0:
                print(f"    Filled {n_fill:,} NaN in '{col}' with median {fill_val}")

    full = out[COLUMNS].astype(float)

    print(f"  Class distribution: {full['Diabetes_binary'].value_counts().to_dict()}")

    balanced = balance_5050(full)
    print(f"  After 50-50 balancing: {len(balanced):,} rows")

    out_path = os.path.join(DST, "nhanes_converted_5050.csv")
    balanced.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    return balanced, full


# ─── 3. Convert Pima Indians Diabetes Dataset ──────────────────────────────


def convert_pima(reference_medians):
    """Convert Pima Indians diabetes dataset. Only BMI and Age can be
    directly mapped; other features use population medians from BRFSS."""
    print("\n" + "=" * 60)
    print("3. Converting Pima Indians Diabetes → BRFSS format 50-50")
    print("=" * 60)

    path = os.path.join(SRC, "diabetes.csv")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"  Original columns: {list(df.columns)}")
    print(f"  NOTE: Only BMI and Age map directly. Other features are derived")
    print(f"        or filled with BRFSS population medians.")
    print(f"        This dataset is ALL FEMALE (Pima Indian women, age 21+).")

    out = pd.DataFrame(index=df.index)

    # Target
    out["Diabetes_binary"] = df["Outcome"].astype(float)

    # HighBP: derive from BloodPressure (diastolic). Hypertension if >= 90 mmHg
    bp = df["BloodPressure"].replace(0, np.nan)  # 0 = missing in Pima
    out["HighBP"] = (bp >= 90).astype(float)
    out.loc[bp.isna(), "HighBP"] = reference_medians.get("HighBP", 0.0)

    # HighChol: not available → impute
    out["HighChol"] = reference_medians.get("HighChol", 0.0)

    # CholCheck: not available → impute
    out["CholCheck"] = reference_medians.get("CholCheck", 1.0)

    # BMI: direct mapping (Pima uses 0 for missing)
    bmi = df["BMI"].replace(0, np.nan)
    out["BMI"] = bmi.fillna(reference_medians.get("BMI", 28.0))

    # Smoker: not available → impute
    out["Smoker"] = reference_medians.get("Smoker", 0.0)

    # Stroke: not available → impute
    out["Stroke"] = reference_medians.get("Stroke", 0.0)

    # HeartDiseaseorAttack: not available → impute
    out["HeartDiseaseorAttack"] = reference_medians.get("HeartDiseaseorAttack", 0.0)

    # PhysActivity: not available → impute
    out["PhysActivity"] = reference_medians.get("PhysActivity", 1.0)

    # Fruits: not available → impute
    out["Fruits"] = reference_medians.get("Fruits", 1.0)

    # Veggies: not available → impute
    out["Veggies"] = reference_medians.get("Veggies", 1.0)

    # HvyAlcoholConsump: not available → impute
    out["HvyAlcoholConsump"] = reference_medians.get("HvyAlcoholConsump", 0.0)

    # AnyHealthcare: not available → impute
    out["AnyHealthcare"] = reference_medians.get("AnyHealthcare", 1.0)

    # NoDocbcCost: not available → impute
    out["NoDocbcCost"] = reference_medians.get("NoDocbcCost", 0.0)

    # GenHlth: not available → impute
    out["GenHlth"] = reference_medians.get("GenHlth", 3.0)

    # MentHlth: not available → impute
    out["MentHlth"] = reference_medians.get("MentHlth", 0.0)

    # PhysHlth: not available → impute
    out["PhysHlth"] = reference_medians.get("PhysHlth", 0.0)

    # DiffWalk: not available → impute
    out["DiffWalk"] = reference_medians.get("DiffWalk", 0.0)

    # Sex: all female in Pima dataset
    out["Sex"] = 0.0

    # Age: convert years to BRFSS category
    out["Age"] = age_years_to_brfss_category(df["Age"])

    # Education: not available → impute
    out["Education"] = reference_medians.get("Education", 4.0)

    # Income: not available → impute
    out["Income"] = reference_medians.get("Income", 5.0)

    full = out[COLUMNS].astype(float)
    print(f"  Class distribution: {full['Diabetes_binary'].value_counts().to_dict()}")

    balanced = balance_5050(full)
    print(f"  After 50-50 balancing: {len(balanced):,} rows")

    out_path = os.path.join(DST, "pima_converted_5050.csv")
    balanced.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    return balanced, full


# ─── 4. Combine all sources ────────────────────────────────────────────────


def load_brfss_binary_full():
    """Load the full BRFSS binary dataset (253k rows, already in target format)."""
    print("\n" + "=" * 60)
    print("0. Loading full BRFSS binary dataset")
    print("=" * 60)

    path = os.path.join(SRC, "diabetes_binary_health_indicators_BRFSS2015.csv")
    df = pd.read_csv(path)
    df = df[COLUMNS].astype(float)
    print(f"  Loaded {len(df):,} rows")
    print(f"  Class distribution: {df['Diabetes_binary'].value_counts().to_dict()}")
    return df


def combine_all(full_datasets, names):
    """Merge ALL unbalanced rows from every source, then balance 50-50 once."""
    print("\n" + "=" * 60)
    print("4. Creating FULL combined dataset from all sources")
    print("=" * 60)

    for name, ds in zip(names, full_datasets):
        pos = int((ds["Diabetes_binary"] == 1).sum())
        neg = int((ds["Diabetes_binary"] == 0).sum())
        print(f"  {name:25s}: {len(ds):>8,} rows  (pos={pos:,}, neg={neg:,})")

    combined = pd.concat(full_datasets, ignore_index=True)
    print(f"\n  Total rows (all sources): {len(combined):,}")
    print(f"  Class distribution: {combined['Diabetes_binary'].value_counts().to_dict()}")

    # Save ALL rows (no balancing)
    all_path = os.path.join(DST, "combined_all_sources.csv")
    combined.to_csv(all_path, index=False)
    print(f"  Saved → {all_path}")

    # Also save a 50-50 balanced version
    balanced = balance_5050(combined)
    print(f"  After 50-50 balancing:    {len(balanced):,} rows")

    bal_path = os.path.join(DST, "combined_all_sources_5050.csv")
    balanced.to_csv(bal_path, index=False)
    print(f"  Saved → {bal_path}")
    return balanced


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    print("Dataset Preparation Script")
    print("=" * 60)

    os.makedirs(DST, exist_ok=True)

    # Load reference dataset to compute medians for imputation
    ref_path = os.path.join(DST, "diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    if not os.path.exists(ref_path):
        # Fallback: check in source folder
        ref_path = os.path.join(SRC, "diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

    print(f"Loading reference dataset for median imputation: {ref_path}")
    ref = pd.read_csv(ref_path)
    reference_medians = ref.median().to_dict()
    print(f"  Reference medians computed from {len(ref):,} rows\n")
    for col in COLUMNS:
        if col in reference_medians:
            print(f"    {col:>25s}: {reference_medians[col]:.1f}")

    # Load the full BRFSS binary dataset (base, 253k rows)
    brfss_full = load_brfss_binary_full()

    # Run conversions (each returns (balanced, full_unbalanced))
    _, brfss012_full = convert_brfss_012()
    _, nhanes_full = convert_nhanes(reference_medians)
    _, pima_full = convert_pima(reference_medians)

    # Combine ALL unbalanced rows, then balance once
    # Note: brfss_full and brfss012_full are the same BRFSS survey data
    # (012 just had 3-class target). We use brfss_full as the base and
    # add only the non-BRFSS sources (NHANES + Pima) to avoid duplicates.
    combine_all(
        [brfss_full, nhanes_full, pima_full],
        ["BRFSS binary full (253k)", "NHANES (all adults)", "Pima (all rows)"],
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output files in '{DST}/':")
    for f in sorted(os.listdir(DST)):
        if f.endswith(".csv"):
            fpath = os.path.join(DST, f)
            n = sum(1 for _ in open(fpath)) - 1
            print(f"  {f:55s} {n:>8,} rows")

    print("\nTo train with a specific dataset:")
    print("  python train_model.py --data dataset/brfss2015_012_converted_5050.csv")
    print("  python train_model.py --data dataset/nhanes_converted_5050.csv")
    print("  python train_model.py --data dataset/pima_converted_5050.csv")
    print("  python train_model.py --data dataset/combined_all_sources_5050.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()

# cycleaware_ml.py

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import warnings

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

from model_runner import ModelRunner
from plotter import ResultsPlotter

# Load Data
#df = pd.read_csv("menstrual_aware_dataset_with_user_id.csv")
df = pd.read_csv("menstrual_aware_largedataset.csv")
df = df.sort_values(["User_ID"]).reset_index(drop=True)

# Create Cycle Phase (Categorical)
if "cycle_day" not in df.columns:
    df["cycle_day"] = df.groupby("User_ID").cumcount() % 28 + 1

def get_phase(day):
    if day <= 5:
        return "menstrual"
    elif day <= 12:
        return "follicular"
    elif day <= 16:
        return "ovulatory"
    else:
        return "luteal"

df["cycle_phase"] = df["cycle_day"].apply(get_phase)

# Within-User Normalization
df["sleep_user_mean"] = df.groupby("User_ID")["sleep"].transform("mean")
df["stress_user_mean"] = df.groupby("User_ID")["stress"].transform("mean")
df["sleep_z"] = (
                        df["sleep"] - df["sleep_user_mean"]
                ) / df.groupby("User_ID")["sleep"].transform("std")
df["stress_z"] = (
                         df["stress"] - df["stress_user_mean"]
                 ) / df.groupby("User_ID")["stress"].transform("std")

df = df.dropna().reset_index(drop=True)

# Feature Sets
features_with = [
    "sleep_z",
    "stress_z",
    "mood",
    "activity",
    "cycle_phase"
]

features_without = [
    "sleep_z",
    "stress_z",
    "mood",
    "activity"
]

target_attention = "attention_score"
target_memory = "memory_score"

# Group-Aware Train/Test Split
gss = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df["User_ID"]))

X_train_w = df.loc[train_idx, features_with]
X_test_w = df.loc[test_idx, features_with]

X_train_wo = df.loc[train_idx, features_without]
X_test_wo = df.loc[test_idx, features_without]

y_train_att = df.loc[train_idx, target_attention]
y_test_att = df.loc[test_idx, target_attention]

y_train_mem = df.loc[train_idx, target_memory]
y_test_mem = df.loc[test_idx, target_memory]

# Preprocessing
categorical_cols_with = ["activity", "cycle_phase"]
categorical_cols_without = ["activity"]

numeric_cols_with = [
    "sleep_z",
    "stress_z",
    "mood"
]

numeric_cols_without = [
    "sleep_z",
    "stress_z",
    "mood"
]

preprocess_with = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols_with),
    ("num", StandardScaler(), numeric_cols_with)
])

preprocess_without = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols_without),
    ("num", StandardScaler(), numeric_cols_without)
])


# Run Models
models = ["linear", "rf", "xgb", "cat"]

runner = ModelRunner(preprocess_with, preprocess_without)

all_results = runner.run_all_models(
    models,
    X_train_w, X_test_w,
    X_train_wo, X_test_wo,
    y_train_att, y_test_att,
    y_train_mem, y_test_mem
)

# Results Table
results_table = pd.DataFrame()

for model in models:
    results_table[f"Att_R2_{model}"] = [
        all_results["Attention"][model]["r2_with"]
    ]
    results_table[f"Mem_R2_{model}"] = [
        all_results["Memory"][model]["r2_with"]
    ]

print("\n=== Model Comparison Table ===")
print(results_table)

# Cross Validation (Group Aware)
print("\n\n===== CROSS-VALIDATION RESULTS =====")

cv_results = {}

for model in models:
    print(f"\nCross-validating {model.upper()}...")

    att_cv = runner.cross_validate_model(
        model,
        df[features_with],
        df[features_without],
        df[target_attention],
        df["User_ID"]
    )

    mem_cv = runner.cross_validate_model(
        model,
        df[features_with],
        df[features_without],
        df[target_memory],
        df["User_ID"]
    )

    cv_results[model] = {
        "Attention": att_cv,
        "Memory": mem_cv
    }

    print("\nAttention:")
    print(att_cv)

    print("\nMemory:")
    print(mem_cv)

# Plot Results
#ResultsPlotter.plot_model_comparison(all_results)
ResultsPlotter.plot_cycle_comparison(cv_results)
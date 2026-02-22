# CycleAware-AI
# Menstrual Cycle–Aware Modeling

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
plt.close('all')
from sklearn.metrics import r2_score, mean_absolute_error

# Load DataSet
df = pd.read_csv("menstrual_aware_dataset_with_user_id.csv")
print("\nDataset Loaded Successfully\n")

# Define Features
features_with_phase = ["sleep", "stress", "mood", "activity", "cycle_phase"]
features_without_phase = ["sleep", "stress", "mood", "activity"]

# Define Targets
target_attention = "attention_score"
target_memory = "memory_score"

# Train-Test Split
train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, random_state=42
)

X_train_w = df.loc[train_idx, features_with_phase]
X_test_w  = df.loc[test_idx,  features_with_phase]

X_train_wo = df.loc[train_idx, features_without_phase]
X_test_wo  = df.loc[test_idx,  features_without_phase]

y_train_att = df.loc[train_idx, target_attention]
y_test_att  = df.loc[test_idx,  target_attention]

y_train_mem = df.loc[train_idx, target_memory]
y_test_mem  = df.loc[test_idx,  target_memory]

# Pre Processing
categorical_with_phase = ["activity", "cycle_phase"]
categorical_without_phase = ["activity"]

preprocess_with_phase = ColumnTransformer(
    [
        ("cat", OneHotEncoder(), categorical_with_phase),
        ("num", StandardScaler(), ["sleep", "stress", "mood"])
    ]
)

preprocess_without_phase = ColumnTransformer(
    [
        ("cat", OneHotEncoder(), categorical_without_phase),
        ("num", StandardScaler(), ["sleep", "stress", "mood"])
    ]
)

# Model Pipeline
def build_model(preprocess):
    return Pipeline([
        ("prep", preprocess),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

# Build Attention Models
model_attn_with_phase = build_model(preprocess_with_phase)
model_attn_without_phase = build_model(preprocess_without_phase)
# Build Memory Models
model_mem_with_phase = build_model(preprocess_with_phase)
model_mem_without_phase = build_model(preprocess_without_phase)

# Train Attention Models
model_attn_with_phase.fit(X_train_w, y_train_att)
model_attn_without_phase.fit(X_train_wo, y_train_att)
# Train Memory Models
model_mem_with_phase.fit(X_train_w, y_train_mem)
model_mem_without_phase.fit(X_train_wo, y_train_mem)

# Predict using the models
pred_att_with    = model_attn_with_phase.predict(X_test_w)
pred_att_without = model_attn_without_phase.predict(X_test_wo)

pred_mem_with    = model_mem_with_phase.predict(X_test_w)
pred_mem_without = model_mem_without_phase.predict(X_test_wo)

# Training Predictions
train_pred_att_with = model_attn_with_phase.predict(X_train_w)
train_pred_att_without = model_attn_without_phase.predict(X_train_wo)

# Testing R2 Scores
r2_train_with = r2_score(y_train_att, train_pred_att_with)
r2_test_with = r2_score(y_test_att, pred_att_with)

r2_train_without = r2_score(y_train_att, train_pred_att_without)
r2_test_without = r2_score(y_test_att, pred_att_without)

print("\nR2 Scores (Attention Model)")
print("With Cycle - Train:", round(r2_train_with,3),
      "Test:", round(r2_test_with,3))

print("Without Cycle - Train:", round(r2_train_without,3),
      "Test:", round(r2_test_without,3))

# Show results as table
# RESULT: Comparative Results with and without cycle as a feature
results_table = pd.DataFrame({
    "User_ID": df.loc[test_idx, "User_ID"].values,
    "Cycle Phase": df.loc[test_idx, "cycle_phase"].values,
    "Tested_Att": y_test_att.values,
    "Pred_Att_W": np.round(pred_att_with, 1),
    "Pred_Att_WO": np.round(pred_att_without, 1),
    "Tested_Mem": y_test_mem.values,
    "Pred_Mem_W": np.round(pred_mem_with, 1),
    "Pred_Mem_WO": np.round(pred_mem_without, 1),
})

# Sort and show first few users
results_table = results_table.sort_values("User_ID").head(10)

print("\n Displaying Comparative Results for few Users, the model is run over 100 users")
print(results_table.to_string(index=False))

# Statistical Analysis of the results
# compare the accuracy based on predicted output
# for models trained using the menstrual cycle data vs without menstrual cycle data
from statistical_validation import CycleAwareStats

df_test = df.loc[test_idx].copy()

stats = CycleAwareStats(
    y_test_att.values,
    pred_att_with,
    pred_att_without,
    y_test_mem.values,
    pred_mem_with,
    pred_mem_without,
    df_test
)
stats.full_report()

# Show results as side by side graphs for R2 and MAE
fig, axes = plt.subplots(1, 2)

# R2 Plot
labels = ["Train With", "Test With", "Train Without", "Test Without"]
scores = [r2_train_with, r2_test_with,
          r2_train_without, r2_test_without]

axes[0].bar(labels, scores)
axes[0].set_title("R2 Scores (200 Trees)")
axes[0].set_ylabel("R2 Score")
axes[0].tick_params(axis='x', rotation=45)

# MAE Plot
mae_with = mean_absolute_error(y_test_att, pred_att_with)
mae_without = mean_absolute_error(y_test_att, pred_att_without)

axes[1].bar(["With Cycle", "Without Cycle"],
            [mae_with, mae_without])
axes[1].set_title("MAE Comparison")
axes[1].set_ylabel("Mean Absolute Error")

plt.tight_layout()
plt.show()
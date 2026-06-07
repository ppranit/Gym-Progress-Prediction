"""
train_model.py
Trains Gradient Boosting models for:
  - Weight after 90 days  (regression)
  - Body Fat after 90 days (regression)
  - Goal Achievement Probability (classification)
Saves the three pipelines + the feature scaler to models/
"""

import os, joblib, warnings
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score
warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "gym_progress_prediction_dataset_30000.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Load Data ───────────────────────────────────────────────────────────────
df = pd.read_csv("D:\Gym Progress Predicton\gym_predictor\gym_progress_prediction_dataset_30000.csv")

# ─── Feature Engineering ─────────────────────────────────────────────────────
df["Total_Active_Min"] = df["Cardio_Minutes"] + df["Strength_Minutes"]
df["Weekly_Active_Hours"] = (df["Workout_Days_Per_Week"] * df["Workout_Duration_Min"]) / 60
df["Protein_Per_KG"] = df["Protein_Intake_g"] / df["Initial_Weight_kg"]
df["Calorie_Deficit"] = df["Daily_Calories"] - 2000  # rough TDEE anchor
df["Cardio_Ratio"] = df["Cardio_Minutes"] / (df["Total_Active_Min"] + 1e-6)

FEATURES = [
    "Age", "Gender", "Height_cm", "Initial_Weight_kg", "BMI",
    "Workout_Days_Per_Week", "Workout_Duration_Min",
    "Cardio_Minutes", "Strength_Minutes", "Sleep_Hours",
    "Water_Intake_L", "Daily_Steps", "Protein_Intake_g",
    "Daily_Calories", "Stress_Level", "Training_Experience",
    "Total_Active_Min", "Weekly_Active_Hours", "Protein_Per_KG",
    "Calorie_Deficit", "Cardio_Ratio"
]

NUMERIC_FEATURES = [
    "Age", "Height_cm", "Initial_Weight_kg", "BMI",
    "Workout_Days_Per_Week", "Workout_Duration_Min",
    "Cardio_Minutes", "Strength_Minutes", "Sleep_Hours",
    "Water_Intake_L", "Daily_Steps", "Protein_Intake_g",
    "Daily_Calories", "Stress_Level",
    "Total_Active_Min", "Weekly_Active_Hours", "Protein_Per_KG",
    "Calorie_Deficit", "Cardio_Ratio"
]
CATEGORICAL_FEATURES = ["Gender", "Training_Experience"]

X = df[FEATURES]
y_weight  = df["Weight_After_90_Days"]
y_fat     = df["BodyFat_After_90_Days"]
y_goal    = (df["Goal_Achieved"] == "Yes").astype(int)

X_tr, X_te, yw_tr, yw_te, yf_tr, yf_te, yg_tr, yg_te = train_test_split(
    X, y_weight, y_fat, y_goal, test_size=0.15, random_state=42
)

# ─── Pre-processor ───────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
])

# ─── Helper ──────────────────────────────────────────────────────────────────
def build_and_evaluate(name, pipe, X_tr, y_tr, X_te, y_te, task="reg"):
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    if task == "reg":
        mae = mean_absolute_error(y_te, preds)
        r2  = r2_score(y_te, preds)
        print(f"  [{name}]  MAE={mae:.3f}  R²={r2:.4f}")
    else:
        proba = pipe.predict_proba(X_te)[:,1]
        acc   = accuracy_score(y_te, preds)
        auc   = roc_auc_score(y_te, proba)
        print(f"  [{name}]  Accuracy={acc:.4f}  AUC={auc:.4f}")
    return pipe

# ─── 1. Weight Model ─────────────────────────────────────────────────────────
print("\n--- Weight After 90 Days ---")
weight_pipe = Pipeline([
    ("pre", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.08, max_depth=5,
        subsample=0.85, min_samples_leaf=10, random_state=42
    ))
])
weight_pipe = build_and_evaluate("GBR-Weight", weight_pipe, X_tr, yw_tr, X_te, yw_te)
joblib.dump(weight_pipe, os.path.join(MODEL_DIR, "weight_model.pkl"))

# ─── 2. Body Fat Model ───────────────────────────────────────────────────────
print("\n--- Body Fat After 90 Days ---")
fat_pipe = Pipeline([
    ("pre", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.08, max_depth=5,
        subsample=0.85, min_samples_leaf=10, random_state=42
    ))
])
fat_pipe = build_and_evaluate("GBR-BodyFat", fat_pipe, X_tr, yf_tr, X_te, yf_te)
joblib.dump(fat_pipe, os.path.join(MODEL_DIR, "fat_model.pkl"))

# ─── 3. Goal Achievement Model ───────────────────────────────────────────────
print("\n--- Goal Achievement Probability ---")
goal_pipe = Pipeline([
    ("pre", preprocessor),
    ("model", GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=5,
        subsample=0.85, min_samples_leaf=10, random_state=42
    ))
])
goal_pipe = build_and_evaluate("GBC-Goal", goal_pipe, X_tr, yg_tr, X_te, yg_te, task="cls")
joblib.dump(goal_pipe, os.path.join(MODEL_DIR, "goal_model.pkl"))

# Save feature list for app.py
joblib.dump(FEATURES, os.path.join(MODEL_DIR, "feature_list.pkl"))
print("\n  All models saved to ./models/")

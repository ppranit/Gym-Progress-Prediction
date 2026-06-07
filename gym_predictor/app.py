"""
app.py  –  Gym Progress Prediction Flask Application
Run:  python app.py  then open  http://127.0.0.1:5000
"""

import os, joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# ─── Load Models ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, "models")

weight_model = joblib.load(os.path.join(MODELS_DIR, "weight_model.pkl"))
fat_model    = joblib.load(os.path.join(MODELS_DIR, "fat_model.pkl"))
goal_model   = joblib.load(os.path.join(MODELS_DIR, "goal_model.pkl"))

# ─── Feature Engineering (mirrors train_model.py) ────────────────────────────
def build_input_df(form):
    age          = float(form["age"])
    gender       = form["gender"]
    height       = float(form["height"])
    weight       = float(form["initial_weight"])
    bmi          = weight / ((height / 100) ** 2)
    workout_days = int(form["workout_days"])
    workout_dur  = int(form["workout_duration"])
    cardio       = int(form["cardio_minutes"])
    strength     = int(form["strength_minutes"])
    sleep        = float(form["sleep_hours"])
    water        = float(form["water_intake"])
    steps        = int(form["daily_steps"])
    protein      = int(form["protein_intake"])
    calories     = int(form["daily_calories"])
    stress       = int(form["stress_level"])
    experience   = form["training_experience"]

    total_active     = cardio + strength
    weekly_hours     = (workout_days * workout_dur) / 60.0
    protein_per_kg   = protein / (weight + 1e-6)
    calorie_deficit  = calories - 2000
    cardio_ratio     = cardio / (total_active + 1e-6)

    row = {
        "Age": age, "Gender": gender, "Height_cm": height,
        "Initial_Weight_kg": weight, "BMI": bmi,
        "Workout_Days_Per_Week": workout_days,
        "Workout_Duration_Min": workout_dur,
        "Cardio_Minutes": cardio, "Strength_Minutes": strength,
        "Sleep_Hours": sleep, "Water_Intake_L": water,
        "Daily_Steps": steps, "Protein_Intake_g": protein,
        "Daily_Calories": calories, "Stress_Level": stress,
        "Training_Experience": experience,
        "Total_Active_Min": total_active,
        "Weekly_Active_Hours": weekly_hours,
        "Protein_Per_KG": protein_per_kg,
        "Calorie_Deficit": calorie_deficit,
        "Cardio_Ratio": cardio_ratio
    }
    return pd.DataFrame([row])

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        X = build_input_df(request.form)

        pred_weight  = float(weight_model.predict(X)[0])
        pred_fat     = float(fat_model.predict(X)[0])
        goal_prob    = float(goal_model.predict_proba(X)[0][1]) * 100

        initial_weight = float(request.form["initial_weight"])
        weight_change  = pred_weight - initial_weight
        weight_change_str = f"{'+' if weight_change >= 0 else ''}{weight_change:.1f} kg"

        # Classification label
        if goal_prob >= 70:
            goal_label = "High"
            goal_color = "high"
        elif goal_prob >= 40:
            goal_label = "Moderate"
            goal_color = "moderate"
        else:
            goal_label = "Low"
            goal_color = "low"

        results = {
            "weight_after": round(pred_weight, 1),
            "fat_after": round(pred_fat, 1),
            "goal_prob": round(goal_prob, 1),
            "weight_change": weight_change_str,
            "weight_change_val": round(weight_change, 1),
            "goal_label": goal_label,
            "goal_color": goal_color,
            "initial_weight": initial_weight,
            # echo form back for display
            "name": request.form.get("name", "Athlete"),
            "gender": request.form["gender"],
            "age": request.form["age"],
            "experience": request.form["training_experience"],
        }

        # Tips
        tips = []
        sleep = float(request.form["sleep_hours"])
        stress = int(request.form["stress_level"])
        protein = int(request.form["protein_intake"])
        water = float(request.form["water_intake"])
        cardio = int(request.form["cardio_minutes"])
        workout_days = int(request.form["workout_days"])

        if sleep < 7:
            tips.append(" Aim for 7–9 hours of sleep to maximise muscle recovery and fat loss.")
        if stress > 6:
            tips.append(" High stress elevates cortisol — consider meditation or relaxation techniques.")
        if protein < 100:
            tips.append(" Increase protein intake to at least 1.6 g/kg body weight for better muscle retention.")
        if water < 2.5:
            tips.append(" Hydrate well — target 3–4 litres per day for optimal metabolism.")
        if cardio < 20:
            tips.append(" Add more cardio sessions to boost calorie burn and cardiovascular health.")
        if workout_days < 3:
            tips.append(" Training at least 3-4 days per week significantly improves body composition.")
        if not tips:
            tips.append(" Your lifestyle stats look great — stay consistent and trust the process!")

        results["tips"] = tips

        return render_template("result.html", **results)

    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=5000)

═══════════════════════════════════════════════════════════════════
   GymAI – 90-Day Gym Progress Predictor
   Machine Learning · Flask · HTML/CSS
═══════════════════════════════════════════════════════════════════

WHAT THIS PROJECT DOES
──────────────────────
Uses a Gradient Boosting ML model trained on 30,000 gym athletes to
predict, after 90 days of training:

  1. Weight After 90 Days        (Regression – R² 0.989)
  2. Body Fat % After 90 Days    (Regression – R² 0.576)
  3. Goal Achievement Probability (Classification – AUC 0.950)

A beautiful dark gym-themed web UI (HTML + CSS, no JS) is connected
to a Flask backend that runs the models and renders personalised
results + AI recommendations.


REQUIREMENTS
────────────
  • Python 3.8 or higher
  • pip (comes with Python)
  • Internet not required after setup


FOLDER STRUCTURE
────────────────
  gym_predictor/
  ├── app.py                               ← Flask web server
  ├── train_model.py                       ← ML training script
  ├── gym_progress_prediction_dataset_30000.csv
  ├── requirements.txt
  ├── models/                              ← saved ML models (auto-created)
  │   ├── weight_model.pkl
  │   ├── fat_model.pkl
  │   └── goal_model.pkl
  ├── templates/
  │   ├── index.html                       ← Input form page
  │   └── result.html                      ← Prediction result page
  └── static/
      └── css/
          └── style.css                    ← All styling


STEP-BY-STEP SETUP
──────────────────

STEP 1 – Open a terminal / command prompt
  Windows : press Win+R, type cmd, press Enter
  macOS   : open Terminal (Applications → Utilities → Terminal)
  Linux   : open your preferred terminal emulator

STEP 2 – Navigate to the project folder
  cd path/to/gym_predictor

  Example (Windows):  cd C:\Users\YourName\Downloads\gym_predictor
  Example (macOS):    cd ~/Downloads/gym_predictor

STEP 3 – (Recommended) Create a virtual environment
  python -m venv venv

  Activate it:
    Windows:  venv\Scripts\activate
    macOS/Linux:  source venv/bin/activate

STEP 4 – Install required packages
  pip install -r requirements.txt

STEP 5 – Train the ML models (one-time step)
  python train_model.py

  You will see output like:
    ── Weight After 90 Days ──
      [GBR-Weight]  MAE=1.240  R²=0.9891
    ── Body Fat After 90 Days ──
      [GBR-BodyFat]  MAE=1.753  R²=0.5763
    ── Goal Achievement Probability ──
      [GBC-Goal]  Accuracy=0.8758  AUC=0.9502
    ✅  All models saved to ./models/

  Three .pkl model files will appear in the models/ folder.

STEP 6 – Start the Flask web server
  python app.py

  You will see:
    * Running on http://127.0.0.1:5000

STEP 7 – Open your browser
  Go to:  http://127.0.0.1:5000

  Fill in your details in the form and click
  "Predict My 90-Day Progress" to see your results!


WHAT EACH FILE DOES
───────────────────
  app.py           Flask server. Loads models, processes the form,
                   runs predictions, renders result page.

  train_model.py   Reads the CSV, engineers features, trains 3 Gradient
                   Boosting models, saves them as .pkl files.

  index.html       Input form with 4 sections:
                     · Personal Details
                     · Workout Profile
                     · Nutrition & Hydration
                     · Lifestyle Factors

  result.html      Results page showing:
                     · KPI cards (Weight, Body Fat, Goal %)
                     · Progress ring chart (CSS/SVG)
                     · Body fat reference table
                     · Personalised AI tip cards

  style.css        Full dark gym theme (obsidian black + firestorm
                   orange + neon lime). Fully responsive.


ML ALGORITHM DETAILS
────────────────────
  Algorithm   : scikit-learn GradientBoostingRegressor /
                GradientBoostingClassifier
  Estimators  : 300 trees
  Max depth   : 5
  Learning rate: 0.08
  Subsample   : 0.85
  Pre-processing: StandardScaler + OneHotEncoder (ColumnTransformer)

  Feature engineering adds:
    · Total Active Minutes (cardio + strength)
    · Weekly Active Hours
    · Protein per kg body weight
    · Calorie deficit from TDEE estimate
    · Cardio-to-total ratio


INPUT FIELDS EXPLAINED
──────────────────────
  Age                     Your age in years (16–80)
  Gender                  Male / Female
  Height (cm)             Your height in centimetres
  Current Weight (kg)     Your weight today
  Training Experience     Beginner / Intermediate / Advanced
  Workout Days/Week       How many days you train (1–7)
  Session Duration (min)  How long each session lasts
  Cardio per Session (min) Minutes of cardio per session
  Strength per Session    Minutes of weight/resistance training
  Daily Steps             Average daily step count
  Daily Calories (kcal)   Average daily calorie intake
  Protein Intake (g)      Daily protein in grams
  Water Intake (L)        Daily water consumption in litres
  Sleep Duration (hrs)    Average nightly sleep
  Stress Level (1–10)     1 = very calm, 10 = extremely stressed


TROUBLESHOOTING
───────────────
  Q: "ModuleNotFoundError: No module named 'flask'"
  A: Run:  pip install -r requirements.txt

  Q: "FileNotFoundError: models/weight_model.pkl"
  A: Run train_model.py first (Step 5 above).

  Q: "Address already in use" on port 5000
  A: Change port in app.py last line:  app.run(debug=True, port=5001)
     Then open http://127.0.0.1:5001

  Q: "python: command not found"
  A: Try "python3" instead of "python" on macOS/Linux.


STOPPING THE SERVER
───────────────────
  Press  Ctrl + C  in the terminal to stop Flask.


═══════════════════════════════════════════════════════════════════
   Built with scikit-learn · Flask · HTML · CSS
   Dataset: 30,000 gym athletes · 19 features
═══════════════════════════════════════════════════════════════════

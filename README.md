# Lifestyle and Clinical Factors in Stroke Risk Prediction

1. Install Python dependencies

```bash
pip install -r requirements.txt
```

2. Install frontend dependencies

```bash
cd frontend
npm install
```

3. Build the frontend

```bash
npm run build
```

4. Run the backend

```bash
cd ..
python backend/app.py
```

Open http://localhost:5000/ to confirm the app is running.

## Project structure

```
AI-in-Healthcare/
├── ai_module/                          # Core ML pipeline
│   ├── main.ipynb                      # Main EDA and preprocessing notebook
│   ├── train_data_balanced_models.ipynb # Training runs with balancing strategies
│   ├── automatic_fine_tuning.ipynb     # Hyperparameter tuning experiments
│   │
│   ├── feature_groups.py               # Defines clinical, lifestyle, and combined feature sets
│   ├── feature_analysis.py             # Cramér's V association analysis between features
│   │
│   ├── balancing.py                    # Random oversampling (minority class resampling)
│   ├── weighted.py                     # Class-weight balancing (no resampling)
│   ├── smote.py                        # SMOTE synthetic oversampling
│   ├── SMOTENC.py                      # SMOTE-NC for mixed categorical/numerical data
│   ├── smote-tomek.py                  # SMOTE + Tomek links hybrid cleaning
│   │
│   ├── counterfactual_sweep.py         # Per-patient risk change under lifestyle interventions
│   ├── cf_visualize.py                 # Counterfactual result charts
│   ├── cf_visualize_summary.py         # Summary-level counterfactual visualisation
│   │
│   ├── models/                         # Serialised trained models (.pkl)
│   │   └── <algo>_weighted_<feature_set>_<uncertain>__final_<variant>.pkl
│   │       # algo: lightgbm | random_forest | xgboost
│   │       # feature_set: clinical | lifestyle | combined
│   │       # uncertain: with_uncertain | without_uncertain
│   │       # variant: best_balance | low_false_negative
│   │
│   ├── cf_results/                     # Counterfactual analysis outputs
│   │   ├── counterfactual_chart.png
│   │   ├── counterfactual_summary_chart.png
│   │   └── summary.xlsx
│   │
│   ├── balanced_training_cache/        # Cached cross-validation folds (joblib)
│   ├── brfss_2024.zip                  # Raw BRFSS 2024 survey dataset
│   ├── 2024-calculated-variables-version4-508.pdf  # BRFSS codebook
│   ├── Data Balanced Models Overview.xlsx          # Results comparison spreadsheet
│   └── Model Stats Overview.xlsx                   # Per-model metric summary
│
├── backend/                            # Flask API server
│   ├── app.py                          # REST endpoints (predict, train, export, logging)
│   ├── training.py                     # Model training logic with CV and caching
│   ├── registry.py                     # SQLite model registry and dataset loader
│   ├── notebook_training.py            # Notebook-compatible training entry point
│   ├── datasets/                       # Preprocessed feature-set parquet files
│   │   ├── clinical_with_uncertain.parquet
│   │   ├── clinical_without_uncertain.parquet
│   │   ├── lifestyle_with_uncertain.parquet
│   │   ├── lifestyle_without_uncertain.parquet
│   │   ├── combined_with_uncertain.parquet
│   │   └── combined_without_uncertain.parquet
│   ├── healthcare.db                   # SQLite database (model registry + prediction log)
│   └── prediction_log.txt              # Raw prediction audit log
│
├── frontend/                           # React + TypeScript UI (Vite)
│   └── src/
│       ├── App.tsx                     # Routing and layout
│       ├── api.ts                      # Backend API client
│       ├── modelData.ts                # Static model metadata
│       ├── modelMetadata.ts            # Feature-set and variant descriptions
│       ├── modelScoring.ts             # Client-side score interpretation
│       ├── pages/
│       │   ├── Predict.tsx             # Patient risk prediction form
│       │   ├── Patients.tsx            # Patient history and audit log
│       │   ├── ModelDetail.tsx         # Single-model metrics view
│       │   ├── ModelComparison.tsx     # Side-by-side model comparison
│       │   └── ModelCompare.tsx        # Interactive model compare tool
│       └── components/
│           └── SectionCard.tsx         # Reusable card layout component
│
├── requirements.txt                    # Python dependencies
└── USCODE24_LLCP_082125.md            # BRFSS variable reference
```

### Feature sets

| Set | Description |
|---|---|
| `clinical` | Clinical risk factors (age, BMI, blood pressure, diabetes, etc.) |
| `lifestyle` | Behavioural factors (smoking, alcohol, exercise, diet, etc.) |
| `combined` | Union of clinical and lifestyle features |

Each set is trained in two variants: **with_uncertain** (keeps ambiguous survey responses) and **without_uncertain** (excludes them).

### Model variants

| Variant | Optimisation goal |
|---|---|
| `best_balance` | Best F1 / balanced accuracy across classes |
| `low_false_negative` | Minimise false negatives (higher recall for stroke) |

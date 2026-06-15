# Lifestyle and Clinical Factors in Stroke Risk Prediction

1. Install Python dependencies

```bash
pip install -r requirements.txt
```

GPU training is enabled in `ai_module/train_data_balanced_models.ipynb`,
`ai_module/train_models.ipynb`, `ai_module/automatic_fine_tuning.ipynb`,
and the backend fine-tuning API
for XGBoost and LightGBM. Random Forest always uses scikit-learn's
RandomForestClassifier on CPU.
If GPU training fails, check that your NVIDIA drivers/CUDA setup are available
for XGBoost and that LightGBM was installed with GPU support. Existing saved
models are reused by default, so enable force retraining when replacing old
models.

2. Run the data preparation notebook

Open `ai_module/main.ipynb` in VS Code or Jupyter and run all cells from top to bottom or use:

```bash
jupyter nbconvert --to script main.ipynb --stdout | python
```

3. Train data-balance comparison models

Open `ai_module/train_data_balanced_models.ipynb` and run all cells. This
notebook registers random oversampling, SMOTE, SMOTENC, and class-weighted
model sets for each feature/uncertainty dataset variant, and stores those
model files in `ai_module/data-balance-models`.

4. Train the final selected models

Open `ai_module/train_models.ipynb` and run all cells. This notebook trains
the selected weighted models for clinical, lifestyle, and combined feature
sets with and without uncertain features. These model files are stored in
`ai_module/models`.

5. Run automatic fine-tuning when needed

Open `ai_module/automatic_fine_tuning.ipynb`, choose a normal baseline model,
and run the automatic parameter grid. The web interface no longer starts
automatic fine-tuning.

6. Install frontend dependencies (once)

```bash
cd frontend
npm install
```

The frontend is a Vite + React + TypeScript app; Node.js is only needed for installing and building it.

7. Build the frontend

```bash
npm run build
```

8. Run the backend

```bash
cd ..
python backend/app.py
```

Open http://localhost:5000/ to confirm the app is running.

## Project structure

- `backend/app.py` - Flask server
- `backend/notebook_training.py` - notebook training workflows
- `ai_module/` - dataset preparation and model training notebooks
- `frontend/` - React + TypeScript frontend
- `requirements.txt` - Python dependencies

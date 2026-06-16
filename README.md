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

- `backend/app.py` - Flask server
- `backend/notebook_training.py` - notebook training workflows
- `ai_module/` - dataset preparation and model training notebooks
- `frontend/` - React + TypeScript frontend
- `requirements.txt` - Python dependencies

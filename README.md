# Lifestyle and Clinical Factors in Stroke Risk Prediction

1. Install Python dependencies

```bash
pip install -r requirements.txt
```

2. Run the notebook to generate the datasets and models

Open `ai_module/main.ipynb` in VS Code or Jupyter and run all cells from top to bottom.

3. Install frontend dependencies (once)

```bash
cd frontend
npm install
```

The frontend is a Vite + React + TypeScript app; Node.js is only needed for installing and building it.

4. Build the frontend

```bash
npm run build
```

5. Run the backend

```bash
cd ..
python backend/app.py
```

Open http://localhost:5000/ to confirm the app is running.

## Project structure

- `backend/app.py` - Flask server
- `ai_module/` - dataset preperation and model training
- `frontend/` - React + TypeScript frontend
- `requirements.txt` - Python dependencies

# AI in Healthcare

## Setup Instructions

### Install Dependencies

To install all required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

This will install Flask and any other dependencies listed in `requirements.txt`.

For the frontend, install the React dependencies once:

```bash
cd frontend
npm install
```

### Run the Application

Build the React app first:

```bash
cd frontend
npm run build
```

Then start the Flask server from the project root:

```bash
python backend/app.py
```

Open `http://localhost:5000/` in your browser.

## Project Structure

- `backend/app.py` - Flask application and React SPA host on port 5000
- `ai_module/main.ipynb`
- `frontend/` - React + TypeScript + Tailwind app used to generate the build served by Flask
- `requirements.txt` - Python dependencies

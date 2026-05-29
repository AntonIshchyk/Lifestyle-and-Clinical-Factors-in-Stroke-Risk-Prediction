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
npm run build
```

Then start the Flask development server from the project root:

```bash
python backend/app.py
```

The Flask app serves the React build at `http://localhost:5000/`.

## Project Structure

- `backend/app.py` - Flask application and React SPA host
- `ai_module/main.ipynb`
- `frontend/` - React + TypeScript + Tailwind app
- `requirements.txt` - Python dependencies

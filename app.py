from pathlib import Path
import json
import pickle
import sqlite3

import pandas as pd
from flask import Flask, render_template, request


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = BASE_DIR / "main.ipynb"
DB_PATH = BASE_DIR / "healthcare.db"
MODEL_PATHS = [BASE_DIR / "model.pkl", BASE_DIR / "model.joblib"]
PATIENTS_TABLE = "patients"
PATIENT_ID_COLUMN = "patient_id"
TARGET_COLUMN = "CVDSTRK3"
PREDICTION_COLUMN = "stroke_prediction"
PROBABILITY_COLUMN = "stroke_probability"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def table_exists(connection):
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (PATIENTS_TABLE,),
    ).fetchone()
    return row is not None


def get_columns(connection):
    rows = connection.execute(f'PRAGMA table_info("{PATIENTS_TABLE}")').fetchall()
    return [row["name"] for row in rows]


def load_model():
    for model_path in MODEL_PATHS:
        if not model_path.exists():
            continue
        if model_path.suffix == ".joblib":
            import joblib

            return joblib.load(model_path)
        with model_path.open("rb") as file:
            return pickle.load(file)
    return None


def model_features(model, dataframe):
    if hasattr(model, "feature_names_in_"):
        return [column for column in model.feature_names_in_ if column in dataframe.columns]
    ignored = {PATIENT_ID_COLUMN, TARGET_COLUMN, PREDICTION_COLUMN, PROBABILITY_COLUMN}
    return [column for column in dataframe.columns if column not in ignored]


def add_model_results(dataframe, model):
    if model is None or dataframe.empty:
        return dataframe

    features = model_features(model, dataframe)
    if not features:
        return dataframe

    predictions = model.predict(dataframe[features])
    dataframe[PREDICTION_COLUMN] = predictions

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(dataframe[features])
        positive_class_index = 1 if probabilities.shape[1] > 1 else 0
        dataframe[PROBABILITY_COLUMN] = probabilities[:, positive_class_index]

    return dataframe


def load_clean_df_from_notebook():
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Missing notebook: {NOTEBOOK_PATH}")

    namespace = {"__name__": "__notebook__", "__file__": str(NOTEBOOK_PATH)}
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        if "find_high_correlation_pairs" in source:
            continue

        exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)
        if "clean_df" in namespace:
            clean_df = namespace["clean_df"]
            if not isinstance(clean_df, pd.DataFrame):
                raise TypeError("main.ipynb defines clean_df, but it is not a pandas DataFrame.")
            return clean_df.copy()

    raise RuntimeError("main.ipynb must define a pandas DataFrame named clean_df.")


def prepare_patients_dataframe():
    dataframe = load_clean_df_from_notebook().reset_index(drop=True)
    dataframe.insert(0, PATIENT_ID_COLUMN, range(1, len(dataframe) + 1))
    return add_model_results(dataframe, load_model())


def rebuild_database():
    dataframe = prepare_patients_dataframe()

    with get_connection() as connection:
        connection.execute(f'DROP TABLE IF EXISTS "{PATIENTS_TABLE}"')
        dataframe.to_sql(PATIENTS_TABLE, connection, if_exists="replace", index=False)
        connection.execute(
            f'CREATE INDEX IF NOT EXISTS idx_patients_patient_id ON "{PATIENTS_TABLE}" ("{PATIENT_ID_COLUMN}")'
        )
        connection.commit()


def ensure_database():
    with get_connection() as connection:
        if table_exists(connection):
            return
    rebuild_database()


def display_value(value):
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4g}"
    return str(value)


def get_patients(page, per_page, search):
    offset = (page - 1) * per_page
    where = ""
    params = []

    if search:
        where = f'WHERE CAST("{PATIENT_ID_COLUMN}" AS TEXT) LIKE ?'
        params.append(f"%{search}%")

    with get_connection() as connection:
        columns = get_columns(connection)
        total = connection.execute(
            f'SELECT COUNT(*) FROM "{PATIENTS_TABLE}" {where}',
            params,
        ).fetchone()[0]
        rows = connection.execute(
            f'''
            SELECT *
            FROM "{PATIENTS_TABLE}"
            {where}
            ORDER BY "{PATIENT_ID_COLUMN}"
            LIMIT ? OFFSET ?
            ''',
            [*params, per_page, offset],
        ).fetchall()

    patients = [
        {column: display_value(row[column]) for column in columns}
        for row in rows
    ]
    return columns, patients, total


@app.route("/")
def index():
    ensure_database()

    page = max(request.args.get("page", default=1, type=int), 1)
    per_page = min(max(request.args.get("per_page", default=25, type=int), 10), 100)
    search = request.args.get("search", "", type=str).strip()

    columns, patients, total = get_patients(page, per_page, search)
    total_pages = max((total + per_page - 1) // per_page, 1)

    if page > total_pages:
        page = total_pages
        columns, patients, total = get_patients(page, per_page, search)

    return render_template(
        "index.html",
        columns=columns,
        patients=patients,
        page=page,
        per_page=per_page,
        search=search,
        total=total,
        total_pages=total_pages,
        has_model=any(path.exists() for path in MODEL_PATHS),
    )


@app.route("/predict")
def predict():
    return render_template("placeholder.html", title="Predict")


@app.route("/models-comparison")
def models_comparison():
    return render_template("placeholder.html", title="Models Comparison")


@app.cli.command("rebuild-db")
def rebuild_db_command():
    rebuild_database()
    print(f"Rebuilt {DB_PATH}")


if __name__ == "__main__":
    ensure_database()
    app.run(debug=True)

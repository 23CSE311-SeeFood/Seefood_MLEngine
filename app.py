import json
import os
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from ml_pipeline import SalesMLService

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

app = Flask(__name__)

allowed_origins = os.environ.get(
    "CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000"
).split(",")
CORS(app, origins=[origin.strip() for origin in allowed_origins if origin.strip()], supports_credentials=True)
ml_service = SalesMLService(BASE_DIR)


def load_json(filename, default):
    path = DATA_DIR / filename
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_azure_prediction():
    scoring_uri = os.environ.get("AZURE_ML_SCORING_URI", "").strip()
    api_key = os.environ.get("AZURE_ML_API_KEY", "").strip()
    deployment_name = os.environ.get("AZURE_ML_DEPLOYMENT", "").strip()

    if not scoring_uri or not api_key:
        raise RuntimeError("AZURE_ML_SCORING_URI or AZURE_ML_API_KEY missing")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if deployment_name:
        headers["azureml-model-deployment"] = deployment_name

    payload = json.dumps({"mode": "next7"}).encode("utf-8")
    request_obj = Request(scoring_uri, data=payload, headers=headers, method="POST")

    with urlopen(request_obj, timeout=30) as response:
        response_text = response.read().decode("utf-8")

    parsed = json.loads(response_text)
    if isinstance(parsed, str):
        parsed = json.loads(parsed)
    if not isinstance(parsed, dict):
        raise RuntimeError("Unexpected Azure ML response format")
    return parsed


@app.get("/")
def root():
    return jsonify({"status": "ok", "service": "seefood-ml-api", "modelPath": str(ml_service.model_path)})


@app.get("/health")
def health():
    return "OK", 200


@app.get("/sales/report")
def sales_report():
    try:
        return jsonify(ml_service.sales_report())
    except Exception:
        return jsonify(load_json("sales-data.json", {}))


@app.get("/sales/predictions")
def sales_predictions():
    try:
        azure_payload = _resolve_azure_prediction()
        azure_payload.setdefault("meta", {})
        azure_payload["meta"]["source"] = "azure-ml-endpoint"
        return jsonify(azure_payload)
    except (RuntimeError, ValueError, URLError, HTTPError):
        try:
            local_payload = ml_service.predictions()
            local_payload.setdefault("meta", {})
            local_payload["meta"]["source"] = "local-fallback"
            return jsonify(local_payload)
        except Exception:
            fallback = load_json("prediction-data.json", {})
            if isinstance(fallback, dict):
                fallback.setdefault("meta", {})
                fallback["meta"]["source"] = "static-fallback"
            return jsonify(fallback)


@app.post("/ml/train")
def train_model():
    payload = request.get_json(silent=True) or {}
    force = bool(payload.get("force", True))
    if force:
        artifact = ml_service.retrain()
    else:
        artifact = ml_service._load_artifact()
    return jsonify(
        {
            "trainedAt": artifact.get("trained_at"),
            "trainingRows": artifact.get("training_rows", {}),
            "realDataInfo": artifact.get("real_data_info", {}),
            "modelPath": str(ml_service.model_path),
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)

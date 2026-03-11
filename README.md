# seeFood Flask ML API

Flask backend for the dashboard sales/prediction pages with real ML training:
- Stage 1: pretrain on synthetic sales data
- Stage 2: fine-tune on real order logs from PostgreSQL (`Order`, `OrderItem`, `CanteenItem`)

## Setup

```bash
cd /Users/vivekchitturi/Desktop/combine/flask
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Database Configuration

`DATABASE_URL` is read in this order:
1. `flask/.env`
2. `seeback/.env`
3. process environment variable

If DB is unavailable, the API falls back to JSON in `flask/data/`.

## Azure Prediction Endpoint Configuration

To make `/sales/predictions` use Azure ML endpoint inference:

- `AZURE_ML_SCORING_URI` = endpoint scoring URL
- `AZURE_ML_API_KEY` = endpoint key
- `AZURE_ML_DEPLOYMENT` (optional) = deployment name, e.g. `blue`

Fallback order for `/sales/predictions`:
1. Azure endpoint
2. Local model inference
3. Static JSON in `flask/data/prediction-data.json`

## Training

Trigger training manually:

```bash
curl -X POST http://localhost:5001/ml/train -H "Content-Type: application/json" -d '{"force": true}'
```

Training always starts from synthetic data and then applies real DB logs (if present).

## Run

```bash
python app.py
```

Default port is `5001`. Override with `PORT=5001` in env.

## Useful Endpoints

- `GET /sales/report`
- `GET /sales/predictions`
- `POST /ml/train`

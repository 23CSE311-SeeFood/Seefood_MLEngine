import json
import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

ARTIFACT = None


def _load_artifact(model_dir):
    for root, _, files in os.walk(model_dir):
        for name in files:
            if name.endswith('.joblib'):
                return joblib.load(os.path.join(root, name))
    raise FileNotFoundError(f'No .joblib model found under {model_dir}')


def _feature_frame(dates):
    dt = pd.to_datetime(pd.Series(dates)).dt.normalize()
    ordinal = dt.map(pd.Timestamp.toordinal).astype(int)
    return pd.DataFrame(
        {
            'dow': dt.dt.dayofweek.astype(int),
            'month': dt.dt.month.astype(int),
            'day': dt.dt.day.astype(int),
            'week': dt.dt.isocalendar().week.astype(int),
            'is_weekend': (dt.dt.dayofweek >= 5).astype(int),
            'doy_sin': np.sin(2 * np.pi * dt.dt.dayofyear / 365.25),
            'doy_cos': np.cos(2 * np.pi * dt.dt.dayofyear / 365.25),
            't': ordinal.astype(int),
        }
    )


def init():
    global ARTIFACT
    model_dir = os.getenv('AZUREML_MODEL_DIR', '.')
    ARTIFACT = _load_artifact(model_dir)


def run(raw_data):
    global ARTIFACT
    if ARTIFACT is None:
        raise RuntimeError('Model artifact not initialized')

    payload = json.loads(raw_data) if raw_data else {}
    mode = payload.get('mode', 'next7')

    if mode == 'raw':
        # Direct inference mode: pass an explicit feature matrix in payload["input"]
        matrix = np.array(payload.get('input', []), dtype=float)
        preds = ARTIFACT['sales_model'].predict(matrix)
        return {'prediction': preds.tolist()}

    # Dashboard mode: generate next 7-day forecast + category forecast
    today = datetime.utcnow().date()
    future_dates = [today + timedelta(days=i) for i in range(1, 8)]

    sales_features = _feature_frame(future_dates)
    daily_forecast = ARTIFACT['sales_model'].predict(sales_features)
    daily_forecast = np.maximum(0, daily_forecast)

    next7 = []
    prev = daily_forecast[0]
    for idx, (date_val, predicted) in enumerate(zip(future_dates, daily_forecast)):
        trend = 'stable'
        if idx > 0:
            if predicted > prev * 1.02:
                trend = 'up'
            elif predicted < prev * 0.98:
                trend = 'down'
        next7.append(
            {
                'day': pd.Timestamp(date_val).strftime('%a'),
                'predicted': int(round(float(predicted))),
                'trend': trend,
            }
        )
        prev = predicted

    categories = ARTIFACT.get('categories', ['Breakfast', 'Lunch', 'Snacks', 'Dinner', 'Beverages'])
    categories = list(dict.fromkeys(categories))[:5]

    category_rows = []
    for category in categories:
        cat_features = _feature_frame(future_dates)
        cat_features['category'] = category
        predicted_values = ARTIFACT['category_model'].predict(cat_features)
        predicted_daily = float(np.maximum(0, np.mean(predicted_values)))
        category_rows.append(
            {
                'category': category,
                'predicted': int(round(predicted_daily)),
                'current': int(round(max(1.0, predicted_daily * 0.9))),
            }
        )

    category_rows = sorted(category_rows, key=lambda x: x['current'], reverse=True)

    return {
        'next7Days': next7,
        'categoryPrediction': category_rows,
        'meta': {
            'trainedAt': ARTIFACT.get('trained_at'),
            'source': 'azure-ml-endpoint',
        },
    }

import json
import os
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import joblib
import numpy as np
import pandas as pd
from dotenv import dotenv_values
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import create_engine, text


DEFAULT_CATEGORIES = ["Breakfast", "Lunch", "Snacks", "Dinner", "Beverages"]
CATEGORY_MAP = {
    "RICE": "Lunch",
    "CURRIES": "Dinner",
    "ICECREAM": "Snacks",
    "ROOTI": "Dinner",
    "DRINKS": "Beverages",
    "OTHER": "Snacks",
}


class SalesMLService:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir).resolve()
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_dir / "sales_forecaster.joblib"

    def _load_json(self, filename, default):
        path = self.data_dir / filename
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _clean_env_value(self, value):
        if not value:
            return None
        return str(value).strip().strip('"').strip("'")

    def _resolve_database_url(self):
        from_env = self._clean_env_value(os.getenv("DATABASE_URL"))
        if from_env:
            return from_env

        local_env = self.base_dir / ".env"
        if local_env.exists():
            env_values = dotenv_values(local_env)
            local_value = self._clean_env_value(env_values.get("DATABASE_URL"))
            if local_value:
                return local_value

        sibling_env = self.base_dir.parent / "seeback" / ".env"
        if sibling_env.exists():
            env_values = dotenv_values(sibling_env)
            sibling_value = self._clean_env_value(env_values.get("DATABASE_URL"))
            if sibling_value:
                return sibling_value

        return None

    def _normalize_database_url(self, db_url):
        if not db_url:
            return db_url
        normalized = db_url
        if normalized.startswith("postgres://"):
            normalized = normalized.replace("postgres://", "postgresql://", 1)
        parsed = urlsplit(normalized)
        if parsed.scheme.startswith("postgres"):
            filtered_query = [
                (key, value)
                for key, value in parse_qsl(parsed.query, keep_blank_values=True)
                if key.lower() != "schema"
            ]
            return urlunsplit(
                (
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path,
                    urlencode(filtered_query),
                    parsed.fragment,
                )
            )
        return normalized

    def _query_real_logs(self, lookback_days=365):
        db_url = self._resolve_database_url()
        if not db_url:
            return pd.DataFrame(), pd.DataFrame(), {"used_real_data": False, "reason": "DATABASE_URL missing"}
        db_url = self._normalize_database_url(db_url)

        try:
            engine = create_engine(db_url, pool_pre_ping=True)
            order_query = text(
                """
                SELECT
                  o."createdAt" AS created_at,
                  o.total AS total
                FROM "Order" o
                WHERE o.status IN ('PAID', 'READY', 'DELIVERED')
                  AND o."createdAt" >= NOW() - (:lookback || ' days')::interval
                """
            )
            category_query = text(
                """
                SELECT
                  o."createdAt" AS created_at,
                  COALESCE(ci.category::text, 'OTHER') AS category,
                  oi.total AS total
                FROM "Order" o
                JOIN "OrderItem" oi ON oi."orderId" = o.id
                JOIN "CanteenItem" ci ON ci.id = oi."canteenItemId"
                WHERE o.status IN ('PAID', 'READY', 'DELIVERED')
                  AND o."createdAt" >= NOW() - (:lookback || ' days')::interval
                """
            )
            with engine.connect() as conn:
                orders_df = pd.read_sql(order_query, conn, params={"lookback": str(int(lookback_days))})
                category_df = pd.read_sql(category_query, conn, params={"lookback": str(int(lookback_days))})
            engine.dispose()

            if not orders_df.empty:
                orders_df["created_at"] = pd.to_datetime(orders_df["created_at"], utc=True).dt.tz_localize(None)
                orders_df["total"] = orders_df["total"].astype(float)
            if not category_df.empty:
                category_df["created_at"] = pd.to_datetime(category_df["created_at"], utc=True).dt.tz_localize(None)
                category_df["total"] = category_df["total"].astype(float)
                category_df["category"] = category_df["category"].astype(str)

            info = {
                "used_real_data": not orders_df.empty,
                "real_order_rows": int(len(orders_df)),
                "real_category_rows": int(len(category_df)),
            }
            return orders_df, category_df, info
        except Exception as exc:
            return pd.DataFrame(), pd.DataFrame(), {"used_real_data": False, "reason": str(exc)}

    def _generate_synthetic(self, days=540, seed=42):
        rng = np.random.default_rng(seed)
        end_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
        dates = pd.date_range(end=end_date, periods=days, freq="D")

        dow_multipliers = np.array([0.84, 0.90, 0.95, 1.00, 1.08, 1.20, 1.14])
        trend = np.linspace(0, 6200, days)
        seasonal = 1 + 0.12 * np.sin(2 * np.pi * dates.dayofyear.to_numpy() / 365.25)
        weekly = dow_multipliers[dates.dayofweek.to_numpy()]
        noise = rng.normal(0, 1250, size=days)

        daily_sales = np.maximum(3500, (17000 + trend) * seasonal * weekly + noise)
        avg_ticket = np.clip(rng.normal(185, 18, size=days), 120, 260)
        daily_orders = np.maximum(20, np.round(daily_sales / avg_ticket)).astype(int)

        daily_df = pd.DataFrame(
            {
                "date": dates.normalize(),
                "sales": daily_sales.astype(float),
                "orders": daily_orders.astype(int),
                "source": "synthetic",
            }
        )

        category_rows = []
        base_shares = np.array([0.20, 0.35, 0.13, 0.24, 0.08])
        for idx, date_val in enumerate(dates.normalize()):
            shares = rng.dirichlet(base_shares * 45.0)
            for category, share in zip(DEFAULT_CATEGORIES, shares):
                category_rows.append(
                    {
                        "date": date_val,
                        "category": category,
                        "sales": float(daily_sales[idx] * share),
                        "source": "synthetic",
                    }
                )

        category_df = pd.DataFrame(category_rows)
        return daily_df, category_df

    def _aggregate_real_daily(self, orders_df):
        if orders_df.empty:
            return pd.DataFrame(columns=["date", "sales", "orders", "source"])
        temp = orders_df.copy()
        temp["date"] = temp["created_at"].dt.normalize()
        grouped = (
            temp.groupby("date", as_index=False)
            .agg(sales=("total", "sum"), orders=("total", "size"))
            .assign(source="real")
        )
        return grouped

    def _aggregate_real_categories(self, category_df):
        if category_df.empty:
            return pd.DataFrame(columns=["date", "category", "sales", "source"])
        temp = category_df.copy()
        temp["date"] = temp["created_at"].dt.normalize()
        temp["category"] = (
            temp["category"].str.upper().map(CATEGORY_MAP).fillna(
                temp["category"].str.replace("_", " ").str.title()
            )
        )
        grouped = (
            temp.groupby(["date", "category"], as_index=False)
            .agg(sales=("total", "sum"))
            .assign(source="real")
        )
        return grouped

    def _numeric_features(self, dates):
        dt = pd.to_datetime(dates).dt.normalize()
        ordinal = dt.map(pd.Timestamp.toordinal).astype(int)
        return pd.DataFrame(
            {
                "dow": dt.dt.dayofweek.astype(int),
                "month": dt.dt.month.astype(int),
                "day": dt.dt.day.astype(int),
                "week": dt.dt.isocalendar().week.astype(int),
                "is_weekend": (dt.dt.dayofweek >= 5).astype(int),
                "doy_sin": np.sin(2 * np.pi * dt.dt.dayofyear / 365.25),
                "doy_cos": np.cos(2 * np.pi * dt.dt.dayofyear / 365.25),
                "t": ordinal.astype(int),
            }
        )

    def train_models(self):
        synth_daily, synth_cat = self._generate_synthetic(days=540)
        real_orders, real_categories, real_info = self._query_real_logs(lookback_days=365)
        real_daily = self._aggregate_real_daily(real_orders)
        real_cat_daily = self._aggregate_real_categories(real_categories)

        daily_frames = [synth_daily]
        if not real_daily.empty:
            daily_frames.append(real_daily)
        daily_train = pd.concat(daily_frames, ignore_index=True)
        daily_train = daily_train.sort_values("date").reset_index(drop=True)
        daily_weights = np.where(daily_train["source"] == "real", 4.0, 1.0)

        X_daily = self._numeric_features(daily_train["date"])
        y_sales = daily_train["sales"].astype(float)
        y_orders = daily_train["orders"].astype(float)

        sales_model = RandomForestRegressor(
            n_estimators=350,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
        orders_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=11,
            random_state=42,
            n_jobs=-1,
        )

        sales_model.fit(X_daily, y_sales, sample_weight=daily_weights)
        orders_model.fit(X_daily, y_orders, sample_weight=daily_weights)

        cat_frames = [synth_cat]
        if not real_cat_daily.empty:
            cat_frames.append(real_cat_daily)
        cat_train = pd.concat(cat_frames, ignore_index=True)
        cat_train = cat_train.sort_values("date").reset_index(drop=True)
        cat_weights = np.where(cat_train["source"] == "real", 4.0, 1.0)
        cat_numeric = self._numeric_features(cat_train["date"])
        X_cat = cat_numeric.copy()
        X_cat["category"] = cat_train["category"].astype(str)
        y_cat = cat_train["sales"].astype(float)

        cat_preprocessor = ColumnTransformer(
            transformers=[
                ("category", OneHotEncoder(handle_unknown="ignore"), ["category"]),
                (
                    "numeric",
                    "passthrough",
                    ["dow", "month", "day", "week", "is_weekend", "doy_sin", "doy_cos", "t"],
                ),
            ]
        )
        category_model = Pipeline(
            steps=[
                ("prep", cat_preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=280,
                        max_depth=12,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        category_model.fit(X_cat, y_cat, model__sample_weight=cat_weights)

        observed_categories = cat_train["category"].dropna().astype(str).unique().tolist()
        categories = [c for c in DEFAULT_CATEGORIES if c in observed_categories]
        categories.extend([c for c in observed_categories if c not in categories])
        if not categories:
            categories = DEFAULT_CATEGORIES[:]

        artifact = {
            "sales_model": sales_model,
            "orders_model": orders_model,
            "category_model": category_model,
            "categories": categories,
            "trained_at": pd.Timestamp.utcnow().isoformat(),
            "training_rows": {
                "daily": int(len(daily_train)),
                "category": int(len(cat_train)),
            },
            "real_data_info": real_info,
        }
        joblib.dump(artifact, self.model_path)
        return artifact

    def _load_artifact(self):
        if not self.model_path.exists():
            return self.train_models()
        return joblib.load(self.model_path)

    def retrain(self):
        return self.train_models()

    def _report_from_db(self):
        orders_df, _, info = self._query_real_logs(lookback_days=40)
        if orders_df.empty:
            return None, info

        now = pd.Timestamp.today().normalize()
        orders_df = orders_df.copy()
        orders_df["date"] = orders_df["created_at"].dt.normalize()

        # Today in 2-hour slots for UI compatibility.
        slot_hours = [8, 10, 12, 14, 16, 18, 20]
        slot_labels = ["08 AM", "10 AM", "12 PM", "02 PM", "04 PM", "06 PM", "08 PM"]
        today_orders = orders_df[orders_df["date"] == now]

        today_series = []
        for start_hour, label in zip(slot_hours, slot_labels):
            end_hour = start_hour + 2
            slot_rows = today_orders[
                (today_orders["created_at"].dt.hour >= start_hour)
                & (today_orders["created_at"].dt.hour < end_hour)
            ]
            today_series.append(
                {
                    "time": label,
                    "sales": int(round(slot_rows["total"].sum())),
                    "orders": int(len(slot_rows)),
                }
            )

        daily_map = (
            orders_df.groupby("date", as_index=False)
            .agg(sales=("total", "sum"), orders=("total", "size"))
            .set_index("date")
            .to_dict("index")
        )

        weekly_dates = pd.date_range(end=now, periods=7, freq="D")
        weekly_series = []
        for day in weekly_dates:
            point = daily_map.get(day.normalize(), {"sales": 0.0, "orders": 0})
            weekly_series.append(
                {
                    "day": day.strftime("%a"),
                    "sales": int(round(point["sales"])),
                    "orders": int(point["orders"]),
                }
            )

        monthly_dates = pd.date_range(end=now, periods=28, freq="D")
        monthly_series = []
        for idx in range(4):
            week_dates = monthly_dates[idx * 7 : (idx + 1) * 7]
            week_sales = 0.0
            week_orders = 0
            for date_val in week_dates:
                point = daily_map.get(date_val.normalize(), {"sales": 0.0, "orders": 0})
                week_sales += point["sales"]
                week_orders += int(point["orders"])
            monthly_series.append(
                {
                    "week": f"Week {idx + 1}",
                    "sales": int(round(week_sales)),
                    "orders": int(week_orders),
                }
            )

        report = {
            "today": today_series,
            "weekly": weekly_series,
            "monthly": monthly_series,
            "meta": info,
        }
        return report, info

    def sales_report(self):
        report, info = self._report_from_db()
        if report:
            return report
        fallback = self._load_json("sales-data.json", {"today": [], "weekly": [], "monthly": []})
        fallback["meta"] = info
        return fallback

    def _current_category_baseline(self, lookback_days=14):
        _, real_categories, _ = self._query_real_logs(lookback_days=lookback_days)
        if real_categories.empty:
            return {}

        real_cat_daily = self._aggregate_real_categories(real_categories)
        if real_cat_daily.empty:
            return {}
        baseline = (
            real_cat_daily.groupby("category", as_index=False)["sales"]
            .mean()
            .set_index("category")["sales"]
            .to_dict()
        )
        return baseline

    def predictions(self):
        artifact = self._load_artifact()
        today = pd.Timestamp.today().normalize()
        future_dates = pd.date_range(start=today + pd.Timedelta(days=1), periods=7, freq="D")

        sales_features = self._numeric_features(pd.Series(future_dates))
        daily_forecast = artifact["sales_model"].predict(sales_features)
        daily_forecast = np.maximum(0, daily_forecast)

        next_7_days = []
        prev = daily_forecast[0]
        for idx, (date_val, predicted_sales) in enumerate(zip(future_dates, daily_forecast)):
            trend = "stable"
            if idx > 0:
                if predicted_sales > prev * 1.02:
                    trend = "up"
                elif predicted_sales < prev * 0.98:
                    trend = "down"
            next_7_days.append(
                {
                    "day": date_val.strftime("%a"),
                    "predicted": int(round(predicted_sales)),
                    "trend": trend,
                }
            )
            prev = predicted_sales

        baseline = self._current_category_baseline(lookback_days=14)
        categories = artifact.get("categories", DEFAULT_CATEGORIES)
        selected_categories = categories[:5] if len(categories) >= 5 else (categories + DEFAULT_CATEGORIES)[:5]
        selected_categories = list(dict.fromkeys(selected_categories))

        category_rows = []
        for category in selected_categories:
            cat_features = self._numeric_features(pd.Series(future_dates))
            cat_features["category"] = category
            predicted_values = artifact["category_model"].predict(cat_features)
            predicted_daily = float(np.maximum(0, np.mean(predicted_values)))
            current_daily = float(baseline.get(category, predicted_daily * 0.9))
            category_rows.append(
                {
                    "category": category,
                    "predicted": int(round(predicted_daily)),
                    "current": int(round(max(1.0, current_daily))),
                }
            )

        category_rows = sorted(category_rows, key=lambda row: row["current"], reverse=True)

        return {
            "next7Days": next_7_days,
            "categoryPrediction": category_rows,
            "meta": {
                "trainedAt": artifact.get("trained_at"),
                "realDataInfo": artifact.get("real_data_info", {}),
            },
        }


from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")   # silence Prophet's verbose Stan output

try:
    from prophet import Prophet
    _PROPHET_OK = True
except ImportError:
    _PROPHET_OK = False
    logger.warning("[TrendOracle] prophet not installed. Forecasting disabled.")


# ─── Main class ───────────────────────────────────────────────────────────────

class TrendOracle:
    """
    Per-category demand forecaster using Facebook Prophet.

    Usage:
        oracle = TrendOracle(cfg)
        oracle.fit("data/trend_data.csv")
        results = oracle.forecast(horizon_days=90)
        top = oracle.top_trends(n=10)
    """

    CATEGORIES = ["color", "silhouette", "garment_type"]

    def __init__(self, cfg: dict):
        tf_cfg = cfg.get("trend_forecasting", {})
        self.horizon_days     = tf_cfg.get("forecast_horizon_days", 90)
        self.seasonality_mode = tf_cfg.get("seasonality_mode", "multiplicative")
        self.model_path       = cfg["paths"].get("trend_model", "artifacts/trend_model.pkl")
        self._models: dict[tuple[str, str], Prophet] = {}
        self._forecasts: dict[tuple[str, str], pd.DataFrame] = {}
        self._is_fitted = False

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, csv_path: str, save: bool = True) -> None:
        """
        Fit one Prophet model per (category, value) pair.

        Input : trend_data.csv with columns [ds, category, value, y]
        Output: fitted self._models, optionally persisted to disk
        """
        if not _PROPHET_OK:
            raise ImportError("Install prophet: pip install prophet")

        df = pd.read_csv(csv_path, parse_dates=["ds"])
        logger.info(f"[TrendOracle] Fitting on {len(df)} rows, "
                    f"{df['value'].nunique()} unique trend values")

        for (cat, val), grp in df.groupby(["category", "value"]):
            grp_clean = grp[["ds", "y"]].sort_values("ds").reset_index(drop=True)
            if len(grp_clean) < 10:
                logger.warning(f"[TrendOracle] Skipping {cat}/{val} — too few rows ({len(grp_clean)})")
                continue

            m = Prophet(
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
            )
            m.add_seasonality(name="monthly", period=30.5, fourier_order=5)

            with suppress_stdout():
                m.fit(grp_clean)

            self._models[(cat, val)] = m
            logger.debug(f"[TrendOracle] Fitted model for {cat}/{val}")

        self._is_fitted = True
        logger.success(f"[TrendOracle] Fitted {len(self._models)} Prophet models")

        if save:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, "wb") as f:
                pickle.dump(self._models, f)
            logger.info(f"[TrendOracle] Models saved → {self.model_path}")

    def load(self) -> None:
        with open(self.model_path, "rb") as f:
            self._models = pickle.load(f)
        self._is_fitted = True
        logger.info(f"[TrendOracle] Loaded {len(self._models)} models from {self.model_path}")

    # ── Forecast ──────────────────────────────────────────────────────────────

    def forecast(self, horizon_days: int | None = None) -> dict[tuple[str, str], pd.DataFrame]:
        """
        Generate forecasts for all fitted (category, value) pairs.

        Input : horizon_days (int) — days into the future (default from cfg)
        Output: dict[(cat, val)] → DataFrame columns: ds, yhat, yhat_lower, yhat_upper
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() or load() first.")

        h = horizon_days or self.horizon_days
        self._forecasts = {}

        for (cat, val), model in self._models.items():
            future = model.make_future_dataframe(periods=h)
            fc     = model.predict(future)
            fc_tail = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(h).copy()
            fc_tail["yhat"]       = fc_tail["yhat"].clip(lower=0)
            fc_tail["yhat_lower"] = fc_tail["yhat_lower"].clip(lower=0)
            self._forecasts[(cat, val)] = fc_tail

        return self._forecasts

    # ── Top trends ────────────────────────────────────────────────────────────

    def top_trends(self, n: int = 10) -> list[dict]:
        """
        Return the top-N fastest-growing trend values based on forecast slope.

        Output: list of dicts with keys: category, value, growth_pct, peak_date
        """
        if not self._forecasts:
            self.forecast()

        rows = []
        for (cat, val), fc in self._forecasts.items():
            if len(fc) < 2:
                continue
            start = fc["yhat"].iloc[0]
            end   = fc["yhat"].iloc[-1]
            growth = ((end - start) / (abs(start) + 1e-6)) * 100
            peak_date = str(fc.loc[fc["yhat"].idxmax(), "ds"].date())
            rows.append({
                "category":   cat,
                "value":      val,
                "growth_pct": round(growth, 1),
                "peak_date":  peak_date,
            })

        rows.sort(key=lambda x: x["growth_pct"], reverse=True)
        return rows[:n]

    def heatmap_data(self, category: str = "color") -> pd.DataFrame:
        """
        Return a pivot-ready DataFrame for Streamlit heatmap visualisation.

        Output columns: value, week, avg_demand
        """
        if not self._forecasts:
            self.forecast()

        frames = []
        for (cat, val), fc in self._forecasts.items():
            if cat != category:
                continue
            tmp = fc.copy()
            tmp["value"] = val
            tmp["week"]  = tmp["ds"].dt.isocalendar().week.astype(str)
            frames.append(tmp[["value", "week", "yhat"]])

        if not frames:
            return pd.DataFrame(columns=["value", "week", "avg_demand"])

        combined = pd.concat(frames)
        pivot = combined.groupby(["value", "week"])["yhat"].mean().reset_index()
        pivot.rename(columns={"yhat": "avg_demand"}, inplace=True)
        if pivot["avg_demand"].max() > 0:
            pivot["avg_demand"] = (
                pivot["avg_demand"] / pivot["avg_demand"].max() * 100
            ).round(2)
        return pivot


# ─── Utility ──────────────────────────────────────────────────────────────────

import contextlib, io, sys

@contextlib.contextmanager
def suppress_stdout():
    """Silence Prophet's verbose Stan output (cross-platform)."""
    import os
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
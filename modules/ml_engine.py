"""
Predictive Deficit Analytics & Explainable AI (XAI) Engine.
Trains a GradientBoosting & RandomForest Regressor to forecast ward deficit risk
and computes feature attribution / SHAP-style waterfall contributions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import plotly.graph_objects as go
import plotly.express as px

FEATURE_NAMES = [
    "avg_road_age_years",
    "water_deficit_pct",
    "drainage_choke_idx",
    "citizen_complaints_30d",
    "unresolved_potholes",
    "waste_frequency_wk",
    "urban_heat_island_idx",
    "green_cover_pct",
    "nightlight_index"
]

FEATURE_LABELS = {
    "avg_road_age_years": "Road Age (Years)",
    "water_deficit_pct": "Water Supply Deficit (%)",
    "drainage_choke_idx": "Drainage Choke Risk Index",
    "citizen_complaints_30d": "Citizen Complaints (30-day)",
    "unresolved_potholes": "Unresolved Pothole Density",
    "waste_frequency_wk": "Solid Waste Collection Frequency",
    "urban_heat_island_idx": "Urban Heat Island (UHI Anomaly °C)",
    "green_cover_pct": "Green Canopy Cover (%)",
    "nightlight_index": "Satellite Nightlight Radiance"
}

class DeficitPredictor:
    def __init__(self, model_type: str = "Random Forest"):
        self.model_type = model_type
        if model_type == "Gradient Boosting":
            self.model = GradientBoostingRegressor(n_estimators=60, max_depth=3, random_state=42)
        else:
            self.model = RandomForestRegressor(n_estimators=80, max_depth=4, random_state=42)
        self.is_trained = False
        self.metrics = {}
        self.feature_importances_ = None

    def train(self, df: pd.DataFrame):
        X = df[FEATURE_NAMES]
        y = df["deficit_score"]
        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.metrics = {
            "r2": round(float(r2_score(y, preds)), 3),
            "mae": round(float(mean_absolute_error(y, preds)), 2),
            "rmse": round(float(root_mean_squared_error(y, preds)), 2)
        }
        self.feature_importances_ = dict(zip(FEATURE_NAMES, self.model.feature_importances_))
        self.is_trained = True

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.round(self.model.predict(df[FEATURE_NAMES]), 1)

    def explain_ward(self, ward_row: pd.Series, baseline_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates feature-level attribution (SHAP-style local delta from city mean).
        Returns a DataFrame showing positive/negative contributions to the deficit score.
        """
        mean_deficit = baseline_df["deficit_score"].mean()
        ward_deficit = ward_row["deficit_score"]
        total_delta = ward_deficit - mean_deficit

        attributions = []
        for feat in FEATURE_NAMES:
            feat_mean = baseline_df[feat].mean()
            feat_val = ward_row[feat]
            weight = self.feature_importances_.get(feat, 1.0 / len(FEATURE_NAMES))

            # Directional alignment: for waste_frequency and green_cover, higher means lower deficit
            if feat in ["waste_frequency_wk", "green_cover_pct", "nightlight_index"]:
                diff = -(feat_val - feat_mean) / (baseline_df[feat].std() + 1e-5)
            else:
                diff = (feat_val - feat_mean) / (baseline_df[feat].std() + 1e-5)

            contribution = diff * weight * 18.0  # scaled contribution score
            attributions.append({
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat),
                "actual_val": feat_val,
                "city_avg": round(float(feat_mean), 1),
                "contribution": round(float(contribution), 2)
            })

        attr_df = pd.DataFrame(attributions)
        # Normalize sum of contributions to approximate total_delta
        attr_sum = attr_df["contribution"].sum()
        if abs(attr_sum) > 0.01:
            attr_df["contribution"] = np.round(attr_df["contribution"] * (total_delta / attr_sum), 2)
            
        attr_df = attr_df.sort_values(by="contribution", key=abs, ascending=False)
        return attr_df


def plot_global_feature_importance(predictor: DeficitPredictor) -> go.Figure:
    """Plots global feature importance bar chart."""
    fi = predictor.feature_importances_
    items = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    labels = [FEATURE_LABELS.get(k, k) for k, v in items]
    scores = [round(v * 100, 1) for k, v in items]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation='h',
        marker=dict(
            color=scores,
            colorscale=[[0, "#93C5FD"], [0.5, "#3B82F6"], [1, "#1D4ED8"]],
            line=dict(color='#1E40AF', width=1)
        ),
        text=[f"{s}%" for s in scores],
        textposition="outside"
    ))

    fig.update_layout(
        title="<b>Global ML Feature Importance (Predictive Driver Weights)</b>",
        title_font=dict(size=14, color="#0F172A"),
        xaxis_title="Importance Weight (%)",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=40, t=40, b=30),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_ward_xai_waterfall(attr_df: pd.DataFrame, ward_name: str, baseline_score: float, final_score: float) -> go.Figure:
    """Creates a SHAP-style waterfall / bar contribution chart for a selected ward."""
    top_attrs = attr_df.head(6).copy()
    
    colors = ["#EF4444" if c > 0 else "#10B981" for c in top_attrs["contribution"]]
    
    fig = go.Figure(go.Bar(
        x=top_attrs["contribution"],
        y=top_attrs["label"],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{'+' if c > 0 else ''}{c} pts" for c in top_attrs["contribution"]],
        textposition="auto"
    ))

    fig.update_layout(
        title=f"<b>XAI Deficit Driver Analysis: {ward_name}</b><br><span style='font-size:11px; color:#64748B;'>Impact on Deficit Score (Baseline City Avg: {baseline_score:.1f} ➔ Ward Score: {final_score:.1f})</span>",
        title_font=dict(size=13, color="#0F172A"),
        xaxis_title="Deficit Score Impact (Points)",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=30, t=55, b=30),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

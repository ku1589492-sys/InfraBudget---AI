"""
GIS & Mapping Engine for InfraBudget AI.
Renders interactive Folium and Plotly GIS maps with ward color-coding,
K-Means cluster zones, and rich diagnostic tooltips.
"""

import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def create_folium_map(df: pd.DataFrame, color_mode: str = "Deficit Risk", center_lat: float = 28.62, center_lon: float = 77.20) -> folium.Map:
    """
    Creates an interactive Folium map with custom styled markers, circular radius proportional to population,
    and colors mapped to Deficit Risk or K-Means clusters.
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    cluster_color_map = {
        0: "#EF4444", # Red
        1: "#F59E0B", # Amber
        2: "#10B981"  # Green
    }

    for _, row in df.iterrows():
        if color_mode == "Deficit Risk":
            color = row["risk_color"]
            legend_label = f"Deficit: {row['deficit_score']}/100 ({row['risk_category']})"
        else: # K-Means Clusters
            color = cluster_color_map.get(row["kmeans_cluster"], "#3B82F6")
            legend_label = f"Cluster: {row['cluster_label']}"

        village_str = row.get("village_name", row["ward_name"])

        # HTML popup card
        popup_html = f"""
        <div style="font-family: 'Plus Jakarta Sans', sans-serif; width: 240px; padding: 4px;">
            <div style="font-weight: 800; font-size: 14px; color: #0F172A; margin-bottom: 2px;">
                {row['ward_name']}
            </div>
            <div style="font-size: 11px; font-weight: 700; color: #1E293B; margin-bottom: 4px;">
                🏡 Locality / Village: <span style="color: #2563EB;">{village_str}</span>
            </div>
            <div style="font-size: 11px; color: #64748B; margin-bottom: 8px;">
                {row['zone']} Zone | {row['ward_type']}
            </div>
            <div style="background-color: {color}20; border-left: 3px solid {color}; padding: 4px 8px; border-radius: 4px; margin-bottom: 8px;">
                <span style="font-size: 12px; font-weight: 700; color: {color};">
                    Deficit Score: {row['deficit_score']} / 100
                </span>
                <span style="font-size: 10px; color: #475569; display: block;">Status: {row['risk_category']}</span>
            </div>
            <div style="font-size: 11px; line-height: 1.5; color: #334155;">
                • <b>Road Age:</b> {row['avg_road_age_years']} yrs ({row['unresolved_potholes']} potholes)<br/>
                • <b>Water Deficit:</b> {row['water_deficit_pct']}%<br/>
                • <b>Complaints:</b> {row['citizen_complaints_30d']} logged (30d)<br/>
                • <b>Green Cover:</b> {row['green_cover_pct']}% | UHI: +{row['urban_heat_island_idx']}°C<br/>
                • <b>Cluster:</b> {row['cluster_label'].split(' - ')[0]}
            </div>
        </div>
        """

        # Outer pulse circle
        folium.Circle(
            location=[row["lat"], row["lon"]],
            radius=900,
            color=color,
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.15,
        ).add_to(m)

        # Inner pinpoint marker with clear village name tooltip
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=8,
            color="#FFFFFF",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            tooltip=f"📍 {row['ward_name']} ({village_str}) — Deficit: {row['deficit_score']}/100 ({row['risk_category']})",
            popup=folium.Popup(popup_html, max_width=280)
        ).add_to(m)

    return m


def create_plotly_gis_map(df: pd.DataFrame, color_mode: str = "Deficit Risk", center_lat: float = None, center_lon: float = None) -> go.Figure:
    """
    Renders an interactive high-performance Plotly scatter map with hover metrics and styling.
    """
    if color_mode == "Deficit Risk":
        color_col = "risk_category"
        color_discrete_map = {
            "Critical": "#EF4444",
            "Moderate": "#F59E0B",
            "Stable": "#10B981"
        }
    else:
        color_col = "cluster_label"
        color_discrete_map = {
            "Zone A - High Deficit Core": "#EF4444",
            "Zone B - Developing Outskirts": "#F59E0B",
            "Zone C - Planned Moderate Buffer": "#10B981"
        }

    hover_dict = {
        "lat": False,
        "lon": False,
        "village_name": True,
        "deficit_score": True,
        "risk_category": True,
        "avg_road_age_years": True,
        "water_deficit_pct": True,
        "citizen_complaints_30d": True,
        "cluster_label": True
    }

    c_lat = center_lat if center_lat is not None else (float(df["lat"].mean()) if not df.empty else 28.62)
    c_lon = center_lon if center_lon is not None else (float(df["lon"].mean()) if not df.empty else 77.20)

    if hasattr(px, "scatter_map"):
        fig = px.scatter_map(
            df,
            lat="lat",
            lon="lon",
            color=color_col,
            color_discrete_map=color_discrete_map,
            size="deficit_score",
            size_max=22,
            zoom=11,
            center={"lat": c_lat, "lon": c_lon},
            hover_name="ward_name",
            hover_data=hover_dict,
            map_style="open-street-map",
            title=""
        )
    else:
        fig = px.scatter_mapbox(
            df,
            lat="lat",
            lon="lon",
            color=color_col,
            color_discrete_map=color_discrete_map,
            size="deficit_score",
            size_max=22,
            zoom=11,
            center={"lat": c_lat, "lon": c_lon},
            hover_name="ward_name",
            hover_data=hover_dict,
            mapbox_style="open-street-map",
            title=""
        )

    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="",
            font=dict(size=12, color="#1e293b")
        ),
        height=480
    )
    return fig

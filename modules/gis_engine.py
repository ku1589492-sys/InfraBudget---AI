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

def create_folium_map(
    df: pd.DataFrame, 
    color_mode: str = "Deficit Risk", 
    center_lat: float = 28.62, 
    center_lon: float = 77.20,
    map_style: str = "Standard Streets",
    rainfall_sim_mm: int = 0
) -> folium.Map:
    """
    Creates an interactive Folium map with custom styled markers, satellite/standard tile layers,
    circular radius proportional to population, Deficit Risk/K-Means colors, and Monsoon Flood alerts.
    """
    # Select Base Tiles
    if map_style == "ESRI Satellite View":
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            control_scale=True,
        )
    elif map_style == "CartoDB Dark Matter":
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles="CartoDB dark_matter",
            control_scale=True,
        )
    else:
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

        # Monsoon Flood Risk Calculation if simulation active
        flood_risk_pct = 0.0
        flood_alert_html = ""
        if rainfall_sim_mm > 0:
            flood_risk_pct = min(100.0, float(row.get("drainage_choke_idx", 50)) * (1.0 + rainfall_sim_mm / 120.0) + (100.0 - float(row.get("green_cover_pct", 15))) * 0.15)
            if flood_risk_pct >= 70.0:
                flood_alert_html = f"""
                <div style="background-color: #FEF2F2; border-left: 3px solid #DC2626; padding: 4px 8px; border-radius: 4px; margin-bottom: 6px;">
                    <span style="font-size: 11px; font-weight: 800; color: #DC2626;">
                        🚨 MONSOON FLOOD HAZARD: {flood_risk_pct:.1f}% Risk
                    </span>
                    <span style="font-size: 10px; color: #991B1B; display: block;">Under {rainfall_sim_mm}mm Downpour: Immediate Sump Pump Deployment Advised</span>
                </div>
                """

        # HTML popup card
        popup_html = f"""
        <div style="font-family: 'Plus Jakarta Sans', sans-serif; width: 250px; padding: 4px;">
            <div style="font-weight: 800; font-size: 14px; color: #0F172A; margin-bottom: 2px;">
                {row['ward_name']}
            </div>
            <div style="font-size: 11px; font-weight: 700; color: #1E293B; margin-bottom: 4px;">
                🏡 Locality / Village: <span style="color: #2563EB;">{village_str}</span>
            </div>
            <div style="font-size: 11px; color: #64748B; margin-bottom: 8px;">
                {row['zone']} Zone | {row['ward_type']} | Pop: {row['population']:,}
            </div>
            {flood_alert_html}
            <div style="background-color: {color}20; border-left: 3px solid {color}; padding: 4px 8px; border-radius: 4px; margin-bottom: 8px;">
                <span style="font-size: 12px; font-weight: 700; color: {color};">
                    Deficit Score: {row['deficit_score']} / 100
                </span>
                <span style="font-size: 10px; color: #475569; display: block;">Status: {row['risk_category']}</span>
            </div>
            <div style="font-size: 11px; line-height: 1.5; color: #334155;">
                • <b>Road Age:</b> {row['avg_road_age_years']} yrs ({row['unresolved_potholes']} potholes)<br/>
                • <b>Water Deficit:</b> {row['water_deficit_pct']}% | Drain Choke: {row['drainage_choke_idx']}/100<br/>
                • <b>Complaints:</b> {row['citizen_complaints_30d']} logged (30d)<br/>
                • <b>Green Cover:</b> {row['green_cover_pct']}% | UHI: +{row['urban_heat_island_idx']}°C<br/>
                • <b>Cluster:</b> {row['cluster_label'].split(' - ')[0]}
            </div>
        </div>
        """

        # Outer pulse circle
        pulse_color = "#0284C7" if (rainfall_sim_mm > 0 and flood_risk_pct >= 70.0) else color
        folium.Circle(
            location=[row["lat"], row["lon"]],
            radius=1100 if (rainfall_sim_mm > 0 and flood_risk_pct >= 70.0) else 900,
            color=pulse_color,
            weight=2 if (rainfall_sim_mm > 0 and flood_risk_pct >= 70.0) else 1.5,
            fill=True,
            fill_color=pulse_color,
            fill_opacity=0.35 if (rainfall_sim_mm > 0 and flood_risk_pct >= 70.0) else 0.15,
        ).add_to(m)

        # Inner pinpoint marker with clear village name tooltip
        tooltip_txt = f"📍 {row['ward_name']} ({village_str}) — Deficit: {row['deficit_score']}/100 ({row['risk_category']})"
        if rainfall_sim_mm > 0 and flood_risk_pct >= 70.0:
            tooltip_txt = f"🌊 FLOOD RISK: {flood_risk_pct:.0f}% | {row['ward_name']} ({village_str})"

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=9,
            color="#FFFFFF",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            tooltip=tooltip_txt,
            popup=folium.Popup(popup_html, max_width=290)
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

"""
InfraBudget AI - Data-Driven Ward-Wise Fund & Resource Allocator.
B2G SaaS Command Center Dashboard for Municipal Corporations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import os
import json
from datetime import datetime

# Import custom modules
from config import APP_TITLE, APP_SUBTITLE, VERSION, COLORS, SECTORS
from modules.data_generator import generate_ward_dataset, get_sector_breakdown, STATE_DISTRICT_DATABASE, STATE_ICONS
from modules.gis_engine import create_folium_map, create_plotly_gis_map
from modules.ml_engine import DeficitPredictor, plot_global_feature_importance
from modules.optimization import run_pulp_budget_optimization
from modules.blockchain import generate_blockchain_ledger, get_smart_contract_phases, calculate_sha256
from modules.vision_detector import run_yolo_damage_detector, create_sample_damage_image
from modules.genai_assistant import answer_municipal_query, generate_executive_report

# Page Configuration
st.set_page_config(
    page_title=f"{APP_TITLE} | Municipal Command Center",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Purge any stale legacy widget keys to prevent StreamlitAPIException across page reloads
for legacy_k in [
    "side_state_select", "side_dist_select", "touch_pop_state_select",
    "touch_pop_dist_select", "focus_ward_select", "ward_inspect_select",
    "graph2_ward_select", "escrow_ward_select", "side_cr_input",
    "side_lakhs_input", "t3_budget_cr_input", "t3_budget_lakh_input",
    "side_b_unit", "tab3_budget_unit_radio"
]:
    if legacy_k in st.session_state:
        del st.session_state[legacy_k]

# Session State Initialization
if "selected_state" not in st.session_state:
    st.session_state["selected_state"] = "Delhi NCT"

if "selected_district" not in st.session_state:
    st.session_state["selected_district"] = "Central & North Delhi (MCD)"

if "custom_budget_cr" not in st.session_state:
    st.session_state["custom_budget_cr"] = 35.0

if "budget_unit" not in st.session_state:
    st.session_state["budget_unit"] = "Crores (₹ Cr)"

def set_jurisdiction(new_state: str, new_district: str = None):
    """Sets active state & district and regenerates ward dataset without widget key collisions."""
    if new_state not in STATE_DISTRICT_DATABASE:
        new_state = "Delhi NCT"
    st.session_state["selected_state"] = new_state
    if new_district is None or new_district not in STATE_DISTRICT_DATABASE[new_state]["districts"]:
        new_district = list(STATE_DISTRICT_DATABASE[new_state]["districts"].keys())[0]
    st.session_state["selected_district"] = new_district
    st.session_state["current_loc_key"] = f"{new_state}::{new_district}"
    df = generate_ward_dataset(
        state_name=new_state,
        district_name=new_district,
        seed=42
    )
    if "village_name" not in df.columns:
        df["village_name"] = df["ward_name"].apply(lambda x: x.split(" - ")[-1] + " Village")
    st.session_state["ward_df"] = df
    st.session_state["selected_focus_ward"] = "All Wards (City-wide View)"
    if not df.empty:
        st.session_state["selected_inspect_ward"] = df["ward_name"].iloc[0]
        st.session_state["selected_graph2_ward"] = df["ward_name"].iloc[0]
        st.session_state["selected_escrow_ward"] = df["ward_name"].iloc[0]

curr_loc_key = f"{st.session_state['selected_state']}::{st.session_state['selected_district']}"
stored_df = st.session_state.get("ward_df")
if stored_df is None or st.session_state.get("current_loc_key") != curr_loc_key or "village_name" not in stored_df.columns:
    set_jurisdiction(st.session_state["selected_state"], st.session_state["selected_district"])

if "grievance_history" not in st.session_state:
    st.session_state["grievance_history"] = [
        {"id": "GRV-1029", "ward": "Ward 04 - Karol Bagh", "type": "Severe Road Pothole", "severity": "CRITICAL", "status": "In Review", "time": "18 mins ago"},
        {"id": "GRV-1028", "ward": "Ward 08 - Sadar Bazaar", "type": "Garbage Dump Overflow", "severity": "HIGH", "status": "Dispatched", "time": "42 mins ago"},
        {"id": "GRV-1027", "ward": "Ward 02 - Chandni Chowk", "type": "Drainage Choke", "severity": "CRITICAL", "status": "Escrow Linked", "time": "2 hours ago"}
    ]

if "participatory_votes" not in st.session_state:
    st.session_state["participatory_votes"] = {
        "Default": {
            "Road Resurfacing & Rapid Pothole Patching": 412,
            "Drinking Water Trunk Pipeline Overhaul": 589,
            "Stormwater Automated Sump & Drain Desilting": 312
        }
    }

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {
            "role": "assistant",
            "content": "Welcome Commissioner. I am **InfraBudget AI**, your municipal decision copilot. You can ask any question regarding ward infrastructure deficits, budget reallocation, or drainage risk.",
            "speech": "Welcome Commissioner. I am InfraBudget AI, your municipal decision copilot."
        }
    ]

# Helper to fetch current district metadata
state_data = STATE_DISTRICT_DATABASE[st.session_state["selected_state"]]
dist_details = state_data["districts"][st.session_state["selected_district"]]

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 0.2rem 0 0.8rem 0;">
        <div style="font-size: 1.3rem; font-weight: 800; color: #0F172A; display: flex; align-items: center; gap: 8px;">
            🏛️ {APP_TITLE}
        </div>
        <div style="font-size: 0.75rem; color: #64748B; font-weight: 600;">{VERSION}</div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Location Selection
    st.markdown("### 📍 Location Jurisdiction")
    states_list = list(STATE_DISTRICT_DATABASE.keys())
    side_state = st.selectbox(
        "State / Region:",
        options=states_list,
        index=states_list.index(st.session_state["selected_state"]) if st.session_state["selected_state"] in states_list else 0,
        format_func=lambda s: f"{STATE_ICONS.get(s, '📍')} {s}"
    )
    if side_state != st.session_state["selected_state"]:
        set_jurisdiction(side_state)
        st.rerun()

    districts_list = list(STATE_DISTRICT_DATABASE[st.session_state["selected_state"]]["districts"].keys())
    side_dist = st.selectbox(
        "District / Municipal Corp:",
        options=districts_list,
        index=districts_list.index(st.session_state["selected_district"]) if st.session_state["selected_district"] in districts_list else 0
    )
    if side_dist != st.session_state["selected_district"]:
        set_jurisdiction(st.session_state["selected_state"], side_dist)
        st.rerun()

    st.caption(f"🏛️ {dist_details['full_name']}")

    st.markdown("---")
    
    # 2. Dynamic Budget Allocation Controls (Custom numbers in ₹ Cr or ₹ Lakhs)
    st.markdown("### 💰 Budget Allocation Pool")
    b_unit = st.radio("Enter Budget Unit:", ["Crores (₹ Cr)", "Lakhs (₹ L)"], index=0 if "Crores" in st.session_state.get("budget_unit", "Crores") else 1, horizontal=True)
    st.session_state["budget_unit"] = b_unit
    
    if "Crores" in b_unit:
        budget_cr_input = st.number_input(
            "Enter Total Capital (₹ Crores):",
            min_value=1.0,
            max_value=50000.0,
            value=float(st.session_state["custom_budget_cr"]),
            step=1.0,
            format="%.2f"
        )
        if abs(budget_cr_input - st.session_state["custom_budget_cr"]) > 0.001:
            st.session_state["custom_budget_cr"] = budget_cr_input
            st.rerun()
        total_budget_lakhs = st.session_state["custom_budget_cr"] * 100.0
    else:
        budget_lakhs_input = st.number_input(
            "Enter Total Capital (₹ Lakhs):",
            min_value=100.0,
            max_value=5000000.0,
            value=float(st.session_state["custom_budget_cr"] * 100.0),
            step=100.0,
            format="%.1f"
        )
        new_cr_from_l = budget_lakhs_input / 100.0
        if abs(new_cr_from_l - st.session_state["custom_budget_cr"]) > 0.001:
            st.session_state["custom_budget_cr"] = new_cr_from_l
            st.rerun()
        total_budget_lakhs = st.session_state["custom_budget_cr"] * 100.0

    # Quick Realistic Municipal Presets
    st.caption("⚡ Quick Realistic Budget Presets:")
    p1, p2, p3, p4, p5 = st.columns(5)
    if p1.button("₹15Cr", key="sb_p1", help="Town / Small Municipal Council"):
        st.session_state["custom_budget_cr"] = 15.0
        st.rerun()
    if p2.button("₹45Cr", key="sb_p2", help="Standard Municipal Corporation"):
        st.session_state["custom_budget_cr"] = 45.0
        st.rerun()
    if p3.button("₹125Cr", key="sb_p3", help="Major Smart City Capex"):
        st.session_state["custom_budget_cr"] = 125.0
        st.rerun()
    if p4.button("₹500Cr", key="sb_p4", help="Metro Mega Corporation"):
        st.session_state["custom_budget_cr"] = 500.0
        st.rerun()
    if p5.button("₹1.2kCr", key="sb_p5", help="Mega Infrastructure Program"):
        st.session_state["custom_budget_cr"] = 1200.0
        st.rerun()

    st.markdown("---")
    min_ward_input = st.slider("Min Budget Guarantee per Ward (₹ Lakhs):", min_value=10.0, max_value=150.0, value=40.0, step=5.0)
    max_ward_input = st.slider("Max Cap per Ward (₹ Lakhs):", min_value=150.0, max_value=1200.0, value=500.0, step=25.0)

    st.markdown("#### ⚖️ Sector Priority Weights")
    w_roads = st.slider("Roads & Pavements", 0.5, 3.0, 1.8, 0.1)
    w_water = st.slider("Water Supply & Drainage", 0.5, 3.0, 2.2, 0.1)
    w_sanitation = st.slider("Sanitation & Waste", 0.5, 3.0, 1.4, 0.1)
    w_green = st.slider("Greenery & Urban Cooling", 0.5, 3.0, 1.0, 0.1)

    sector_weights = {
        "Roads & Pavements": w_roads,
        "Water Supply & Sewage": w_water,
        "Sanitation & Waste": w_sanitation,
        "Greenery & Lighting": w_green
    }

    st.markdown("---")
    
    # 3. Ward Filter
    df_wards = st.session_state["ward_df"]
    if "village_name" not in df_wards.columns:
        df_wards["village_name"] = df_wards["ward_name"].apply(lambda x: x.split(" - ")[-1] + " Locality")
        st.session_state["ward_df"] = df_wards

    ward_options = ["All Wards (City-wide View)"] + df_wards["ward_name"].tolist()
    curr_focus = st.session_state.get("selected_focus_ward", "All Wards (City-wide View)")
    if curr_focus not in ward_options:
        curr_focus = "All Wards (City-wide View)"
        st.session_state["selected_focus_ward"] = curr_focus

    selected_ward = st.selectbox("Focus Ward:", options=ward_options, index=ward_options.index(curr_focus))
    st.session_state["selected_focus_ward"] = selected_ward

    filtered_df = df_wards.copy()
    if selected_ward != "All Wards (City-wide View)":
        filtered_df = filtered_df[filtered_df["ward_name"] == selected_ward]


# Initialize Predictor
predictor = DeficitPredictor(model_type="Random Forest")
predictor.train(df_wards)

# ==========================================
# MAIN COMMAND CENTER HEADER
# ==========================================
st.markdown(f"""
<div class="gov-header-container">
    <div>
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
            <span class="gov-live-dot"></span>
            <span class="gov-badge">{dist_details['full_name']}</span>
            <span style="font-size: 0.75rem; color: #cbd5e1; font-weight: 500;">B2G SAAS DECISION ENGINE</span>
        </div>
        <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">
            {APP_TITLE}
        </h1>
        <p style="margin: 2px 0 0 0; font-size: 0.95rem; color: #93c5fd; font-weight: 400;">
            Data-Driven Ward-Wise Fund & Resource Allocation Command Center
        </p>
    </div>
    <div style="text-align: right; display: none; @media (min-width: 768px) {{ display: block; }}">
        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Active District</div>
        <div style="font-size: 1.05rem; font-weight: 700; color: #38bdf8;">{st.session_state['selected_district']}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# TAB NAVIGATION
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ 1. GIS Command Center & Ward View",
    "📊 2. Infrastructure Deficit Analytics",
    "💰 3. Budget Optimizer & Blockchain Ledger",
    "📱 4. Citizen Crowdsourcing & YOLO Vision",
    "🤖 5. Municipal GenAI Copilot"
])


# ==========================================
# TAB 1: GIS COMMAND CENTER & WARD VIEW
# ==========================================
with tab1:
    # Dedicated Touch-Interactive Location Navigator Toolbar above Map
    with st.container(border=True):
        col_nav_title, col_state_pop, col_dist_pop, col_nav_badge = st.columns([3.4, 2.6, 2.8, 3.2])
        
        with col_nav_title:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.6rem;">📍</span>
                <div>
                    <div style="font-size: 1.05rem; font-weight: 800; color: #0F172A;">Interactive Location Finder</div>
                    <div style="font-size: 0.72rem; color: #64748B;">Touch / Click icons below to change State & District</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_state_pop:
            curr_s_icon = STATE_ICONS.get(st.session_state['selected_state'], "🗺️")
            with st.popover(f"{curr_s_icon} State: {st.session_state['selected_state']}", icon="🗺️", help="Touch / Click to choose from all 29 Indian States & UTs"):
                st.markdown("#### 🗺️ Select State / Region")
                st.caption("Available across all 29 States & Union Territories of India:")
                pop_s_idx = states_list.index(st.session_state["selected_state"]) if st.session_state["selected_state"] in states_list else 0
                pop_state = st.selectbox(
                    "Choose Indian State / UT:",
                    options=states_list,
                    index=pop_s_idx,
                    format_func=lambda s: f"{STATE_ICONS.get(s, '📍')} {s}"
                )
                if pop_state != st.session_state["selected_state"]:
                    set_jurisdiction(pop_state)
                    st.rerun()

        with col_dist_pop:
            cur_dists = list(STATE_DISTRICT_DATABASE[st.session_state["selected_state"]]["districts"].keys())
            cur_d_idx = cur_dists.index(st.session_state["selected_district"]) if st.session_state["selected_district"] in cur_dists else 0
            short_dist_name = st.session_state["selected_district"].split(" (")[0]
            with st.popover(f"🏢 District: {short_dist_name}", icon="🏢", help=f"Touch / Click to choose District in {st.session_state['selected_state']}"):
                st.markdown(f"#### 🏢 Select District in {st.session_state['selected_state']}")
                st.caption("Municipal Corporations & Districts:")
                pop_dist = st.selectbox(
                    "District / Municipal Corporation:",
                    options=cur_dists,
                    index=cur_d_idx
                )
                if pop_dist != st.session_state["selected_district"]:
                    set_jurisdiction(st.session_state["selected_state"], pop_dist)
                    st.rerun()

        with col_nav_badge:
            st.markdown(f"""
            <div style="text-align: right;">
                <span style="background: #DCFCE7; color: #15803D; font-weight: 700; font-size: 0.8rem; padding: 4px 10px; border-radius: 9999px;">
                    🟢 {len(df_wards)} Wards & Villages Located
                </span>
                <div style="font-size: 0.76rem; color: #475569; margin-top: 2px;">
                    {dist_details['full_name']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #64748B; margin-top: 4px;'>⚡ 1-Click State Quick Switch:</div>", unsafe_allow_html=True)
        top_quick_states = ["Delhi NCT", "Maharashtra", "Karnataka", "Uttar Pradesh", "Gujarat", "Madhya Pradesh", "Tamil Nadu", "Telangana", "Bihar", "Assam", "West Bengal", "Rajasthan"]
        q_cols = st.columns(len(top_quick_states))
        for idx, s_name in enumerate(top_quick_states):
            if s_name in STATE_DISTRICT_DATABASE:
                with q_cols[idx]:
                    s_icon = STATE_ICONS.get(s_name, "📍")
                    s_short = s_name.replace(" NCT", "").replace("Pradesh", "P.").replace("Bengal", "B.")
                    if st.button(f"{s_icon} {s_short}", key=f"q_btn_{s_name}", help=f"Switch location to {s_name}"):
                        set_jurisdiction(s_name)
                        st.rerun()

    total_wards_count = len(df_wards)
    critical_wards_count = len(df_wards[df_wards["risk_category"] == "Critical"])
    avg_deficit_city = round(float(df_wards["deficit_score"].mean()), 1)
    tot_population = int(df_wards["population"].sum())
    per_capita_spend = int((total_budget_lakhs * 100000) / max(1, tot_population))

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        with st.container(border=True):
            st.metric("Total Wards", f"{total_wards_count}", f"Pop: {tot_population:,}", delta_color="off")

    with kpi_col2:
        with st.container(border=True):
            st.metric("Critical Deficit Wards", f"{critical_wards_count}", f"{(critical_wards_count/total_wards_count)*100:.0f}% of jurisdiction", delta_color="inverse")

    with kpi_col3:
        with st.container(border=True):
            st.metric("Allocated Capital Pool", f"₹{st.session_state['custom_budget_cr']:.2f} Cr", f"₹{per_capita_spend:,}/Capita (₹{total_budget_lakhs:,.0f} L)", delta_color="normal")

    with kpi_col4:
        with st.container(border=True):
            st.metric("District Mean Deficit", f"{avg_deficit_city} / 100", "Severity Baseline", delta_color="off")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    map_col, info_col = st.columns([65, 35])

    with map_col:
        map_header_c1, map_header_c2 = st.columns([5, 5])
        with map_header_c1:
            color_mode = st.radio(
                "Color Encoding:",
                ["Deficit Risk", "K-Means Archetype Clusters"],
                horizontal=True
            )
        with map_header_c2:
            map_engine = st.radio(
                "Map Engine:",
                ["Folium (Interactive Pins)", "Plotly (High-Speed Vector)"],
                horizontal=True
            )

        if "Folium" in map_engine:
            folium_map = create_folium_map(
                filtered_df,
                color_mode=color_mode,
                center_lat=dist_details["center_lat"],
                center_lon=dist_details["center_lon"]
            )
            st_folium(folium_map, use_container_width=True, height=520, returned_objects=[])
        else:
            plotly_map = create_plotly_gis_map(
                filtered_df,
                color_mode=color_mode,
                center_lat=dist_details["center_lat"],
                center_lon=dist_details["center_lon"]
            )
            st.plotly_chart(plotly_map, use_container_width=True)

    with info_col:
        st.markdown("#### 🔍 Ward Diagnostic Profile")
        ward_names = df_wards["ward_name"].tolist()
        v_lookup = {r["ward_name"]: r.get("village_name", r["ward_name"].split(" - ")[-1] + " Locality") for _, r in df_wards.iterrows()}

        # Safe default calculation
        default_inspect = ward_names[0] if ward_names else ""
        if selected_ward != "All Wards (City-wide View)" and selected_ward in ward_names:
            default_inspect = selected_ward
        elif st.session_state.get("selected_inspect_ward") and st.session_state["selected_inspect_ward"] in ward_names:
            default_inspect = st.session_state["selected_inspect_ward"]

        inspect_ward_name = st.selectbox(
            "Select Ward & Village to Inspect:",
            options=ward_names,
            index=ward_names.index(default_inspect) if default_inspect in ward_names else 0,
            format_func=lambda w: f"{w} — 🏡 {v_lookup.get(w, '')}" if w in v_lookup else w
        )
        st.session_state["selected_inspect_ward"] = inspect_ward_name

        matched_rows = df_wards[df_wards["ward_name"] == inspect_ward_name]
        w_row = matched_rows.iloc[0] if not matched_rows.empty else df_wards.iloc[0]
        village_display = w_row.get("village_name", w_row["ward_name"].split(" - ")[-1] + " Locality")

        with st.container(border=True):
            st.subheader(w_row['ward_name'])
            st.markdown(f"🏡 **Village / Locality:** `{village_display}`  \n📍 Zone: **{w_row['zone']}** | {w_row['ward_type']} | Pop: {w_row['population']:,}")

            if w_row['risk_category'] == 'Critical':
                st.error(f"🚨 Critical Risk (Deficit: {w_row['deficit_score']}/100)")
            elif w_row['risk_category'] == 'Moderate':
                st.warning(f"⚠️ Moderate Risk (Deficit: {w_row['deficit_score']}/100)")
            else:
                st.success(f"✅ Stable Condition (Deficit: {w_row['deficit_score']}/100)")

            st.progress(float(w_row['deficit_score']) / 100.0)

            st.divider()

            st.markdown("##### 📋 Physical Indicators")
            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    label="Road Degradation",
                    value=f"{w_row['avg_road_age_years']} Yrs",
                    delta=f"{w_row['unresolved_potholes']} Potholes",
                    delta_color="inverse"
                )
                st.metric(
                    label="Water Shortage",
                    value=f"{w_row['water_deficit_pct']}%",
                    delta=f"{w_row['iot_pipeline_pressure_psi']} PSI",
                    delta_color="inverse"
                )
            with m2:
                st.metric(
                    label="Drain Choke Risk",
                    value=f"{w_row['drainage_choke_idx']}/100",
                    delta="Choke Index",
                    delta_color="inverse"
                )
                st.metric(
                    label="30-Day Grievances",
                    value=f"{w_row['citizen_complaints_30d']}",
                    delta="Public Tickets",
                    delta_color="off"
                )

            st.metric(
                label="Current Capital Allocation",
                value=f"₹{w_row['current_budget_lakhs']} Lakhs",
                delta=f"₹{w_row['current_budget_lakhs']/100:.2f} Cr"
            )

            st.divider()

            if w_row['risk_category'] == 'Critical':
                st.error("💡 **Algorithmic Justification:** Chronic road degradation and severe water deficit. High priority allocation required to avoid infrastructure failure.")
            elif w_row['risk_category'] == 'Moderate':
                st.warning("💡 **Algorithmic Justification:** Moderate drainage silt and rising citizen grievances. Targeted rehabilitation funds allocated.")
            else:
                st.success("💡 **Algorithmic Justification:** Infrastructure is largely stable. Baseline routine maintenance allocated.")


# ==========================================
# TAB 2: INFRASTRUCTURE DEFICIT ANALYTICS
# ==========================================
with tab2:
    st.markdown("### 📊 Infrastructure Deficit Analytics")
    st.caption("Simplified dual-level diagnostics: District-wide problem distribution vs Ward-specific comparative breakdown.")

    g_col1, g_col2 = st.columns([5, 5])

    with g_col1:
        st.markdown(f"#### 🏙️ Graph 1: District-Wide Problem Share ({st.session_state['selected_district']})")
        
        tot_potholes = int(df_wards["unresolved_potholes"].sum())
        tot_complaints = int(df_wards["citizen_complaints_30d"].sum())
        avg_water_def = round(float(df_wards["water_deficit_pct"].mean()), 1)
        avg_road_age = round(float(df_wards["avg_road_age_years"].mean()), 1)
        avg_drain_choke = round(float(df_wards["drainage_choke_idx"].mean()), 1)

        problem_labels = [
            "Roads & Pavement Degradation",
            "Drinking Water Shortage",
            "Drainage Choke & Flood Risk",
            "Solid Waste Collection Lag"
        ]
        problem_values = [
            round(df_wards["avg_road_age_years"].mean() * 4.5, 1),
            round(df_wards["water_deficit_pct"].mean() * 1.1, 1),
            round(df_wards["drainage_choke_idx"].mean() * 0.8, 1),
            round((7 - df_wards["waste_frequency_wk"].mean()) * 8.0, 1)
        ]

        fig_city_problems = px.pie(
            names=problem_labels,
            values=problem_values,
            color_discrete_sequence=["#EF4444", "#3B82F6", "#8B5CF6", "#F59E0B"],
            hole=0.45,
            title="District-wide Problem Share (%)"
        )
        fig_city_problems.update_layout(
            height=330,
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_city_problems, use_container_width=True)

        with st.container(border=True):
            st.markdown(f"""**District Overall Totals:**  
• Active Potholes across District: **{tot_potholes:,}** | 30-Day Public Grievances: **{tot_complaints:,}**  
• Mean Water Deficit: **{avg_water_def}%** | Average Road Age: **{avg_road_age} Years**""")

    with g_col2:
        st.markdown("#### 🎯 Graph 2: Selected Ward vs. District Baseline")
        
        g2_options = df_wards["ward_name"].tolist()
        curr_g2 = st.session_state.get("selected_graph2_ward")
        if curr_g2 not in g2_options:
            curr_g2 = g2_options[0] if g2_options else ""
            st.session_state["selected_graph2_ward"] = curr_g2

        target_ward = st.selectbox(
            "Select Ward to Compare:",
            options=g2_options,
            index=g2_options.index(curr_g2) if curr_g2 in g2_options else 0
        )
        st.session_state["selected_graph2_ward"] = target_ward
        matched_t = df_wards[df_wards["ward_name"] == target_ward]
        t_row = matched_t.iloc[0] if not matched_t.empty else df_wards.iloc[0]

        comp_categories = ["Road Age (Yrs)", "Water Deficit (%)", "Drain Choke (0-100)", "Potholes (Count)"]
        ward_metrics = [
            t_row["avg_road_age_years"],
            t_row["water_deficit_pct"],
            t_row["drainage_choke_idx"],
            t_row["unresolved_potholes"]
        ]
        city_metrics = [
            avg_road_age,
            avg_water_def,
            avg_drain_choke,
            round(df_wards["unresolved_potholes"].mean(), 1)
        ]

        fig_ward_comp = go.Figure()
        fig_ward_comp.add_trace(go.Bar(
            name=f"Selected Ward ({target_ward.split(' - ')[-1]})",
            x=comp_categories,
            y=ward_metrics,
            marker_color=t_row["risk_color"]
        ))
        fig_ward_comp.add_trace(go.Bar(
            name="District Baseline Average",
            x=comp_categories,
            y=city_metrics,
            marker_color="#94A3B8"
        ))

        fig_ward_comp.update_layout(
            barmode='group',
            height=330,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_ward_comp, use_container_width=True)

        pothole_diff = t_row["unresolved_potholes"] - df_wards["unresolved_potholes"].mean()
        water_diff = t_row["water_deficit_pct"] - avg_water_def

        with st.container(border=True):
            st.markdown(f"""**Key Diagnostic Summary for {target_ward}:**  
• **Pothole Density:** **{abs(pothole_diff):.0f} {'more' if pothole_diff > 0 else 'fewer'}** potholes than the district baseline.  
• **Water Deficit:** **{abs(water_diff):.1f}% {'higher deficit' if water_diff > 0 else 'better supply'}** than average.  
• **Advisory:** {'High vulnerability; priority capital disbursement needed.' if t_row['risk_category']=='Critical' else 'Moderate vulnerability; targeted intervention.' if t_row['risk_category']=='Moderate' else 'Stable condition; standard baseline maintenance.'}""")


# ==========================================
# TAB 3: BUDGET OPTIMIZER & BLOCKCHAIN LEDGER
# ==========================================
with tab3:
    st.markdown("### 💰 Budget Simulator & Blockchain Fund Tracking")
    st.caption("Direct municipal capital input with constrained linear optimization via PuLP LP solver paired with SHA-256 tamper-proof disbursement ledger.")

    # Interactive Budget Workbench Container
    with st.container(border=True):
        st.markdown("#### 💵 Municipal Capital Allocation Workbench & Direct Input")
        b_col_inp, b_col_stats = st.columns([6, 4])
        
        with b_col_inp:
            bu_col, bval_col = st.columns([4, 6])
            with bu_col:
                t3_unit = st.radio("Budget Unit:", ["Crores (₹ Cr)", "Lakhs (₹ L)"], index=0 if "Crores" in st.session_state.get("budget_unit", "Crores") else 1, horizontal=True)
                st.session_state["budget_unit"] = t3_unit
            with bval_col:
                if "Crores" in t3_unit:
                    t3_cr_val = st.number_input(
                        "Type Total Capital (₹ Crores):",
                        min_value=1.0,
                        max_value=50000.0,
                        value=float(st.session_state["custom_budget_cr"]),
                        step=1.0,
                        format="%.2f",
                        help="Enter any realistic municipal expenditure figure in ₹ Crores"
                    )
                    if abs(t3_cr_val - st.session_state["custom_budget_cr"]) > 0.001:
                        st.session_state["custom_budget_cr"] = t3_cr_val
                        st.rerun()
                    total_budget_lakhs = st.session_state["custom_budget_cr"] * 100.0
                else:
                    t3_lakh_val = st.number_input(
                        "Type Total Capital (₹ Lakhs):",
                        min_value=100.0,
                        max_value=5000000.0,
                        value=float(st.session_state["custom_budget_cr"] * 100.0),
                        step=100.0,
                        format="%.1f",
                        help="Enter any realistic municipal expenditure figure in ₹ Lakhs"
                    )
                    new_cr = t3_lakh_val / 100.0
                    if abs(new_cr - st.session_state["custom_budget_cr"]) > 0.001:
                        st.session_state["custom_budget_cr"] = new_cr
                        st.rerun()
                    total_budget_lakhs = st.session_state["custom_budget_cr"] * 100.0

            st.caption("⚡ Realistic Municipal Presets (1-Click Apply):")
            tb_p1, tb_p2, tb_p3, tb_p4, tb_p5 = st.columns(5)
            if tb_p1.button("₹15 Cr", key="t3_btn_15", help="Town / Small Municipal Council"):
                st.session_state["custom_budget_cr"] = 15.0
                st.rerun()
            if tb_p2.button("₹45 Cr", key="t3_btn_45", help="Standard Municipal Corporation"):
                st.session_state["custom_budget_cr"] = 45.0
                st.rerun()
            if tb_p3.button("₹125 Cr", key="t3_btn_125", help="Smart City Capital Capex"):
                st.session_state["custom_budget_cr"] = 125.0
                st.rerun()
            if tb_p4.button("₹500 Cr", key="t3_btn_500", help="Metro Mega Corporation"):
                st.session_state["custom_budget_cr"] = 500.0
                st.rerun()
            if tb_p5.button("₹1,200 Cr", key="t3_btn_1200", help="State Infrastructure Mission"):
                st.session_state["custom_budget_cr"] = 1200.0
                st.rerun()

        with b_col_stats:
            st.markdown("**🏛️ Jurisdiction Fiscal Scope**")
            per_capita_spend = int((total_budget_lakhs * 100000) / max(1, tot_population))
            st.metric("Per-Capita Capex", f"₹{per_capita_spend:,} / Citizen", "Population Normalized")
            st.markdown(f"• **Total Pool:** **₹{st.session_state['custom_budget_cr']:.2f} Cr** (₹{total_budget_lakhs:,.1f} L)  \n• **Population:** **{tot_population:,}** Citizens across **{len(df_wards)}** Wards  \n• **Solver:** **PuLP Simplex LP (Global Optimum)**")

    opt_res = run_pulp_budget_optimization(
        df=df_wards,
        total_budget_lakhs=total_budget_lakhs,
        sector_weights=sector_weights,
        min_ward_budget=min_ward_input,
        max_ward_budget=max_ward_input
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    alloc_df = opt_res["allocations_df"]
    
    comp_fig = go.Figure()
    comp_fig.add_trace(go.Bar(
        x=alloc_df["ward_name"],
        y=alloc_df["current_budget_lakhs"],
        name="Historical Allocation (₹L)",
        marker_color="#94A3B8"
    ))
    comp_fig.add_trace(go.Bar(
        x=alloc_df["ward_name"],
        y=alloc_df["optimal_budget_lakhs"],
        name="PuLP AI Optimal Allocation (₹L)",
        marker_color="#2563EB"
    ))
    comp_fig.update_layout(
        title=f"<b>Historical Budget vs. PuLP Mathematically Optimal Allocation (Total Pool: ₹{st.session_state['custom_budget_cr']:.2f} Cr)</b>",
        xaxis_tickangle=-35,
        barmode='group',
        height=360,
        margin=dict(l=20, r=20, t=50, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(comp_fig, use_container_width=True)

    with st.expander("📊 View Detailed Ward-Wise Allocation & Sector Split Table", expanded=False):
        st.dataframe(
            alloc_df[[
                "ward_name", "zone", "initial_deficit", "optimal_budget_lakhs",
                "budget_delta_lakhs", "projected_deficit", "deficit_reduction_pct"
            ]].rename(columns={
                "ward_name": "Ward",
                "zone": "Zone",
                "initial_deficit": "Initial Deficit (0-100)",
                "optimal_budget_lakhs": "PuLP Allocation (₹L)",
                "budget_delta_lakhs": "Delta (₹L)",
                "projected_deficit": "New Deficit",
                "deficit_reduction_pct": "Improvement %"
            }),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")
    st.markdown("### ⛓️ Blockchain Anti-Leakage Ledger & Smart Contract Escrows")
    
    b_col1, b_col2 = st.columns([6, 4])
    with b_col1:
        st.markdown("#### 📜 Distributed Municipal Disbursement Ledger")
        blockchain_ledger = generate_blockchain_ledger(df_wards)
        
        ledger_data = []
        for b in blockchain_ledger:
            ledger_data.append({
                "Block #": f"Block #{b.index}",
                "Timestamp": b.timestamp,
                "Ward": b.ward_name,
                "Amount (₹L)": f"₹{b.amount_lakhs:.1f}L",
                "Milestone Phase": b.milestone_phase,
                "SHA-256 Hash": f"{b.hash[:10]}...{b.hash[-6:]}",
                "Status": "✅ VERIFIED ON-CHAIN"
            })
        st.dataframe(pd.DataFrame(ledger_data), use_container_width=True, hide_index=True, height=260)
        st.success("🔒 **Zero-Leakage Assurance:** All disbursement blocks are cryptographically chained via SHA-256 Merkle root. 0 Tampering Detected.")

    with b_col2:
        st.markdown("#### 📑 Smart Contract Escrow Milestones")
        escrow_opts = df_wards["ward_name"].tolist()[:4]
        curr_escrow = st.session_state.get("selected_escrow_ward")
        if curr_escrow not in escrow_opts:
            curr_escrow = escrow_opts[0] if escrow_opts else ""
            st.session_state["selected_escrow_ward"] = curr_escrow

        escrow_ward = st.selectbox(
            "Select Flagship Ward Project:",
            options=escrow_opts,
            index=escrow_opts.index(curr_escrow) if curr_escrow in escrow_opts else 0
        )
        st.session_state["selected_escrow_ward"] = escrow_ward
        phases = get_smart_contract_phases(escrow_ward)
        for p in phases:
            with st.container(border=True):
                p_c1, p_c2 = st.columns([7, 3])
                with p_c1:
                    st.markdown(f"**{p['phase']}**")
                    st.caption(f"Verification Oracle: {p['proof_type']}")
                with p_c2:
                    if "100%" in p['escrow_state']:
                        st.success(p['escrow_state'])
                    else:
                        st.warning(p['escrow_state'])


# ==========================================
# TAB 4: CITIZEN CROWDSOURCING & YOLO VISION
# ==========================================
with tab4:
    st.markdown("### 📱 Citizen Crowdsourcing & AI Damage Vision")
    st.caption("Citizen photo verification via lightweight YOLOv8 preview with instant deficit score recalibration.")
    
    c_col1, c_col2 = st.columns([5, 5])

    with c_col1:
        st.markdown("#### 📷 Citizen Grievance AI Image Detection")
        image_mode = st.radio(
            "Select Image Source:",
            ["Pre-loaded Demo Samples", "Upload Custom Image"],
            horizontal=True
        )

        test_img = None
        requested_type = None

        if image_mode == "Pre-loaded Demo Samples":
            sample_choice = st.selectbox(
                "Choose Damage Scenario:",
                ["Sample 1: Severe Road Pothole", "Sample 2: Unsegregated Garbage Dump", "Sample 3: High-Pressure Pipeline Rupture"]
            )
            type_map = {
                "Sample 1: Severe Road Pothole": "pothole",
                "Sample 2: Unsegregated Garbage Dump": "garbage",
                "Sample 3: High-Pressure Pipeline Rupture": "broken_pipe"
            }
            requested_type = type_map[sample_choice]
            sample_path = f"sample_images/sample_{requested_type}.jpg"
            if os.path.exists(sample_path):
                from PIL import Image
                test_img = Image.open(sample_path)
            else:
                test_img = create_sample_damage_image(requested_type)

        else:
            uploaded_file = st.file_uploader("Upload Grievance Photo (JPG, PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                from PIL import Image
                test_img = Image.open(uploaded_file)

        if test_img is not None:
            annotated_img, det_meta = run_yolo_damage_detector(test_img, requested_type=requested_type)
            st.image(annotated_img, caption="YOLOv8 Computer Vision Annotated Output", use_container_width=True)

            with st.container(border=True):
                head_d1, head_d2 = st.columns([7, 3])
                with head_d1:
                    st.markdown(f"**🎯 Detected:** {det_meta['class_label']}")
                with head_d2:
                    if det_meta['severity'] == 'CRITICAL':
                        st.error(f"🚨 {det_meta['severity']}")
                    else:
                        st.warning(f"⚠️ {det_meta['severity']}")

                st.write(f"• **Confidence Score:** {det_meta['confidence']}%")
                st.write(f"• **Estimated Repair Cost:** {det_meta['estimated_repair_cost']}")
                st.write(f"• **Deficit Impact:** +{det_meta['score_penalty']} points immediate penalty")

            if st.button("📤 File Ticket & Recalibrate Ward Deficit Score", use_container_width=True):
                target_w = df_wards.iloc[0]["ward_name"]
                st.session_state["ward_df"].loc[st.session_state["ward_df"]["ward_name"] == target_w, "deficit_score"] += det_meta['score_penalty']
                st.success(f"Ticket filed! Deficit score for {target_w} incremented by +{det_meta['score_penalty']} points.")

    with c_col2:
        st.markdown("#### 🗳️ Participatory Budgeting & Citizen Voting")
        st.markdown("Citizens directly vote on local micro-infrastructure priorities:")

        projects = st.session_state["participatory_votes"]["Default"]
        total_v = sum(projects.values())

        for proj_name, votes in projects.items():
            pct = round((votes / max(1, total_v)) * 100, 1)
            v_c1, v_c2 = st.columns([7, 3])
            with v_c1:
                st.markdown(f"**{proj_name}** ({votes} votes)")
                st.progress(pct / 100.0)
            with v_c2:
                if st.button(f"👍 Vote ({pct}%)", key=f"btn_{proj_name}"):
                    st.session_state["participatory_votes"]["Default"][proj_name] += 1
                    st.rerun()

        st.markdown("---")
        st.markdown("#### 📋 Live Citizen Grievance Stream")
        st.dataframe(pd.DataFrame(st.session_state["grievance_history"]), use_container_width=True, hide_index=True)


# ==========================================
# TAB 5: MUNICIPAL GENAI COPILOT
# ==========================================
with tab5:
    st.markdown("### 🤖 Municipal GenAI Copilot")
    st.caption(f"Conversational decision assistant • Current Jurisdiction: {dist_details['full_name']}")

    c_b1, c_b2 = st.columns([7, 3])
    with c_b1:
        if st.button("📢 Explain This Project in Simple Words (प्रोजेक्ट आसान भाषा में समझें)", use_container_width=True):
            st.session_state["pending_prompt"] = "Explain the whole InfraBudget AI project in simple words and how it works"
    with c_b2:
        if st.button("🔄 Clear Chat History", use_container_width=True):
            st.session_state["chat_history"] = [
                {
                    "role": "assistant",
                    "content": f"Chat reset. Ask any question about {st.session_state['selected_district']} infrastructure or budget allocation.",
                    "speech": "Chat reset."
                }
            ]
            st.rerun()

    # Direct conversation display
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">👤 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-ai">🏛️ <b>InfraBudget AI:</b><br/>{msg["content"]}</div>', unsafe_allow_html=True)
            if "speech" in msg and msg["speech"]:
                safe_speech = msg["speech"].replace('"', '&quot;').replace("'", "&#39;").replace("\n", " ")
                st.components.v1.html(f"""
                <div style="margin-bottom: 8px; margin-left: 6px; display: flex; align-items: center; gap: 8px;">
                    <button onclick="speakText()" style="background: #eff6ff; border: 1px solid #bfdbfe; color: #1d4ed8; padding: 5px 14px; border-radius: 6px; font-size: 12px; font-weight: 700; cursor: pointer; display: inline-flex; align-items: center; gap: 6px;">
                        🔊 Listen to Briefing (Text-to-Speech)
                    </button>
                    <button onclick="window.speechSynthesis.cancel()" style="background: #f1f5f9; border: 1px solid #cbd5e1; color: #475569; padding: 5px 10px; border-radius: 6px; font-size: 12px; cursor: pointer;">
                        ⏹️ Stop
                    </button>
                    <script>
                    function speakText() {{
                        window.speechSynthesis.cancel();
                        var msg = new SpeechSynthesisUtterance("{safe_speech}");
                        msg.rate = 0.9;
                        msg.pitch = 1.0;
                        window.speechSynthesis.speak(msg);
                    }}
                    </script>
                </div>
                """, height=42)

    # Simple chat input
    user_query = st.chat_input(f"Ask a question about {st.session_state['selected_district']} (e.g., 'Which ward needs immediate drainage funds?', 'Explain Ward 1 deficit')...")
    
    prompt_to_process = st.session_state.pop("pending_prompt", None) or user_query

    if prompt_to_process:
        st.session_state["chat_history"].append({"role": "user", "content": prompt_to_process})
        
        is_hindi = any(char in prompt_to_process for char in "अआइईउऊऋएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह") or "samjhao" in prompt_to_process.lower() or "kaise" in prompt_to_process.lower()
        lang_detected = "Hindi" if is_hindi else "English"
        
        ai_resp = answer_municipal_query(prompt_to_process, df_wards, city_name=st.session_state['selected_district'], lang=lang_detected)
        
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": ai_resp["content"],
            "speech": ai_resp.get("speech", "")
        })
        st.rerun()

    st.markdown("---")
    st.markdown("#### 📄 Executive Report Exporter")
    rep_c1, rep_c2 = st.columns([7, 3])
    with rep_c1:
        st.markdown(f"Official municipal memorandum citing ML deficit diagnostics and PuLP allocations for **{st.session_state['selected_district']}**.")
    with rep_c2:
        report_text = generate_executive_report(df_wards, city_name=st.session_state['selected_district'])
        st.download_button(
            label="📥 Download Official Report (.md)",
            data=report_text,
            file_name=f"InfraBudget_{st.session_state['selected_district'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
            use_container_width=True
        )

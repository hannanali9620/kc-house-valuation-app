import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
import plotly.graph_objects as go

# --- PAGE SETUP ---
st.set_page_config(
    page_title="ValuEdge AI | King County Real Estate",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN ENTERPRISE UI STYLING (CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Top Banner Gradient */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #2563EB 100%);
        padding: 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 8px;
        color: #FFFFFF;
    }
    .hero-sub {
        color: #94A3B8;
        font-size: 1rem;
        margin: 0;
    }
    
    /* SaaS Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 6px;
    }
    .metric-tag {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 8px;
        border-radius: 6px;
        margin-top: 6px;
    }
    .tag-blue { background: #EFF6FF; color: #2563EB; }
    .tag-green { background: #ECFDF5; color: #059669; }
    
    /* Section Headers */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD SAVED PIPELINE ---
@st.cache_resource
def load_model_pipeline():
    model = joblib.load('best_knn_model.joblib')
    scaler = joblib.load('scaler.joblib')
    try:
        features = joblib.load('model_features.joblib')
    except FileNotFoundError:
        features = [
            'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'grade', 'sqft_above',
            'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
            'lat', 'long', 'sqft_living15', 'sqft_lot15'
        ]
    return model, scaler, features

# Call the function with the exact same name:
model, scaler, feature_cols = load_model_pipeline()

def run_valuation(payload):
    df = pd.DataFrame([payload]).reindex(columns=feature_cols, fill_value=0)
    scaled = scaler.transform(df)
    return float(model.predict(scaled)[0])

# --- TOP CLIENT BANNER ---
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">ValuEdge™ Real Estate Valuation Suite</div>
    <p class="hero-sub">Institutional automated valuation model (AVM) for King County, WA. Instant market appraisals, comps analysis, and capital expenditure modeling.</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR: CLIENT PROPERTY CONFIGURATOR ---
st.sidebar.markdown("### 🏢 Property Profile")

# Client-friendly preset selector
preset = st.sidebar.selectbox(
    "Quick Neighborhood Selector",
    options=["Custom Coordinates", "Bellevue Luxury (Eastside)", "Capitol Hill (Historic Seattle)", "Redmond Tech Corridor", "Renton Affordable Suburban"],
    index=1
)

# Preset Coordinates & Defaults
presets_dict = {
    "Bellevue Luxury (Eastside)": {"lat": 47.6104, "long": -122.2007, "zipcode": 98004, "grade": 10, "sqft": 3600},
    "Capitol Hill (Historic Seattle)": {"lat": 47.6253, "long": -122.3222, "zipcode": 98102, "grade": 9, "sqft": 2400},
    "Redmond Tech Corridor": {"lat": 47.6740, "long": -122.1215, "zipcode": 98052, "grade": 8, "sqft": 2600},
    "Renton Affordable Suburban": {"lat": 47.4829, "long": -122.2171, "zipcode": 98055, "grade": 7, "sqft": 1750}
}

if preset != "Custom Coordinates":
    default_lat = presets_dict[preset]["lat"]
    default_long = presets_dict[preset]["long"]
    default_zip = presets_dict[preset]["zipcode"]
    default_grade = presets_dict[preset]["grade"]
    default_sqft = presets_dict[preset]["sqft"]
else:
    default_lat, default_long, default_zip, default_grade, default_sqft = 47.6062, -122.3321, 98101, 8, 2200

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    bedrooms = st.sidebar.number_input("Bedrooms", 1, 10, 3)
    floors = st.sidebar.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0], index=1)
    condition = st.sidebar.slider("Condition", 1, 5, 3)

with col_sb2:
    bathrooms = st.sidebar.number_input("Bathrooms", 0.75, 8.0, 2.5, step=0.25)
    grade = st.sidebar.slider("Building Grade", 3, 13, default_grade)
    waterfront = st.sidebar.selectbox("Waterfront", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📐 Dimensions & Coordinates")
sqft_living = st.sidebar.slider("Living Area (sqft)", 500, 8000, default_sqft, step=50)
sqft_lot = st.sidebar.number_input("Lot Size (sqft)", 500, 50000, 6000, step=250)
sqft_above = st.sidebar.number_input("Above Ground (sqft)", 500, 6000, int(sqft_living * 0.8), step=50)
sqft_basement = max(0, sqft_living - sqft_above)

lat = st.sidebar.slider("Latitude", 47.1500, 47.7800, default_lat, step=0.001, format="%.4f")
long = st.sidebar.slider("Longitude", -122.5200, -121.3100, default_long, step=0.001, format="%.4f")
zipcode = st.sidebar.number_input("Zipcode", 98001, 98199, default_zip, step=1)

# Build current specs payload
current_specs = {
    'bedrooms': bedrooms, 'bathrooms': bathrooms, 'sqft_living': sqft_living,
    'sqft_lot': sqft_lot, 'floors': floors, 'waterfront': waterfront, 'view': 0,
    'condition': condition, 'grade': grade, 'sqft_above': sqft_above,
    'sqft_basement': sqft_basement, 'yr_built': 1992, 'yr_renovated': 0,
    'zipcode': zipcode, 'lat': lat, 'long': long, 'sqft_living15': sqft_living, 'sqft_lot15': sqft_lot
}

# Run model
base_valuation = run_valuation(current_specs)
low_val = base_valuation * 0.85
high_val = base_valuation * 1.15
price_sqft = base_valuation / sqft_living

# --- EXECUTIVE CLIENT KPI CARDS ---
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Estimated Appraisal</div>
        <div class="metric-val" style="color: #2563EB;">${base_valuation:,.0f}</div>
        <span class="metric-tag tag-blue">Market Median Basis</span>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Confidence Band (±15%)</div>
        <div class="metric-val">${low_val:,.0f} - ${high_val:,.0f}</div>
        <span class="metric-tag tag-blue">Model Range</span>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Unit Economics</div>
        <div class="metric-val">${price_sqft:,.2f}<span style="font-size: 1rem; color: #64748B;">/sqft</span></div>
        <span class="metric-tag tag-green">County Avg: ~$310/sqft</span>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Structural Index</div>
        <div class="metric-val">Grade {grade}<span style="font-size: 1rem; color: #64748B;">/13</span></div>
        <span class="metric-tag tag-blue">{bedrooms} Beds | {bathrooms} Baths</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# --- TABS INTERFACE ---
tab1, tab2, tab3 = st.tabs(["🗺️ Geospatial 3D Intelligence", "📈 Renovation & ROI Engine", "📋 Client Export Dossier"])

# TAB 1: 3D GEOSPATIAL INTELLIGENCE
with tab1:
    st.markdown('<div class="section-title">Micro-Market Spatial Mapping</div>', unsafe_allow_html=True)
    st.caption("3D Column altitude represents relative valuation magnitude in current view.")
    
    map_df = pd.DataFrame([{
        'lat': lat, 'lon': long, 'price': base_valuation, 'sqft': sqft_living, 'grade': grade
    }])
    
    col_layer = pdk.Layer(
        "ColumnLayer",
        data=map_df,
        get_position=["lon", "lat"],
        get_elevation="price",
        elevation_scale=0.012,
        radius=140,
        get_fill_color=[37, 99, 235, 200],
        pickable=True,
        auto_highlight=True
    )
    
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["lon", "lat"],
        get_radius=300,
        get_color=[239, 68, 68, 220],
        pickable=True
    )
    
    view_state = pdk.ViewState(latitude=lat, longitude=long, zoom=12.2, pitch=50, bearing=-20)
    
    st.pydeck_chart(pdk.Deck(
        layers=[col_layer, scatter_layer],
        initial_view_state=view_state,
        tooltip={"text": "Valuation: ${price}\nLiving Area: {sqft} sqft\nGrade: {grade}/13"}
    ))

# TAB 2: RENOVATION & CAPITAL ALLOCATION SIMULATOR
with tab2:
    st.markdown('<div class="section-title">Value-Add & Renovation Simulator</div>', unsafe_allow_html=True)
    st.write("Calculate return on investment (ROI) for pre-sale renovations or client fix-and-flip scenarios.")
    
    c_sim1, c_sim2 = st.columns([1, 1])
    with c_sim1:
        grade_bump = st.slider("Quality Upgrade (Grade Boost)", 0, 3, 1, help="Custom cabinets, marble countertops, high-end woodwork")
        cond_bump = st.slider("Condition Upgrade", 0, 2, 1, help="New roof, HVAC, updated plumbing")
        extra_sqft = st.number_input("Square Footage Addition (sqft)", 0, 1500, 250, step=50)
        reno_cost = st.number_input("Estimated Capital Outlay ($)", 5000, 300000, 50000, step=5000)
    
    # Calculate simulated property
    sim_specs = current_specs.copy()
    sim_specs['grade'] = min(13, grade + grade_bump)
    sim_specs['condition'] = min(5, condition + cond_bump)
    sim_specs['sqft_living'] = sqft_living + extra_sqft
    sim_specs['sqft_above'] = sqft_above + extra_sqft
    
    sim_valuation = run_valuation(sim_specs)
    val_lift = sim_valuation - base_valuation
    net_gain = val_lift - reno_cost
    roi_percent = (net_gain / reno_cost) * 100 if reno_cost > 0 else 0
    
    with c_sim2:
        # Plotly Comparison Bar Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Current State", "Post-Renovation", "Total Invested (Value + Budget)"],
            y=[base_valuation, sim_valuation, base_valuation + reno_cost],
            marker_color=['#94A3B8', '#2563EB', '#F59E0B'],
            text=[f"${base_valuation:,.0f}", f"${sim_valuation:,.0f}", f"${(base_valuation + reno_cost):,.0f}"],
            textposition='auto',
        ))
        fig.update_layout(
            title="Capital Appreciation Analysis",
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric(
            label="Estimated Net Value Creation",
            value=f"${val_lift:,.0f}",
            delta=f"${net_gain:+,.0f} Net Profit ({roi_percent:.1f}% ROI)"
        )

# TAB 3: CLIENT EXPORT DOSSIER
with tab3:
    st.markdown('<div class="section-title">Executive Summary & Export</div>', unsafe_allow_html=True)
    
    summary_df = pd.DataFrame([
        {"Metric": "Appraisal Valuation", "Value": f"${base_valuation:,.2f}"},
        {"Metric": "Valuation Lower Bound (15%)", "Value": f"${low_val:,.2f}"},
        {"Metric": "Valuation Upper Bound (15%)", "Value": f"${high_val:,.2f}"},
        {"Metric": "Price Per Square Foot", "Value": f"${price_sqft:,.2f}"},
        {"Metric": "Total Square Footage", "Value": f"{sqft_living:,} sqft"},
        {"Metric": "Construction Grade", "Value": f"{grade} / 13"},
        {"Metric": "Maintenance Condition", "Value": f"{condition} / 5"},
        {"Metric": "Geographic Coordinates", "Value": f"{lat:.4f}, {long:.4f}"},
        {"Metric": "Zip Code Jurisdiction", "Value": str(zipcode)}
    ])
    
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    csv_bytes = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Appraisal Report (CSV)",
        data=csv_bytes,
        file_name=f"valuation_report_{zipcode}.csv",
        mime="text/csv"
    )
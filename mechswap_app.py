import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MechSwap — Industrial Machinery Marketplace",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0b0e1a;
    color: #e8eaf0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0b0e1a; }
::-webkit-scrollbar-thumb { background: #2563eb; border-radius: 3px; }

/* ── Header ── */
.site-header {
    background: linear-gradient(135deg, #0f1629 0%, #0b0e1a 100%);
    border-bottom: 1px solid rgba(37,99,235,0.3);
    padding: 1rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.logo-badge {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.4rem;
    padding: 0.4rem 1rem;
    border-radius: 10px;
    letter-spacing: -0.5px;
}
.logo-badge span { color: #f59e0b; }
.site-tagline {
    font-size: 0.8rem;
    color: #94a3b8;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 40%, #2563eb 100%);
    border-radius: 20px;
    padding: 3.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '⚙️';
    font-size: 200px;
    position: absolute;
    right: -30px;
    top: -30px;
    opacity: 0.06;
}
.hero::after {
    content: '🔧';
    font-size: 150px;
    position: absolute;
    right: 150px;
    bottom: -40px;
    opacity: 0.05;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    color: white;
    margin: 0 0 0.5rem 0;
    line-height: 1.1;
}
.hero h1 span { color: #f59e0b; }
.hero p {
    color: rgba(255,255,255,0.75);
    font-size: 1.1rem;
    max-width: 500px;
    margin: 0;
}
.hero-stats {
    display: flex;
    gap: 2.5rem;
    margin-top: 2rem;
}
.stat-block { text-align: left; }
.stat-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #f59e0b;
    line-height: 1;
}
.stat-label { color: rgba(255,255,255,0.6); font-size: 0.8rem; margin-top: 2px; }

/* ── Section Titles ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #e8eaf0;
    margin: 0 0 0.25rem 0;
}
.section-sub {
    color: #64748b;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}

/* ── Machine Cards ── */
.machine-card {
    background: linear-gradient(145deg, #131929, #0f1629);
    border: 1px solid rgba(37,99,235,0.15);
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.machine-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #2563eb, #f59e0b);
    opacity: 0;
    transition: opacity 0.2s;
}
.machine-card:hover::before { opacity: 1; }
.machine-card:hover { border-color: rgba(37,99,235,0.4); transform: translateY(-1px); }

.card-category {
    display: inline-block;
    background: rgba(37,99,235,0.2);
    color: #60a5fa;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    border: 1px solid rgba(37,99,235,0.3);
}
.card-name {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #e8eaf0;
    margin-bottom: 0.25rem;
}
.card-price {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: #f59e0b;
}
.card-meta {
    color: #64748b;
    font-size: 0.8rem;
    margin-top: 0.4rem;
}
.card-meta span { color: #94a3b8; }
.card-spec {
    background: rgba(255,255,255,0.03);
    border-left: 2px solid #2563eb;
    padding: 0.5rem 0.75rem;
    border-radius: 0 6px 6px 0;
    margin-top: 0.75rem;
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.5;
}
.condition-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-left: 0.5rem;
}
.cond-excellent { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.cond-good { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.cond-fair { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

/* ── Recommendation Cards ── */
.rec-card {
    background: linear-gradient(145deg, #0f1d38, #0b1529);
    border: 1px solid rgba(37,99,235,0.25);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    border-left: 3px solid #2563eb;
}
.rec-score {
    float: right;
    background: rgba(37,99,235,0.2);
    color: #60a5fa;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: 10px;
}
.rec-name {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    color: #e8eaf0;
}
.rec-price { color: #f59e0b; font-weight: 600; font-size: 0.95rem; margin-top: 0.2rem; }
.rec-cat { color: #64748b; font-size: 0.75rem; }

/* ── Sidebar ── */
.sidebar-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

/* ── Streamlit overrides ── */
.stSelectbox > div > div { background: #131929 !important; border-color: rgba(37,99,235,0.3) !important; color: #e8eaf0 !important; }
.stSlider > div > div { color: #e8eaf0 !important; }
[data-testid="stSidebar"] { background: #0f1629 !important; border-right: 1px solid rgba(37,99,235,0.2); }
[data-testid="stSidebar"] * { color: #e8eaf0 !important; }
.stTextInput > div > div > input { background: #131929 !important; border-color: rgba(37,99,235,0.3) !important; color: #e8eaf0 !important; }
button[kind="primary"] { background: #2563eb !important; border: none !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; }
.stMetric { background: #131929; border: 1px solid rgba(37,99,235,0.2); border-radius: 12px; padding: 1rem; }
.stMetric label { color: #64748b !important; font-size: 0.8rem !important; }
.stMetric [data-testid="metric-container"] > div { color: #f59e0b !important; font-family: 'Syne', sans-serif !important; }
div[data-testid="stExpander"] { background: #131929 !important; border-color: rgba(37,99,235,0.2) !important; border-radius: 10px !important; }

/* ── Category Pills ── */
.cat-pill {
    display: inline-block;
    background: rgba(37,99,235,0.12);
    border: 1px solid rgba(37,99,235,0.25);
    color: #60a5fa;
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 0.2rem;
    cursor: pointer;
}

/* ── Divider ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(37,99,235,0.4), transparent);
    margin: 2rem 0;
}

/* ── Tag badge ── */
.years-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(255,255,255,0.05);
    color: #94a3b8;
    font-size: 0.75rem;
    padding: 0.15rem 0.5rem;
    border-radius: 6px;
    margin-left: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    machines = [
        {
            "id": 1, "name": "Automatic Outer Ear Loop Machine", "category": "Mask Making Machine",
            "price": 400000, "location": "Burhanpur, Madhya Pradesh", "years": 0,
            "condition": "Excellent", "seller": "Ni******* ****",
            "specs": "Fully automatic ear loop welding. High production efficiency. Strong and consistent ultrasonic bonding. Capacity: 120/min. Model: 8479.",
            "description": "Designed for efficient and precise attachment of ear loops to disposable surgical masks. Stable performance with high output.",
            "tags": ["automatic", "ultrasonic", "mask", "surgical", "welding", "ear loop"],
            "model": "8479", "capacity": 120
        },
        {
            "id": 2, "name": "Paper Bowl Making Machine", "category": "Paper Dish Making Machines",
            "price": 10000, "location": "India", "years": 1,
            "condition": "Good", "seller": "Seller #1",
            "specs": "Best in condition paper bowl making machine. Suitable for food grade paper products.",
            "description": "Paper bowl forming machine in excellent working condition. Ideal for small-scale paper product manufacturing.",
            "tags": ["paper", "bowl", "food", "packaging", "forming"],
            "model": "N/A", "capacity": None
        },
        {
            "id": 3, "name": "REISHAUER Grinding Machine", "category": "Grinding Machine",
            "price": 1250000, "location": "Switzerland (available in India)", "years": 15,
            "condition": "Good", "seller": "NEW FELLOWS",
            "specs": "High precision profile grinding machine. CNC controlled. Suitable for gear and spline grinding.",
            "description": "Swiss-made high-precision grinding machine for industrial gear and profile applications.",
            "tags": ["grinding", "precision", "CNC", "gear", "Swiss", "profile", "Reishauer"],
            "model": "RZP", "capacity": None
        },
        {
            "id": 4, "name": "Air Cooled Water Chiller", "category": "Chiller",
            "price": 206500, "location": "Bathinda, Punjab", "years": 2,
            "condition": "Good", "seller": "RSIG0035D",
            "specs": "Brand: Reynold. Nominal Cooling Capacity: 35 kW. Power Supply: 400 VAC, 50 Hz. Maximum ambient temp: 45°C.",
            "description": "Industrial air-cooled water chiller in excellent condition. Suitable for plastics, food, and pharma industries.",
            "tags": ["chiller", "cooling", "HVAC", "water", "industrial cooling", "Reynold"],
            "model": "RSIG0035D", "capacity": 35
        },
        {
            "id": 5, "name": "Ultrasonic Machine", "category": "Ultrasonic Testing Machines",
            "price": 111111, "location": "India", "years": 4,
            "condition": "Good", "seller": "WUS",
            "specs": "3 stage ultrasonic machine with oil skimmer and air agitation. Industrial grade cleaning system.",
            "description": "High-frequency ultrasonic cleaning and testing machine with three-stage processing.",
            "tags": ["ultrasonic", "testing", "cleaning", "NDT", "3 stage", "industrial"],
            "model": "WUS-3S", "capacity": None
        },
        {
            "id": 6, "name": "LINE O MATIC High-Speed Paper Ruling & Cutting Machine", "category": "Paper Converting Machinery",
            "price": 850000, "location": "Maharashtra", "years": 10,
            "condition": "Fair", "seller": "N/A",
            "specs": "Brand: LINE O MATIC. Configuration: Roll-to-Sheet. Ruling Type: Mechanical ruling system. High-speed operation.",
            "description": "Industrial paper ruling and cutting machine for notebook and stationery production lines.",
            "tags": ["paper", "cutting", "ruling", "stationery", "notebook", "roll-to-sheet", "LINE O MATIC"],
            "model": "LOM-HS", "capacity": None
        },
        {
            "id": 7, "name": "Colour Mixing Machine (COROB)", "category": "Cleaning & Painting Equipment",
            "price": 40000, "location": "India", "years": 0,
            "condition": "Excellent", "seller": "T11M020203",
            "specs": "COROB Mixer used for mixing colours. Machine is never used — in new condition. Ideal for paint shops.",
            "description": "Brand new COROB colour mixing machine for paint and coating applications.",
            "tags": ["colour", "mixing", "paint", "COROB", "new", "coating"],
            "model": "COROB", "capacity": None
        },
        {
            "id": 8, "name": "Transformer (Industrial)", "category": "Transformer Machine",
            "price": 350000, "location": "India", "years": 10,
            "condition": "Good", "seller": "32/9/08/15",
            "specs": "Transformer is in excellent running condition. High voltage industrial transformer. Well maintained.",
            "description": "Industrial power transformer in excellent running condition. Suitable for factory installations.",
            "tags": ["transformer", "electrical", "power", "industrial", "high voltage"],
            "model": "HVT-32", "capacity": None
        },
        {
            "id": 9, "name": "Press Brake Machine (CNC)", "category": "Press Brake Machines",
            "price": 750000, "location": "Delhi", "years": 5,
            "condition": "Good", "seller": "MetalCraft Industries",
            "specs": "CNC Press Brake. Capacity: 100 Ton. Bed Length: 3200mm. Back gauge: CNC controlled. Estun controller.",
            "description": "CNC hydraulic press brake for precision sheet metal bending operations.",
            "tags": ["press brake", "CNC", "bending", "sheet metal", "hydraulic", "100 ton"],
            "model": "WC67K-100/3200", "capacity": 100
        },
        {
            "id": 10, "name": "Shearing Machine (Hydraulic)", "category": "Shearing Machine",
            "price": 480000, "location": "Gujarat", "years": 7,
            "condition": "Good", "seller": "SteeelFab Co.",
            "specs": "Hydraulic guillotine shearing machine. Capacity: 6mm x 2500mm. Blade clearance adjustable.",
            "description": "Heavy-duty hydraulic shearing machine for cutting sheet metal up to 6mm thickness.",
            "tags": ["shearing", "hydraulic", "sheet metal", "cutting", "guillotine"],
            "model": "QC11Y-6/2500", "capacity": 6
        },
        {
            "id": 11, "name": "CNC Lathe Machine", "category": "CNC Machines",
            "price": 920000, "location": "Pune, Maharashtra", "years": 3,
            "condition": "Excellent", "seller": "PrecisionTech",
            "specs": "CNC Lathe with Fanuc controller. Swing over bed: 400mm. Max turning length: 1000mm. Spindle speed: 4000 RPM.",
            "description": "High-precision CNC lathe for turning operations. Excellent for batch production.",
            "tags": ["CNC", "lathe", "turning", "Fanuc", "precision", "machining"],
            "model": "CK6140", "capacity": None
        },
        {
            "id": 12, "name": "Injection Moulding Machine", "category": "Plastic Machinery",
            "price": 1500000, "location": "Ahmedabad, Gujarat", "years": 6,
            "condition": "Good", "seller": "PlastiPro",
            "specs": "Injection moulding machine. Clamping force: 250 Ton. Screw diameter: 55mm. Shot weight: 350g.",
            "description": "Industrial injection moulding machine for plastic parts production. Suitable for automotive and packaging.",
            "tags": ["injection moulding", "plastic", "250 ton", "polymer", "automotive"],
            "model": "HTF250", "capacity": 250
        },
        {
            "id": 13, "name": "Surgical Mask Making Machine (Full Line)", "category": "Mask Making Machine",
            "price": 2200000, "location": "Surat, Gujarat", "years": 2,
            "condition": "Excellent", "seller": "MedEquip Traders",
            "specs": "Full automatic surgical mask production line. Output: 60-80 masks/min. Includes nose wire feeder, ear loop welder.",
            "description": "Complete surgical mask manufacturing line with nose wire feeder and automatic ear loop welding.",
            "tags": ["surgical mask", "mask making", "automatic", "ear loop", "nose wire", "medical"],
            "model": "SM-80A", "capacity": 80
        },
        {
            "id": 14, "name": "Paper Cup Making Machine", "category": "Paper Dish Making Machines",
            "price": 185000, "location": "Rajkot, Gujarat", "years": 3,
            "condition": "Good", "seller": "PackagePro",
            "specs": "Paper cup forming machine. Speed: 45-55 cups/min. Cup size: 4oz to 16oz. Servo driven.",
            "description": "Automatic paper cup making machine with servo drive system. Suitable for hot and cold cups.",
            "tags": ["paper cup", "forming", "packaging", "servo", "food grade", "paper"],
            "model": "ZB-12", "capacity": 55
        },
        {
            "id": 15, "name": "Industrial Air Compressor (Screw)", "category": "Compressors",
            "price": 320000, "location": "Ludhiana, Punjab", "years": 4,
            "condition": "Good", "seller": "AirTech Solutions",
            "specs": "Screw air compressor. Capacity: 10 HP. Air delivery: 35 CFM. Pressure: 10 bar. Brand: Atlas Copco.",
            "description": "Atlas Copco screw compressor with after cooler and auto drain. Reliable for industrial use.",
            "tags": ["compressor", "screw", "Atlas Copco", "air", "10 bar", "industrial"],
            "model": "GA11", "capacity": 35
        },
    ]
    return pd.DataFrame(machines)

df = load_data()

# ─── Recommendation Engine ───────────────────────────────────────────────────────
@st.cache_data
def build_recommender(df):
    df = df.copy()
    df['text_features'] = (
        df['category'] + ' ' +
        df['name'] + ' ' +
        df['tags'].apply(lambda x: ' '.join(x)) + ' ' +
        df['specs']
    )
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(df['text_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = build_recommender(df)

def get_recommendations(machine_id, n=4):
    idx = df[df['id'] == machine_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [(i, s) for i, s in sim_scores if i != idx][:n]
    results = []
    for i, score in sim_scores:
        row = df.iloc[i]
        results.append({**row.to_dict(), 'score': round(score * 100, 1)})
    return results

def format_price(p):
    if p >= 100000:
        return f"₹{p/100000:.2f}L"
    return f"₹{p:,}"

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="logo-badge">mech<span>swap</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="site-tagline">Industrial Machinery Marketplace</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown('<div class="sidebar-header">🔍 Search & Filter</div>', unsafe_allow_html=True)
    search_query = st.text_input("", placeholder="Search machines, categories...", label_visibility="collapsed")
    
    st.markdown('<div class="sidebar-header" style="margin-top:1.2rem">📂 Category</div>', unsafe_allow_html=True)
    all_categories = ["All Categories"] + sorted(df['category'].unique().tolist())
    selected_cat = st.selectbox("", all_categories, label_visibility="collapsed")
    
    st.markdown('<div class="sidebar-header" style="margin-top:1.2rem">💰 Price Range (₹)</div>', unsafe_allow_html=True)
    price_min, price_max = st.slider("", 0, 3000000, (0, 3000000), step=10000, format="₹%d", label_visibility="collapsed")
    
    st.markdown('<div class="sidebar-header" style="margin-top:1.2rem">🔧 Condition</div>', unsafe_allow_html=True)
    conditions = st.multiselect("", ["Excellent", "Good", "Fair"], default=["Excellent", "Good", "Fair"], label_visibility="collapsed")
    
    st.markdown('<div class="sidebar-header" style="margin-top:1.2rem">📅 Max Years Used</div>', unsafe_allow_html=True)
    max_years = st.slider("", 0, 20, 20, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown('<div class="sidebar-header">📞 Contact MechSwap</div>', unsafe_allow_html=True)
    st.markdown("📧 mechswap09@gmail.com")
    st.markdown("📱 +91 99148-65830")
    st.markdown("📱 +91 99158-65830")

# ─── Session state ───────────────────────────────────────────────────────────────
if 'selected_machine' not in st.session_state:
    st.session_state.selected_machine = None
if 'page' not in st.session_state:
    st.session_state.page = 'listing'

# ─── Filtering Logic ─────────────────────────────────────────────────────────────
def filter_machines():
    filtered = df.copy()
    if selected_cat != "All Categories":
        filtered = filtered[filtered['category'] == selected_cat]
    if search_query:
        q = search_query.lower()
        filtered = filtered[
            filtered['name'].str.lower().str.contains(q) |
            filtered['category'].str.lower().str.contains(q) |
            filtered['specs'].str.lower().str.contains(q) |
            filtered['tags'].apply(lambda tags: any(q in t.lower() for t in tags))
        ]
    filtered = filtered[(filtered['price'] >= price_min) & (filtered['price'] <= price_max)]
    if conditions:
        filtered = filtered[filtered['condition'].isin(conditions)]
    filtered = filtered[filtered['years'] <= max_years]
    return filtered

# ─── Machine Detail View ─────────────────────────────────────────────────────────
def show_detail(machine_id):
    m = df[df['id'] == machine_id].iloc[0]
    
    if st.button("← Back to Listings", type="secondary"):
        st.session_state.page = 'listing'
        st.session_state.selected_machine = None
        st.rerun()
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0f1629, #131929); border-radius: 16px; padding: 2rem; border: 1px solid rgba(37,99,235,0.2); margin: 1rem 0;">
        <div class="card-category">{m['category']}</div>
        <h2 style="font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #e8eaf0; margin: 0.5rem 0;">{m['name']}</h2>
        <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;">
            <span style="font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #f59e0b;">{format_price(m['price'])}</span>
            <span class="condition-badge cond-{'excellent' if m['condition']=='Excellent' else 'good' if m['condition']=='Good' else 'fair'}">{m['condition']}</span>
            <span class="years-badge">📅 {m['years']} years used</span>
        </div>
        <p style="color: #94a3b8; margin: 0 0 1rem 0;">📍 {m['location']} &nbsp;|&nbsp; 👤 {m['seller']}</p>
        <p style="color: #cbd5e1; font-size: 1rem; line-height: 1.7;">{m['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="section-title">📋 Key Specifications</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card-spec" style="font-size: 0.9rem;">{m["specs"]}</div>', unsafe_allow_html=True)
        
        if m['model'] != 'N/A':
            st.markdown(f"""
            <div style="margin-top: 1rem; display: flex; gap: 1rem; flex-wrap: wrap;">
                <div class="rec-card" style="flex: 1; min-width: 140px;">
                    <div style="color: #64748b; font-size: 0.75rem;">Model</div>
                    <div style="color: #e8eaf0; font-weight: 600; font-family: 'Syne', sans-serif;">{m['model']}</div>
                </div>
                {'<div class="rec-card" style="flex: 1; min-width: 140px;"><div style="color: #64748b; font-size: 0.75rem;">Capacity</div><div style="color: #e8eaf0; font-weight: 600; font-family: Syne, sans-serif;">' + str(m['capacity']) + '</div></div>' if m['capacity'] else ''}
            </div>
            """, unsafe_allow_html=True)
        
        # Tags
        tags_html = "".join([f'<span class="cat-pill">{t}</span>' for t in m['tags']])
        st.markdown(f'<div style="margin-top: 1rem;">{tags_html}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #0f1629; border: 1px solid rgba(37,99,235,0.25); border-radius: 14px; padding: 1.5rem;">
            <div style="font-family: 'Syne', sans-serif; font-weight: 700; color: #e8eaf0; margin-bottom: 1rem; font-size: 1.1rem;">📬 Contact Seller</div>
        """, unsafe_allow_html=True)
        st.markdown(f"**Seller:** {m['seller']}")
        st.markdown(f"**Location:** {m['location']}")
        st.button("📞 Request Seller Details", type="primary", use_container_width=True)
        st.button("❤️ Save to Wishlist", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Recommendations
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 AI-Powered Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Similar machines you might be interested in</div>', unsafe_allow_html=True)
    
    recs = get_recommendations(machine_id, n=4)
    rec_cols = st.columns(2)
    for i, rec in enumerate(recs):
        with rec_cols[i % 2]:
            st.markdown(f"""
            <div class="rec-card">
                <span class="rec-score">🎯 {rec['score']}% match</span>
                <div class="rec-cat">{rec['category']}</div>
                <div class="rec-name">{rec['name']}</div>
                <div class="rec-price">{format_price(rec['price'])}</div>
                <div class="card-meta" style="margin-top: 0.3rem;">📍 <span>{rec['location']}</span></div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"View Details", key=f"rec_{rec['id']}", use_container_width=True):
                st.session_state.selected_machine = rec['id']
                st.rerun()

# ─── Main Listing View ───────────────────────────────────────────────────────────
def show_listing():
    # Header
    st.markdown("""
    <div class="site-header">
        <div class="logo-badge">mech<span>swap</span></div>
        <div>
            <div style="font-size: 0.85rem; color: #64748b;">Industrial Machinery</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero
    filtered = filter_machines()
    st.markdown(f"""
    <div class="hero">
        <h1>Global Marketplace for<br><span>Industrial Machinery</span></h1>
        <p>Buy and sell used industrial machinery with confidence. Connect with verified sellers and find the perfect equipment for your business.</p>
        <div class="hero-stats">
            <div class="stat-block">
                <div class="stat-num">{len(df)}</div>
                <div class="stat-label">Active Listings</div>
            </div>
            <div class="stat-block">
                <div class="stat-num">{df['category'].nunique()}</div>
                <div class="stat-label">Categories</div>
            </div>
            <div class="stat-block">
                <div class="stat-num">100%</div>
                <div class="stat-label">Verified Sellers</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Showing Results", f"{len(filtered)} machines")
    with col2:
        if len(filtered):
            st.metric("Avg. Price", f"₹{int(filtered['price'].mean()):,}")
    with col3:
        if len(filtered):
            st.metric("Lowest Price", format_price(int(filtered['price'].min())))
    with col4:
        if len(filtered):
            st.metric("Highest Price", format_price(int(filtered['price'].max())))
    
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    
    # Category quick-filter pills
    cats = df['category'].unique().tolist()
    pills_html = "".join([f'<span class="cat-pill">{c}</span>' for c in cats])
    st.markdown(f'<div style="margin-bottom: 1.5rem;">{pills_html}</div>', unsafe_allow_html=True)
    
    # Results
    if len(filtered) == 0:
        st.warning("No machines found matching your filters. Try adjusting the search criteria.")
        return
    
    st.markdown(f'<div class="section-title">Industrial Equipment</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">{len(filtered)} listings found · Sorted by relevance</div>', unsafe_allow_html=True)
    
    cols = st.columns(3)
    for i, (_, row) in enumerate(filtered.iterrows()):
        with cols[i % 3]:
            cond_class = 'cond-excellent' if row['condition']=='Excellent' else 'cond-good' if row['condition']=='Good' else 'cond-fair'
            st.markdown(f"""
            <div class="machine-card">
                <div class="card-category">{row['category']}</div>
                <div class="card-name">{row['name']}</div>
                <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 0.4rem;">
                    <div class="card-price">{format_price(row['price'])}</div>
                    <span class="years-badge">📅 {row['years']}yr</span>
                </div>
                <div class="card-meta" style="margin-top: 0.5rem;">
                    📍 <span>{row['location']}</span>
                    <span class="condition-badge {cond_class}" style="float: right;">{row['condition']}</span>
                </div>
                <div class="card-spec">{row['specs'][:120]}...</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("View Details →", key=f"view_{row['id']}", use_container_width=True, type="primary"):
                st.session_state.selected_machine = row['id']
                st.session_state.page = 'detail'
                st.rerun()

# ─── Smart Recommendations Panel (sidebar bottom) ────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown('<div class="sidebar-header">🤖 Trending Now</div>', unsafe_allow_html=True)
        trending = df.sample(3)
        for _, t in trending.iterrows():
            st.markdown(f"""
            <div class="rec-card" style="cursor:pointer;">
                <div class="rec-cat">{t['category']}</div>
                <div class="rec-name">{t['name'][:35]}{'...' if len(t['name'])>35 else ''}</div>
                <div class="rec-price">{format_price(t['price'])}</div>
            </div>
            """, unsafe_allow_html=True)

# ─── Routing ─────────────────────────────────────────────────────────────────────
if st.session_state.page == 'detail' and st.session_state.selected_machine:
    show_detail(st.session_state.selected_machine)
else:
    show_listing()

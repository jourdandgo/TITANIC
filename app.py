import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import google.generativeai as genai

# ----------------------------------------------------------------------------
# 1. PAGE & THEME CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Titanic Intelligence",
    page_icon="🚢",
    layout="wide",
)

# REFINED PREMIUM CSS - REMOVING GHOST WRAPPERS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
    
    :root {
        --primary: #4f46e5;
        --secondary: #6366f1;
        --accent: #10b981;
        --text-dark: #0f172a;
        --text-slate: #64748b;
        --border-color: #e2e8f0;
    }

    /* GLOBAL THEME */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
        color: var(--text-dark) !important;
        background-color: #ffffff !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }

    /* TYPOGRAPHY */
    h1, h2, h3 { color: var(--text-dark) !important; font-weight: 900 !important; margin: 0; }
    h1 { font-size: 4.5rem !important; letter-spacing: -3px !important; line-height: 1 !important; margin-bottom: 2rem !important; }
    h2 { font-size: 2.2rem !important; letter-spacing: -1.2px !important; margin-bottom: 1.5rem !important; }
    h3 { font-size: 1.1rem !important; font-weight: 800 !important; color: var(--text-slate) !important; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 1rem !important; }
    
    /* THE CARD SYSTEM - TARGETING NATIVE CONTAINERS */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 20px !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important;
        background: white !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease;
    }
    
    /* Ensure no double borders when containers are nested */
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] {
        box-shadow: none !important;
        border: 1px solid #f1f5f9 !important;
        background: #fcfdfe !important;
    }

    /* METRICS POLISH */
    [data-testid="stMetricValue"] {
        font-size: 34px !important;
        font-weight: 900 !important;
        color: var(--primary) !important;
        letter-spacing: -1px;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-slate) !important;
        font-size: 11px !important;
    }

    /* SIMULATOR RESULT */
    .result-box {
        padding: 30px;
        border-radius: 16px;
        background: #f8fafc;
        border: 1px solid var(--border-color);
        height: 100%;
    }
    .res-text { font-size: 44px !important; font-weight: 950; margin: 0; line-height: 1; }
    
    /* UTILS */
    .stDivider { margin: 3rem 0 !important; opacity: 0.2; }
    .block-container { padding-top: 2rem !important; }
    
    /* Clear custom margins from form elements */
    [data-testid="stForm"] { border: none !important; padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 2. DATA UTILS
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('titanic.csv')
    df['Outcome'] = df['Survived'].map({0: 'Perished', 1: 'Survived'})
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    # Grouping rare titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs')
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df

df = load_data()
model = joblib.load('champion_titanic_model.joblib') if os.path.exists('champion_titanic_model.joblib') else None

# ----------------------------------------------------------------------------
# 3. PREMIUM VISUALIZATIONS
# ----------------------------------------------------------------------------
def plot_donut(val, color, size=240, label="Rate"):
    fig = go.Figure(go.Pie(
        values=[val, 1-val], hole=0.82,
        marker_colors=[color, "#f1f5f9"],
        showlegend=False, hoverinfo="none", textinfo="none"
    ))
    fig.update_layout(
        height=size, margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        annotations=[
            dict(text=f"<b>{val:.0%}</b>", x=0.5, y=0.55, font_size=42, showarrow=False, font_family="Inter", font_color="#0f172a"),
            dict(text=label, x=0.5, y=0.38, font_size=10, showarrow=False, font_family="Inter", font_color="#64748b", font_weight=800)
        ]
    )
    return fig

# ----------------------------------------------------------------------------
# 4. APP LAYOUT
# ----------------------------------------------------------------------------

# HEADER
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 1.5rem; margin-bottom: 2rem; border-bottom: 1px solid #f1f5f9;">
    <div style="display: flex; align-items: center; gap: 10px;">
        <div style="background: var(--primary); color: white; padding: 8px 12px; border-radius: 10px; font-weight: 900; font-size: 18px;">🚢</div>
        <div style="font-size: 18px; font-weight: 900; letter-spacing: -0.5px; text-transform: uppercase;">Titanic Intel</div>
    </div>
    <div style="display: flex; gap: 30px; font-weight: 700; font-size: 12px; color: #64748b;">
        <span>ANALYSIS</span> <span>PREDICTION</span> <span>BENCHMARKS</span>
    </div>
</div>
""", unsafe_allow_html=True)

# HERO SECTION
h_col1, h_col2 = st.columns([1.1, 1])

with h_col1:
    st.markdown('<p style="color: var(--primary); font-weight: 900; letter-spacing: 2px; font-size: 11px; margin-bottom: 10px;">✦ AI ANALYTICS ENGINE</p>', unsafe_allow_html=True)
    st.markdown('<h1>Decoding the<br><span style="color: var(--primary);">Titanic Dataset.</span></h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.1rem; color: var(--text-slate); line-height: 1.6; margin-bottom: 2.5rem;">Uncovering survivors patterns using Random Forest classification.</p>', unsafe_allow_html=True)
    
    hm1, hm2, hm3 = st.columns(3)
    with hm1:
        with st.container(border=True):
            st.metric("Passengers", "891")
    with hm2:
        with st.container(border=True):
            st.metric("Accuracy", "82%")
    with hm3:
        with st.container(border=True):
            st.metric("Model", "R-Forest")

with h_col2:
    fig_3d = px.scatter_3d(
        df.dropna(subset=['Age']), x='Age', y='Pclass', z='Fare',
        color='Outcome', color_discrete_map={'Survived': '#10b981', 'Perished': '#4f46e5'},
        template="plotly_white", opacity=0.7
    )
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=380, scene_camera_eye=dict(x=1.6, y=1.6, z=0.6))
    st.plotly_chart(fig_3d, use_container_width=True, config={'displayModeBar': False})

st.divider()

# EXPLORATORY DATA ANALYSIS
st.markdown('### 🔍 Historical Forensics', unsafe_allow_html=True)
# Main Section Container
with st.container(border=True):
    eda1, eda2, eda3 = st.columns(3)
    
    with eda1:
        with st.container(border=True): # Sub Card
            st.markdown('<h3>Global Survival</h3>', unsafe_allow_html=True)
            st.plotly_chart(plot_donut(df['Survived'].mean(), "#10b981", 220, "RATE"), use_container_width=True)
            st.write(f"**{df['Survived'].sum()} survivors** identifed from the records.")
        
    with eda2:
        with st.container(border=True): # Sub Card
            st.markdown('<h3>Gender Impact</h3>', unsafe_allow_html=True)
            g_df = df.groupby('Sex')['Survived'].mean().reset_index()
            fig_g = px.bar(g_df, x='Sex', y='Survived', color='Sex',
                          color_discrete_map={'female': '#10b981', 'male': '#f1f5f9'},
                          text=g_df['Survived'].apply(lambda x: f'{x:.0%}'), template='plotly_white')
            fig_g.update_layout(showlegend=False, height=200, margin=dict(t=10, b=0, l=0, r=0), yaxis_visible=False, xaxis_title=None)
            st.plotly_chart(fig_g, use_container_width=True)
            st.write("Female passengers were significantly prioritized.")
        
    with eda3:
        with st.container(border=True): # Sub Card
            st.markdown('<h3>Class Hierarchy</h3>', unsafe_allow_html=True)
            c_df = df.groupby('Pclass')['Survived'].mean().reset_index()
            fig_c = px.bar(c_df, x='Pclass', y='Survived', color='Pclass',
                          color_continuous_scale=['#f1f5f9', '#4f46e5'],
                          text=c_df['Survived'].apply(lambda x: f'{x:.0%}'), template='plotly_white')
            fig_c.update_layout(showlegend=False, coloraxis_showscale=False, height=200, margin=dict(t=10, b=0, l=0, r=0), 
                               yaxis_visible=False, xaxis=dict(title=None, tickvals=[1,2,3]))
            st.plotly_chart(fig_c, use_container_width=True)
            st.write("Upper class access impacted rescue speed.")

st.divider()

# PREDICTION SIMULATOR
st.markdown('### 🔮 Survival Predictor', unsafe_allow_html=True)
with st.container(border=True):
    pc1, pc2 = st.columns([1, 1.4])
    
    with pc1:
        with st.form("final_simulator_form"):
            st.markdown("### Profile Settings")
            st.write("")
            s_sex = st.selectbox("GENDER", ["female", "male"])
            s_class = st.radio("CLASS", [1, 2, 3], horizontal=True)
            s_age = st.slider("AGE", 0, 80, 25)
            s_sib = st.number_input("SIBLINGS", 0, 8, 0)
            s_par = st.number_input("PARENTS", 0, 6, 0)
            st.form_submit_button("CALCULATE SURVIVAL FATE", use_container_width=True, type="primary")
        
    with pc2:
        if model:
            # DYNAMIC FEATURE ENGINEERING (Matching train_model.py)
            def get_title(sex, age):
                if sex == 'female':
                    return 'Mrs' if age >= 18 else 'Miss'
                else:
                    return 'Mr' if age >= 18 else 'Master'
            
            def get_fare(pclass):
                if pclass == 1: return 84.0
                if pclass == 2: return 20.0
                return 8.0
            
            row = pd.DataFrame({
                'Pclass': [s_class], 'Sex': [s_sex], 'Age': [s_age],
                'Fare': [get_fare(s_class)], 'Embarked': ['S'],
                'Title': [get_title(s_sex, s_age)], 'FamilySize': [s_sib + s_par + 1]
            })
            prob = model.predict_proba(row)[0][1]
            survived = prob >= 0.5
            color = "#10b981" if survived else "#ef4444"
            
            # Using a simple custom HTML block for result to avoid nested border issues
            st.markdown(f"""
            <div class="result-box">
                <div style="display: flex; align-items: center; gap: 30px;">
                    <div style="flex: 1;">
                        <p class="res-text" style="color: {color} !important;">YOU {'SURVIVED' if survived else 'PERISHED'}</p>
                        <p style="color: #64748b; font-size: 16px; margin-top: 15px;">Model predicts a <b>{prob:.1%} survival chance</b> based on historical patterns.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show the donut chart below the custom block
            st.plotly_chart(plot_donut(prob, color, 300, "CHANCE"), use_container_width=True)

st.divider()

# COMPARISONS
st.markdown('### 📊 Benchmark Analysis', unsafe_allow_html=True)
with st.container(border=True):
    bc1, bc2, bc3 = st.columns(3)
    
    with bc1:
        with st.container(border=True):
            st.write("#### Log. Regression")
            st.metric("Accuracy", "81%")
    with bc2:
        with st.container(border=True):
            st.markdown("<span style='background: #10b981; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: 900;'>CHAMPION</span>", unsafe_allow_html=True)
            st.write("#### Random Forest")
            st.metric("Accuracy", "82%")
    with bc3:
        with st.container(border=True):
            st.write("#### KNN Classifier")
            st.metric("Accuracy", "82%")


# ----------------------------------------------------------------------------
# 5. GEMINI AI ASSISTANT
# ----------------------------------------------------------------------------
st.divider()
st.markdown('### 🤖 Titanic Intelligence Assistant', unsafe_allow_html=True)

# GEMINI SETUP
# Priority: 1. Streamlit Secrets, 2. Environment Variable
api_key = None
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
elif os.getenv("GOOGLE_API_KEY"):
    api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
    model_ai = genai.GenerativeModel('gemini-1.5-flash')
    
    # SYSTEM PROMPT CONTEXT
    data_summary = f"""
    Dataset context:
    - Total passengers: {len(df)}
    - Survival rate: {df['Survived'].mean():.1%}
    - Columns: {', '.join(df.columns)}
    - Summary Stats:
      - Average Age: {df['Age'].mean():.1f}
      - Female Survival: {df[df['Sex']=='female']['Survived'].mean():.1%}
      - Male Survival: {df[df['Sex']=='male']['Survived'].mean():.1%}
    Model context:
    - Algorithm: Random Forest (Best performing)
    - Accuracy: 82%
    - Benchmarked against: Logistic Regression (81%), KNN (82%)
    """
    
    system_instruction = f"""
    You are the 'Titanic Intel' AI Assistant. You help users understand the Titanic dataset, 
    the survival prediction model, and historical context.
    
    Guidelines:
    1. Use the following context to answer: {data_summary}
    2. Be professional, insightful, and concise.
    3. If asked about model predictions, explain that features like Class, Sex, and Age are primary drivers.
    4. Maintain the 'Titanic Intel' brand voice: authoritative yet accessible.
    """

    # CHAT UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask me anything about the Titanic data or model..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                full_prompt = f"{system_instruction}\n\nUser: {prompt}"
                response = model_ai.generate_content(full_prompt)
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Error communicating with Gemini: {e}")
else:
    st.info("To enable the AI Assistant, please set the GOOGLE_API_KEY environment variable.")

st.write("")
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 12px;'>Purified Forensic Dashboard | © 2026 Titanic Intel</p>", unsafe_allow_html=True)
st.write("")

# app.py

import streamlit as st
import numpy as np
from scipy.special import expit  # Sigmoid function
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

# Define the dynamic amplification matrix D
D = np.array([
    [1.0, 0.2, 0.3, -0.1],
    [0.2, 1.0, 0.1, -0.2],
    [0.3, 0.1, 1.0, -0.3],
    [-0.1, -0.2, -0.3, 1.0]
])

# Define weights (eigenvalues) for each SWOT element
w_S = np.array([1.5, 1.4, 1.3, 1.2])
w_W = np.array([1.2, 1.1, 1.3, 1.0])
w_O = np.array([1.4, 1.3, 1.5, 1.2])
w_T = np.array([1.1, 1.2, 1.0, 1.3])

# Define keywords and impact factors for each SWOT category
KEYWORDS = {
    "Leadership": ["leadership", "kepemimpinan"],
    "Influence": ["influence", "pengaruh"],
    "Vision": ["vision", "visi", "pandangan jauh ke depan"],
    "Communication": ["communication", "komunikasi"],
    "Empathy": ["empathy", "empati", "peduli"],
    "Teamwork": ["teamwork", "kerja sama", "kolaborasi"],
    "Conflict Resolution": ["conflict resolution", "resolusi konflik", "penyelesaian konflik"],
    "Strategic Thinking": ["strategic thinking", "berpikir strategis", "strategi"],
    "Problem-Solving": ["problem-solving", "pemecahan masalah"],
    "Decision-Making": ["decision-making", "pengambilan keputusan"],
    "Risk Management": ["risk management", "manajemen risiko"],
    "Goal Orientation": ["goal orientation", "orientasi tujuan"],
    "Time Management": ["time management", "manajemen waktu", "pengaturan waktu"],
    "Accountability": ["accountability", "tanggung jawab"],
    "Resource Management": ["resource management", "manajemen sumber daya"],
    "Integrity": ["integrity", "integritas", "kejujuran"],
    "Resilience": ["resilience", "ketahanan", "daya tahan"],
    "Adaptability": ["adaptability", "adaptasi", "fleksibilitas"],
    "Confidence": ["confidence", "kepercayaan diri", "percaya diri"]
}

# Function to analyze text input for each SWOT category and assign a score
def analyze_text(input_text, keywords):
    score = 0
    input_text = input_text.lower()
    for keyword in keywords:
        score += input_text.count(keyword)
    return score

# Function to calculate entropy for each element
def entropy(values, weights):
    p = (values * weights) / np.sum(values * weights)
    return -np.sum(p * np.log(p + 1e-9))  # Adding a small value to avoid log(0)

# Calculate the KP using the complex formula
def calculate_kp(S_norm, W_norm, O_norm, T_norm, H_S, H_W, H_O, H_T):
    numerator = S_norm * H_S * w_S.sum() + O_norm * H_O * w_O.sum()
    denominator = W_norm * H_W * w_W.sum() + T_norm * H_T * w_T.sum()
    
    # Interaction component
    interaction_sum = 0
    for i in range(len(S_norm)):
        for j in range(len(O_norm)):
            interaction_sum += D[i, j] * S_norm[i] * O_norm[j]
    
    kp_score = np.log(numerator / (denominator + 1e-9)) + interaction_sum
    return kp_score

# Streamlit app layout
st.title("🌟 Comprehensive SWOT-Based Leadership Viability Assessment 🌟")
st.write("**AI Created by Allam Rafi FKUI 2022**")
st.markdown("Analyze your suitability for leadership based on a comprehensive SWOT evaluation.")

# Dynamic input for SWOT texts
st.subheader("📝 Enter Your SWOT Descriptions")
strengths_text = st.text_area("Enter your Strengths", placeholder="Describe your strengths, e.g., 'I am a natural leader who is good at communication.'")
weaknesses_text = st.text_area("Enter your Weaknesses", placeholder="Describe your weaknesses.")
opportunities_text = st.text_area("Enter your Opportunities", placeholder="Describe your opportunities.")
threats_text = st.text_area("Enter your Threats", placeholder="Describe your threats.")

# Check if texts are provided
if strengths_text or weaknesses_text or opportunities_text or threats_text:
    # Step-by-Step Calculation
    # Analyze each SWOT text input for keywords and calculate raw scores
    S = np.array([analyze_text(strengths_text, KEYWORDS[key]) for key in ["Leadership", "Influence", "Vision", "Communication"]])
    W = np.array([analyze_text(weaknesses_text, KEYWORDS[key]) for key in ["Integrity", "Resilience", "Confidence", "Adaptability"]])
    O = np.array([analyze_text(opportunities_text, KEYWORDS[key]) for key in ["Strategic Thinking", "Problem-Solving", "Decision-Making", "Risk Management"]])
    T = np.array([analyze_text(threats_text, KEYWORDS[key]) for key in ["Teamwork", "Conflict Resolution", "Goal Orientation", "Resource Management"]])

    # Step 1: Entropy Calculation
    H_S = entropy(S, w_S)
    H_W = entropy(W, w_W)
    H_O = entropy(O, w_O)
    H_T = entropy(T, w_T)

    # Step 2: Apply sigmoid transformation to normalize values
    S_norm = expit(S * w_S)
    W_norm = expit(W * w_W)
    O_norm = expit(O * w_O)
    T_norm = expit(T * w_T)

    # Step 3: Calculate the viability score
    kp_score = calculate_kp(S_norm, W_norm, O_norm, T_norm, H_S, H_W, H_O, H_T)

    # Display Results
    st.subheader("🏆 Leadership Viability Score")
    st.write(f"### Your Viability Score: **{kp_score:.2f}**")

    # Interpretation of Score
    st.subheader("📈 Interpretation of Your Score")
    if kp_score > 200:
        interpretation = "Outstanding potential for leadership."
    elif kp_score > 100:
        interpretation = "Suitable for leadership with some improvement areas."
    elif kp_score > 50:
        interpretation = "Moderate potential for leadership; requires significant development."
    else:
        interpretation = "Not recommended for leadership without major improvements."
    st.write(f"**{interpretation}**")

    # Visualizations
    components_df = pd.DataFrame({
        "Components": ["Strengths", "Weaknesses", "Opportunities", "Threats"],
        "Normalized Values": [np.mean(S_norm), np.mean(W_norm), np.mean(O_norm), np.mean(T_norm)],
        "Entropy": [H_S, H_W, H_O, H_T]
    })

    # 1. Radar Chart
    fig_radar = px.line_polar(
        components_df, 
        r="Normalized Values", 
        theta="Components", 
        line_close=True, 
        title="2D Radar Chart of SWOT Elements"
    )
    fig_radar.update_traces(fill='toself')
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    st.plotly_chart(fig_radar)

    # 2. Bar Chart
    fig_bar = px.bar(
        components_df, 
        x="Components", 
        y="Normalized Values", 
        color="Normalized Values", 
        labels={'x': 'Components', 'y': 'Normalized Values'}, 
        title="2D Bar Chart of Normalized SWOT Values", 
        color_continuous_scale='Plasma'
    )
    st.plotly_chart(fig_bar)

    # Footer note
    st.write("**AI Created by Allam Rafi FKUI 2022**")
else:
    st.write("Please enter text for at least one SWOT element.")

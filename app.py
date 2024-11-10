# app.py

import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import expit  # Sigmoid function
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import plotly.graph_objects as go

# Load model for semantic similarity
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Define variables and keywords for SWOT analysis
LEADERSHIP_QUALITIES = {
    "Leadership": "Ability to lead and inspire others",
    "Influence": "Capability to influence and motivate",
    "Vision": "Having a clear and inspiring vision",
    "Communication": "Effective communication skills",
    "Empathy": "Understanding and empathy towards others",
    "Teamwork": "Collaboration and teamwork",
    "Conflict Resolution": "Skill in resolving conflicts",
    "Strategic Thinking": "Ability to think strategically",
    "Problem-Solving": "Strong problem-solving skills",
    "Decision-Making": "Effective decision-making skills",
    "Risk Management": "Managing risks effectively",
    "Goal Orientation": "Focused on achieving goals",
    "Time Management": "Efficient time management",
    "Accountability": "Being accountable and responsible",
    "Resource Management": "Managing resources effectively",
    "Integrity": "Integrity and honesty",
    "Resilience": "Resilience and adaptability",
    "Adaptability": "Ability to adapt to changes",
    "Confidence": "Confidence in abilities"
}

# Define dynamic amplification matrix D for interactions
D = np.array([
    [1.0, 0.2, 0.3, -0.1],
    [0.2, 1.0, 0.1, -0.2],
    [0.3, 0.1, 1.0, -0.3],
    [-0.1, -0.2, -0.3, 1.0]
])

# Define eigenvalues for each SWOT element
w_S = np.array([1.5, 1.4, 1.3, 1.2])
w_W = np.array([1.2, 1.1, 1.3, 1.0])
w_O = np.array([1.4, 1.3, 1.5, 1.2])
w_T = np.array([1.1, 1.2, 1.0, 1.3])

# Entropy calculation
def entropy(values, weights):
    p = (values * weights) / np.sum(values * weights)
    return -np.sum(p * np.log(p + 1e-9))  # Adding small value to avoid log(0)

# Leadership viability calculation (KP)
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

# Calculate similarity scores with NLP model
def calculate_leadership_scores(swot_text, model, qualities):
    scores = {}
    for quality, description in qualities.items():
        quality_embedding = model.encode(description, convert_to_tensor=True)
        swot_embedding = model.encode(swot_text, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(swot_embedding, quality_embedding).item()
        scores[quality] = similarity_score * 100  # Scale to 0-100
    return scores

# Streamlit app layout
st.title("🌟 Advanced SWOT-Based Leadership Viability Assessment 🌟")
st.write("**AI Created by Allam Rafi FKUI 2022**")
st.markdown("Analyze your suitability for leadership with NLP and mathematical modeling.")

# Input fields for SWOT descriptions
st.subheader("📝 Enter Your SWOT Descriptions")
strengths_text = st.text_area("Enter your Strengths")
weaknesses_text = st.text_area("Enter your Weaknesses")
opportunities_text = st.text_area("Enter your Opportunities")
threats_text = st.text_area("Enter your Threats")

if strengths_text or weaknesses_text or opportunities_text or threats_text:
    st.subheader("🔍 Leadership Relevance Scores")
    
    # Calculate relevance scores using NLP model
    strengths_scores = calculate_leadership_scores(strengths_text, model, LEADERSHIP_QUALITIES)
    weaknesses_scores = calculate_leadership_scores(weaknesses_text, model, LEADERSHIP_QUALITIES)
    opportunities_scores = calculate_leadership_scores(opportunities_text, model, LEADERSHIP_QUALITIES)
    threats_scores = calculate_leadership_scores(threats_text, model, LEADERSHIP_QUALITIES)

    # Aggregate scores into a DataFrame
    scores_df = pd.DataFrame({
        "Qualities": LEADERSHIP_QUALITIES.keys(),
        "Strengths": list(strengths_scores.values()),
        "Weaknesses": list(weaknesses_scores.values()),
        "Opportunities": list(opportunities_scores.values()),
        "Threats": list(threats_scores.values())
    })

    # Calculate average normalized scores for KP calculation
    S_norm = expit(np.array(list(strengths_scores.values())) * w_S)
    W_norm = expit(np.array(list(weaknesses_scores.values())) * w_W)
    O_norm = expit(np.array(list(opportunities_scores.values())) * w_O)
    T_norm = expit(np.array(list(threats_scores.values())) * w_T)

    # Calculate entropy
    H_S = entropy(S_norm, w_S)
    H_W = entropy(W_norm, w_W)
    H_O = entropy(O_norm, w_O)
    H_T = entropy(T_norm, w_T)

    # Calculate KP score
    kp_score = calculate_kp(S_norm, W_norm, O_norm, T_norm, H_S, H_W, H_O, H_T)
    st.subheader("🏆 Leadership Viability Score")
    st.write(f"### Your Viability Score: **{kp_score:.2f}**")

    # Interpretation
    st.subheader("📈 Interpretation of Your Score")
    if kp_score > 200:
        interpretation = "Outstanding potential for leadership."
    elif kp_score > 100:
        interpretation = "Suitable for leadership with some improvement areas."
    elif kp_score > 50:
        interpretation = "Moderate potential for leadership; requires development."
    else:
        interpretation = "Not recommended for leadership without major improvements."
    st.write(f"**{interpretation}**")

    # 5 Visualizations (Radar, Bar, Scatter, Surface, Heatmap)
    # Radar Chart
    fig_radar = px.line_polar(scores_df, r="Strengths", theta="Qualities", line_close=True, title="Radar Chart of Strengths")
    fig_radar.update_traces(fill='toself')
    st.plotly_chart(fig_radar)

    # Bar Chart
    fig_bar = px.bar(scores_df, x="Qualities", y="Strengths", color="Strengths", title="Bar Chart of Strengths")
    st.plotly_chart(fig_bar)

    # 3D Scatter Plot
    fig_scatter = go.Figure(data=[go.Scatter3d(x=scores_df["Qualities"], y=scores_df["Strengths"], z=scores_df["Weaknesses"], mode='markers', marker=dict(size=10))])
    fig_scatter.update_layout(title="3D Scatter Plot of Strengths and Weaknesses")
    st.plotly_chart(fig_scatter)

    # Surface Plot
    fig_surface = go.Figure(data=[go.Surface(z=D, x=list(scores_df["Qualities"]), y=list(scores_df["Qualities"]))])
    fig_surface.update_layout(title="Surface Plot of Dynamic Amplification Matrix")
    st.plotly_chart(fig_surface)

    # Heatmap
    fig_heatmap = px.imshow(D, title="Heatmap of SWOT Interaction Matrix")
    st.plotly_chart(fig_heatmap)
else:
    st.write("Please enter text for at least one SWOT element.")

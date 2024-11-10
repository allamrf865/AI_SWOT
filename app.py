# app.py

import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import expit  # Sigmoid function
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import plotly.graph_objects as go

# Load the transformer model for semantic similarity
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

model = load_model()

# Define leadership qualities
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

# Define weights (eigenvalues) for each SWOT element
w_S = np.array([1.5, 1.4, 1.3, 1.2])
w_W = np.array([1.2, 1.1, 1.3, 1.0])
w_O = np.array([1.4, 1.3, 1.5, 1.2])
w_T = np.array([1.1, 1.2, 1.0, 1.3])

# Normalize scores safely
def normalize_scores(scores, weights):
    scores_array = np.array(scores)
    if len(scores_array) < len(weights):
        scores_array = np.pad(scores_array, (0, len(weights) - len(scores_array)), 'constant')
    elif len(scores_array) > len(weights):
        scores_array = scores_array[:len(weights)]
    return expit(scores_array * weights)

# Calculate entropy
def entropy(values, weights):
    p = (values * weights) / np.sum(values * weights)
    return -np.sum(p * np.log(p + 1e-9))  # Avoid log(0)

# Leadership viability calculation
def calculate_kp(S_norm, W_norm, O_norm, T_norm, H_S, H_W, H_O, H_T):
    numerator = S_norm * H_S * w_S.sum() + O_norm * H_O * w_O.sum()
    denominator = W_norm * H_W * w_W.sum() + T_norm * H_T * w_T.sum()
    interaction_sum = np.dot(S_norm, O_norm)  # Simplified interaction
    kp_score = np.log(numerator / (denominator + 1e-9)) + interaction_sum
    return float(kp_score)  # Ensure kp_score is a float

# Calculate semantic similarity scores
def calculate_leadership_scores(swot_text, model, qualities, confidence):
    scores = {}
    for quality, description in qualities.items():
        quality_embedding = model.encode(description, convert_to_tensor=True)
        swot_embedding = model.encode(swot_text, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(swot_embedding, quality_embedding).item()
        # Multiply by confidence level (1-10 scale) and scale similarity to 0-100
        scores[quality] = similarity_score * 100 * (confidence / 10)
    return scores

# Streamlit app layout
st.title("🌟 Advanced SWOT-Based Leadership Viability Assessment 🌟")
st.write("**AI Created by Allam Rafi FKUI 2022**")
st.markdown("Analyze your suitability for leadership with NLP and mathematical modeling.")

# Input fields for SWOT descriptions and confidence levels
st.subheader("📝 Enter Your SWOT Descriptions and Confidence Levels (1-10)")

# Strengths
st.write("### Strengths")
strengths_text = st.text_area("Enter your Strengths")
strengths_confidence = st.slider("Confidence Level for Strengths", 1, 10, 5)

# Weaknesses
st.write("### Weaknesses")
weaknesses_text = st.text_area("Enter your Weaknesses")
weaknesses_confidence = st.slider("Confidence Level for Weaknesses", 1, 10, 5)

# Opportunities
st.write("### Opportunities")
opportunities_text = st.text_area("Enter your Opportunities")
opportunities_confidence = st.slider("Confidence Level for Opportunities", 1, 10, 5)

# Threats
st.write("### Threats")
threats_text = st.text_area("Enter your Threats")
threats_confidence = st.slider("Confidence Level for Threats", 1, 10, 5)

# Button for Analyze
if st.button("Analyze"):
    if strengths_text or weaknesses_text or opportunities_text or threats_text:
        # Calculate relevance scores using NLP model, adjusted by confidence
        strengths_scores = calculate_leadership_scores(strengths_text, model, LEADERSHIP_QUALITIES, strengths_confidence)
        weaknesses_scores = calculate_leadership_scores(weaknesses_text, model, LEADERSHIP_QUALITIES, weaknesses_confidence)
        opportunities_scores = calculate_leadership_scores(opportunities_text, model, LEADERSHIP_QUALITIES, opportunities_confidence)
        threats_scores = calculate_leadership_scores(threats_text, model, LEADERSHIP_QUALITIES, threats_confidence)

        # Normalize scores and calculate entropies
        S_norm = normalize_scores(list(strengths_scores.values()), w_S)
        W_norm = normalize_scores(list(weaknesses_scores.values()), w_W)
        O_norm = normalize_scores(list(opportunities_scores.values()), w_O)
        T_norm = normalize_scores(list(threats_scores.values()), w_T)

        H_S = entropy(S_norm, w_S)
        H_W = entropy(W_norm, w_W)
        H_O = entropy(O_norm, w_O)
        H_T = entropy(T_norm, w_T)

        # Calculate KP score
        kp_score = calculate_kp(S_norm, W_norm, O_norm, T_norm, H_S, H_W, H_O, H_T)
        st.subheader("🏆 Leadership Viability Score")
        st.write(f"Your Viability Score: **{kp_score:.2f}**")

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

        # Visualization: Radar, Bar, 3D Scatter, and Surface Plot
        scores_df = pd.DataFrame({
            "Qualities": list(LEADERSHIP_QUALITIES.keys()),
            "Strengths": list(strengths_scores.values()),
            "Weaknesses": list(weaknesses_scores.values()),
            "Opportunities": list(opportunities_scores.values()),
            "Threats": list(threats_scores.values())
        })

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
        fig_surface = go.Figure(data=[go.Surface(z=np.outer(S_norm, O_norm), x=list(scores_df["Qualities"]), y=list(scores_df["Qualities"]))])
        fig_surface.update_layout(title="Surface Plot of SWOT Interaction Matrix")
        st.plotly_chart(fig_surface)

    else:
        st.write("Please enter text for at least one SWOT element.")

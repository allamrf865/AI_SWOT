# app.py

import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import expit  # Sigmoid function
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import plotly.graph_objects as go

# Load NLP model
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
    "Strategic Thinking": "Ability to think strategically",
    "Adaptability": "Ability to adapt to changes",
    "Decision-Making": "Effective decision-making skills",
    "Confidence": "Confidence in abilities",
}

# Helper functions for scoring and normalizing
def normalize_scores(scores, weights):
    scores_array = np.array(list(scores.values()))
    scores_array = np.pad(scores_array, (0, max(0, len(weights) - len(scores_array))), 'constant')
    return expit(scores_array * weights[:len(scores_array)])

def calculate_scores(swot_text, model, qualities):
    scores = {}
    for quality, description in qualities.items():
        quality_embedding = model.encode(description, convert_to_tensor=True)
        swot_embedding = model.encode(swot_text, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(swot_embedding, quality_embedding).item()
        scores[quality] = similarity_score * 100  # Scale to 0-100
    return scores

# Streamlit layout
st.title("🌟 Advanced SWOT Analysis and Leadership Viability 🌟")
st.write("**AI Created by Allam Rafi FKUI 2022**")
st.markdown("Input your SWOT analysis in English or Indonesian for a detailed assessment of leadership suitability and other key qualities.")

# Input text fields for SWOT descriptions
st.subheader("📝 Enter Your SWOT Descriptions")
strengths_text = st.text_area("Enter your Strengths")
weaknesses_text = st.text_area("Enter your Weaknesses")
opportunities_text = st.text_area("Enter your Opportunities")
threats_text = st.text_area("Enter your Threats")

# Confidence sliders (1-10 scale)
st.subheader("🔢 Set Confidence Level for Each Input (1-10)")
strength_conf = st.slider("Strength Confidence", 1, 10, 5)
weakness_conf = st.slider("Weakness Confidence", 1, 10, 5)
opportunity_conf = st.slider("Opportunity Confidence", 1, 10, 5)
threat_conf = st.slider("Threat Confidence", 1, 10, 5)

if st.button("Analyze"):
    if strengths_text or weaknesses_text or opportunities_text or threats_text:
        # Calculate relevance scores using NLP model
        strengths_scores = calculate_scores(strengths_text, model, LEADERSHIP_QUALITIES) if strengths_text else {k: 0 for k in LEADERSHIP_QUALITIES.keys()}
        weaknesses_scores = calculate_scores(weaknesses_text, model, LEADERSHIP_QUALITIES) if weaknesses_text else {k: 0 for k in LEADERSHIP_QUALITIES.keys()}
        opportunities_scores = calculate_scores(opportunities_text, model, LEADERSHIP_QUALITIES) if opportunities_text else {k: 0 for k in LEADERSHIP_QUALITIES.keys()}
        threats_scores = calculate_scores(threats_text, model, LEADERSHIP_QUALITIES) if threats_text else {k: 0 for k in LEADERSHIP_QUALITIES.keys()}

        # Normalize and weigh scores
        weights = np.array([strength_conf, weakness_conf, opportunity_conf, threat_conf])
        S_norm = normalize_scores(strengths_scores, weights)
        W_norm = normalize_scores(weaknesses_scores, weights)
        O_norm = normalize_scores(opportunities_scores, weights)
        T_norm = normalize_scores(threats_scores, weights)

        # Calculate Leadership Viability and additional metrics
        viability_score = np.log(np.sum(S_norm) + np.sum(O_norm)) / (np.sum(W_norm) + np.sum(T_norm) + 1e-9)
        communication_potential = (strengths_scores["Communication"] + weaknesses_scores["Communication"]) / 2
        strategic_creativity = (opportunities_scores["Strategic Thinking"] + strengths_scores["Adaptability"]) / 2
        risk_management = (threats_scores["Risk Management"] + weaknesses_scores["Confidence"]) / 2
        adaptability_score = strengths_scores["Adaptability"] + opportunities_scores["Adaptability"]

        # Display Results
        st.subheader("Leadership Viability and Other Scores")
        st.write(f"Leadership Viability Score: **{viability_score:.2f}**")
        st.write(f"Communication Potential: **{communication_potential:.2f}**")
        st.write(f"Strategic Creativity: **{strategic_creativity:.2f}**")
        st.write(f"Risk Management Ability: **{risk_management:.2f}**")
        st.write(f"Adaptability Score: **{adaptability_score:.2f}**")

        # Visualization: 3D and 2D Colorful and Interactive Charts
        scores_df = pd.DataFrame({
            "Qualities": list(LEADERSHIP_QUALITIES.keys()),
            "Strengths": list(strengths_scores.values()),
            "Weaknesses": list(weaknesses_scores.values()),
            "Opportunities": list(opportunities_scores.values()),
            "Threats": list(threats_scores.values())
        })

        st.subheader("Interactive Visualizations")

        # Radar Chart (2D)
        fig_radar = px.line_polar(scores_df, r="Strengths", theta="Qualities", line_close=True, title="2D Radar Chart")
        fig_radar.update_traces(fill='toself')
        st.plotly_chart(fig_radar)

        # Bar Chart (2D)
        fig_bar = px.bar(scores_df, x="Qualities", y="Strengths", color="Strengths", title="2D Bar Chart of Strengths")
        st.plotly_chart(fig_bar)

        # 3D Scatter Plot
        fig_scatter = go.Figure(data=[go.Scatter3d(x=scores_df["Qualities"], y=scores_df["Strengths"], z=scores_df["Weaknesses"], mode='markers', marker=dict(size=10))])
        fig_scatter.update_layout(title="3D Scatter Plot of Strengths and Weaknesses")
        st.plotly_chart(fig_scatter)

        # Heatmap (2D)
        fig_heatmap = px.imshow(scores_df[["Strengths", "Weaknesses", "Opportunities", "Threats"]].values, labels=dict(x="SWOT Aspects", y="Qualities"), title="Heatmap of SWOT Scores")
        st.plotly_chart(fig_heatmap)

        # Surface Plot (3D)
        fig_surface = go.Figure(data=[go.Surface(z=scores_df[["Strengths", "Weaknesses", "Opportunities", "Threats"]].values)])
        fig_surface.update_layout(title="3D Surface Plot of SWOT Analysis")
        st.plotly_chart(fig_surface)
    else:
        st.warning("Please enter text for at least one SWOT element.")

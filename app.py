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

# Define leadership qualities with sub-variables
LEADERSHIP_QUALITIES = {
    "Communication": {
        "description": "Effective communication skills",
        "sub_variables": ["Verbal Communication", "Non-Verbal Communication", "Written Communication"]
    },
    "Leadership": {
        "description": "Ability to lead and inspire others",
        "sub_variables": ["Visionary Leadership", "Strategic Leadership", "Situational Leadership"]
    },
    "Teamwork": {
        "description": "Collaboration and teamwork",
        "sub_variables": ["Collaboration", "Conflict Resolution", "Delegation"]
    },
    "Problem-Solving": {
        "description": "Strong problem-solving skills",
        "sub_variables": ["Critical Thinking", "Creativity in Solutions", "Analytical Skills"]
    },
    "Decision-Making": {
        "description": "Effective decision-making skills",
        "sub_variables": ["Risk Assessment", "Timely Decisions", "Outcome Evaluation"]
    },
    "Adaptability": {
        "description": "Ability to adapt to changes",
        "sub_variables": ["Flexibility", "Open-Mindedness", "Resilience"]
    },
    "Integrity": {
        "description": "Integrity and honesty",
        "sub_variables": ["Ethical Standards", "Transparency", "Trustworthiness"]
    },
    "Accountability": {
        "description": "Being accountable and responsible",
        "sub_variables": ["Taking Responsibility", "Owning Mistakes", "Reliability"]
    },
    "Empathy": {
        "description": "Understanding and empathy towards others",
        "sub_variables": ["Emotional Intelligence", "Active Listening", "Supportiveness"]
    },
    "Time Management": {
        "description": "Efficient time management",
        "sub_variables": ["Prioritization", "Punctuality", "Efficient Task Execution"]
    },
    "Influence": {
        "description": "Capability to influence and motivate",
        "sub_variables": ["Persuasion", "Motivational Skills", "Inspiring Others"]
    },
    "Strategic Thinking": {
        "description": "Ability to think strategically",
        "sub_variables": ["Long-Term Vision", "Goal Setting", "Resource Planning"]
    },
    "Creativity": {
        "description": "Innovation and creativity",
        "sub_variables": ["Idea Generation", "Out-of-the-Box Thinking", "Creative Problem Solving"]
    },
    "Risk Management": {
        "description": "Managing risks effectively",
        "sub_variables": ["Risk Identification", "Risk Mitigation", "Crisis Management"]
    },
    "Goal Orientation": {
        "description": "Focused on achieving goals",
        "sub_variables": ["Setting Clear Objectives", "Result-Driven", "Commitment to Success"]
    },
    "Resource Management": {
        "description": "Managing resources effectively",
        "sub_variables": ["Budgeting", "Human Resource Management", "Resource Allocation"]
    },
    "Confidence": {
        "description": "Confidence in abilities",
        "sub_variables": ["Self-Assurance", "Assertiveness", "Decision Confidence"]
    },
    "Patience": {
        "description": "Ability to remain patient in challenging situations",
        "sub_variables": ["Tolerance", "Calmness", "Perseverance"]
    },
    "Resilience": {
        "description": "Ability to recover from setbacks",
        "sub_variables": ["Stress Management", "Emotional Resilience", "Bounce-back Ability"]
    },
    "Innovation": {
        "description": "Encouraging and implementing new ideas",
        "sub_variables": ["Encouraging Innovation", "Continuous Improvement", "Embracing Change"]
    }
}

# Helper function to calculate similarity score
def calculate_scores(text, confidence, model, qualities):
    scores = {}
    for quality, details in qualities.items():
        for sub_var in details["sub_variables"]:
            quality_embedding = model.encode(details["description"], convert_to_tensor=True)
            text_embedding = model.encode(text, convert_to_tensor=True)
            similarity_score = util.pytorch_cos_sim(text_embedding, quality_embedding).item()
            scores[sub_var] = similarity_score * confidence / 10  # Adjust by confidence
    return scores

# Normalize scores
def normalize_scores(scores, weights):
    scores_array = np.array(list(scores.values()))
    scores_array = np.pad(scores_array, (0, max(0, len(weights) - len(scores_array))), 'constant')
    return expit(scores_array * weights[:len(scores_array)])

# Streamlit layout
st.title("🌟 Enhanced SWOT Analysis for Leadership Suitability 🌟")
st.write("**AI Created by Allam Rafi FKUI 2022**")
st.markdown("Analyze leadership potential with nested sub-variables for more granular scoring.")

st.subheader("📝 Input Your SWOT Analysis")
st.write("For each primary variable, add up to 3 sub-variables with their respective confidence levels (1-10).")

# Function to handle nested input for each SWOT category
def swot_input(category):
    inputs = []
    for main_var, details in LEADERSHIP_QUALITIES.items():
        st.write(f"**{main_var} - {details['description']}**")
        for sub_var in details["sub_variables"]:
            input_text = st.text_input(f"{category} - {sub_var}", f"Enter a sentence for {sub_var} ({category.lower()})", key=f"{category}_{main_var}_{sub_var}")
            confidence = st.slider(f"Confidence for {sub_var} ({category.lower()})", 1, 10, 5, key=f"{category}_{main_var}_{sub_var}_conf")
            if input_text:
                inputs.append((sub_var, input_text, confidence))
    return inputs

strengths = swot_input("Strength")
weaknesses = swot_input("Weakness")
opportunities = swot_input("Opportunity")
threats = swot_input("Threat")

if st.button("Analyze"):
    # Function to calculate scores for each entry
    def calculate_total_scores(swot_data):
        total_scores = {}
        for sub_var, text, confidence in swot_data:
            scores = calculate_scores(text, confidence, model, LEADERSHIP_QUALITIES)
            for k, v in scores.items():
                total_scores[k] = total_scores.get(k, 0) + v  # Accumulate scores for each sub-variable
        return total_scores

    strengths_scores = calculate_total_scores(strengths)
    weaknesses_scores = calculate_total_scores(weaknesses)
    opportunities_scores = calculate_total_scores(opportunities)
    threats_scores = calculate_total_scores(threats)

    # Define weights
    weights = np.array([3, 3, 2, 2])  # Adjust these based on importance
    S_norm = normalize_scores(strengths_scores, weights)
    W_norm = normalize_scores(weaknesses_scores, weights)
    O_norm = normalize_scores(opportunities_scores, weights)
    T_norm = normalize_scores(threats_scores, weights)

    # Calculate primary and additional scoring metrics
    viability_score = np.log(np.sum(S_norm) + np.sum(O_norm)) / (np.sum(W_norm) + np.sum(T_norm) + 1e-9)
    communication_score = strengths_scores.get("Personal Communication", 0) + weaknesses_scores.get("Interpersonal Communication", 0)
    leadership_score = opportunities_scores.get("Visionary Leadership", 0) + strengths_scores.get("Strategic Leadership", 0)
    adaptability_score = strengths_scores.get("Adaptability", 0) + opportunities_scores.get("Adaptability", 0)

    # Display Results
    st.subheader("Leadership Viability and Other Scores")
    st.write(f"Leadership Viability Score: **{viability_score:.2f}**")
    st.write(f"Communication Score: **{communication_score:.2f}**")
    st.write(f"Leadership Score: **{leadership_score:.2f}**")
    st.write(f"Adaptability Score: **{adaptability_score:.2f}**")

    # Visualization: 3D and 2D charts
    scores_df = pd.DataFrame({
        "Sub-Variables": list(strengths_scores.keys()),
        "Strengths": list(strengths_scores.values()),
        "Weaknesses": list(weaknesses_scores.values()),
        "Opportunities": list(opportunities_scores.values()),
        "Threats": list(threats_scores.values())
    })

    # Radar Chart (2D)
    fig_radar = px.line_polar(scores_df, r="Strengths", theta="Sub-Variables", line_close=True, title="2D Radar Chart")
    fig_radar.update_traces(fill='toself')
    st.plotly_chart(fig_radar)

    # Bar Chart (2D)
    fig_bar = px.bar(scores_df, x="Sub-Variables", y="Strengths", color="Strengths", title="2D Bar Chart of Strengths")
    st.plotly_chart(fig_bar)

    # 3D Scatter Plot
    fig_scatter = go.Figure(data=[go.Scatter3d(x=scores_df["Sub-Variables"], y=scores_df["Strengths"], z=scores_df["Weaknesses"], mode='markers', marker=dict(size=10))])
    fig_scatter.update_layout(title="3D Scatter Plot of Strengths and Weaknesses")
    st.plotly_chart(fig_scatter)

    # Heatmap (2D)
    fig_heatmap = px.imshow(scores_df[["Strengths", "Weaknesses", "Opportunities", "Threats"]].values, labels=dict(x="SWOT Aspects", y="Sub-Variables"), title="Heatmap of SWOT Scores")
    st.plotly_chart(fig_heatmap)

    # Surface Plot (3D)
    fig_surface = go.Figure(data=[go.Surface(z=scores_df[["Strengths", "Weaknesses", "Opportunities", "Threats"]].values)])
    fig_surface.update_layout(title="3D Surface Plot of SWOT Analysis")
    st.plotly_chart(fig_surface)

else:
    st.warning("Please enter text and confidence levels for at least three items in each SWOT category.")

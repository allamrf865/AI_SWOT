# app.py

import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

# Function to generate a 3D radar chart
def generate_3d_radar_chart(scores):
    categories = list(scores.keys())
    values = [data['score'] for data in scores.values()]

    fig = go.Figure(data=go.Scatter3d(
        x=categories,
        y=values,
        z=values,
        mode='markers+lines',
        marker=dict(size=5, color='purple', opacity=0.8),
        line=dict(color='purple', width=2)
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Attributes'),
            yaxis=dict(title='Score'),
            zaxis=dict(title='Level'),
            aspectmode="cube"
        ),
        title="3D Radar Chart of Leadership Attributes",
        showlegend=False
    )
    return fig

# Function to generate a 2D bar chart for score distribution
def generate_bar_chart(scores):
    categories = list(scores.keys())
    values = [data['score'] for data in scores.values()]

    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=values,
        marker=dict(color=values, colorscale='Viridis')
    )])

    fig.update_layout(
        title="Attribute Score Distribution",
        xaxis=dict(title="Attributes"),
        yaxis=dict(title="Score")
    )
    return fig

# Function to generate a 3D scatter plot for attribute correlation
def generate_3d_scatter_plot(scores):
    categories = list(scores.keys())
    values = [data['score'] for data in scores.values()]

    fig = go.Figure(data=go.Scatter3d(
        x=categories,
        y=values,
        z=np.random.normal(size=len(values)),  # Random z-axis for separation in visualization
        mode='markers',
        marker=dict(size=8, color=values, colorscale='Rainbow', opacity=0.8)
    ))

    fig.update_layout(
        title="3D Attribute Correlation Plot",
        scene=dict(
            xaxis=dict(title="Attributes"),
            yaxis=dict(title="Score"),
            zaxis=dict(title="Random Depth"),
            aspectmode="cube"
        )
    )
    return fig

# Function to generate a 3D surface plot for overall suitability
def generate_suitability_surface(scores):
    categories = list(scores.keys())
    values = np.array([data['score'] for data in scores.values()])

    # Generate grid for surface plot
    x, y = np.meshgrid(range(len(categories)), range(len(categories)))
    z = np.tile(values, (len(categories), 1))

    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale='Viridis')])
    fig.update_layout(
        title="3D Suitability Surface Plot",
        scene=dict(
            xaxis=dict(title="Attributes"),
            yaxis=dict(title="Attributes (repeated)"),
            zaxis=dict(title="Score"),
        )
    )
    return fig

# Function to analyze SWOT for leadership suitability
def analyze_swot(strengths, weaknesses, opportunities, threats):
    scores = {
        "Leadership": 0,
        "Decision-Making": 0,
        "Strategic Thinking": 0,
        "Adaptability": 0,
        "Communication": 0,
        "Emotional Intelligence": 0,
        "Problem-Solving": 0,
        "Visionary": 0,
        "Execution": 0,
        "Mentorship": 0
    }

    # Analyzing keywords in both English and Indonesian
    if "leadership" in strengths.lower() or "kepemimpinan" in strengths.lower():
        scores["Leadership"] += 3
    if "strategic thinking" in strengths.lower() or "berpikir strategis" in strengths.lower():
        scores["Strategic Thinking"] += 3
    if "communication" in strengths.lower() or "komunikasi" in strengths.lower():
        scores["Communication"] += 3
    if "adaptability" in strengths.lower() or "adaptasi" in strengths.lower():
        scores["Adaptability"] += 3
    if "problem-solving" in strengths.lower() or "pemecahan masalah" in strengths.lower():
        scores["Problem-Solving"] += 3
    if "emotional intelligence" in strengths.lower() or "kecerdasan emosional" in strengths.lower():
        scores["Emotional Intelligence"] += 3
    if "visionary" in strengths.lower() or "visioner" in strengths.lower():
        scores["Visionary"] += 3
    if "execution" in strengths.lower() or "eksekusi" in strengths.lower():
        scores["Execution"] += 3
    if "mentorship" in strengths.lower() or "pembimbingan" in strengths.lower():
        scores["Mentorship"] += 3

    # Adjust scores based on weaknesses in both English and Indonesian
    if "indecisive" in weaknesses.lower() or "ragu-ragu" in weaknesses.lower():
        scores["Decision-Making"] -= 1
    if "communication" in weaknesses.lower() or "komunikasi" in weaknesses.lower():
        scores["Communication"] -= 2
    if "confidence" in weaknesses.lower() or "kepercayaan diri" in weaknesses.lower():
        scores["Leadership"] -= 2
    if "emotion" in weaknesses.lower() or "emosi" in weaknesses.lower():
        scores["Emotional Intelligence"] -= 2

    # Scale scores to a 1-10 range
    scaler = MinMaxScaler(feature_range=(1, 10))
    score_values = list(scores.values())
    scaled_scores = scaler.fit_transform(np.array(score_values).reshape(-1, 1)).flatten()

    # Categorize scores
    categorized_scores = {}
    for idx, skill in enumerate(scores.keys()):
        score = scaled_scores[idx]
        level = "High" if score >= 7 else "Moderate" if score >= 4 else "Low"
        categorized_scores[skill] = {"score": round(score, 2), "level": level}

    return categorized_scores

# Streamlit app layout
st.title("🌟 Comprehensive SWOT Analysis with Leadership Suitability 🌟")
st.write("**Enter your SWOT details below to get detailed insights and leadership suitability recommendations.**")

# Input fields for SWOT details
strengths = st.text_area("Enter your Strengths (English or Indonesian):", "")
weaknesses = st.text_area("Enter your Weaknesses (English or Indonesian):", "")
opportunities = st.text_area("Enter your Opportunities (English or Indonesian):", "")
threats = st.text_area("Enter your Threats (English or Indonesian):", "")

if st.button("Analyze"):
    with st.spinner("Analyzing your SWOT..."):
        scores = analyze_swot(strengths, weaknesses, opportunities, threats)
        radar_fig = generate_3d_radar_chart(scores)
        bar_chart_fig = generate_bar_chart(scores)
        scatter_plot_fig = generate_3d_scatter_plot(scores)
        suitability_fig = generate_suitability_surface(scores)

    # Display each visualization
    st.subheader("🌐 3D Radar Chart of Leadership Attributes")
    st.plotly_chart(radar_fig, use_container_width=True)

    st.subheader("📊 Bar Chart of Attribute Scores")
    st.plotly_chart(bar_chart_fig, use_container_width=True)

    st.subheader("💡 3D Scatter Plot of Attribute Correlation")
    st.plotly_chart(scatter_plot_fig, use_container_width=True)

    st.subheader("🏆 3D Suitability Surface Plot")
    st.plotly_chart(suitability_fig, use_container_width=True)

    # Final Suitability Recommendation
    total_score = sum([data['score'] for data in scores.values()])
    if total_score >= 70:
        suitability = "Highly Suitable for Leadership"
    elif total_score >= 50:
        suitability = "Moderately Suitable for Leadership"
    else:
        suitability = "Not Suitable for Leadership"
    
    st.write(f"### Leadership Suitability: **{suitability}**")

    st.success("Analysis complete! Explore the insights and visualizations to understand your SWOT profile.")

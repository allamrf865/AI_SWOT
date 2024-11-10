# app.py

import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import plotly.graph_objects as go
import PyPDF2
import pandas as pd

# Function to generate a 3D radar chart using Plotly
@st.cache_data
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
            xaxis=dict(title='Skills'),
            yaxis=dict(title='Score'),
            zaxis=dict(title='Level'),
            aspectmode="cube"
        ),
        title="SWOT Analysis 3D Radar Chart",
        showlegend=False
    )

    # Adding watermark
    fig.add_annotation(
        text="AI Created by Allam Rafi FKUI 2022",
        xref="paper", yref="paper",
        x=0.5, y=-0.2, showarrow=False,
        font=dict(color="gray", size=12)
    )

    return fig

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    text = ""
    for page_num in range(pdf_reader.getNumPages()):
        text += pdf_reader.getPage(page_num).extract_text()
    return text

# Function to assess leadership quality based on specific scores
def assess_leadership_quality(scores):
    leadership_traits = ["Leadership", "Decision Making", "Communication", "Strategic Thinking", "Empathy"]
    leadership_score = sum(scores[trait]["score"] for trait in leadership_traits if trait in scores)
    max_score = len(leadership_traits) * 10  # Max score is 10 per trait
    leadership_quality_percentage = (leadership_score / max_score) * 100

    # Determine the leadership quality level
    if leadership_quality_percentage >= 80:
        quality = "Excellent Leader"
        color = "green"
    elif leadership_quality_percentage >= 60:
        quality = "Good Leader"
        color = "blue"
    elif leadership_quality_percentage >= 40:
        quality = "Moderate Leader"
        color = "orange"
    else:
        quality = "Needs Improvement"
        color = "red"

    return leadership_quality_percentage, quality, color

# Enhanced 3D bar chart for leadership assessment
def generate_3d_leadership_chart(scores):
    categories = ["Leadership", "Decision Making", "Communication", "Strategic Thinking", "Empathy"]
    values = [scores[trait]["score"] for trait in categories]

    fig = go.Figure(data=[go.Bar3d(
        x=categories,
        y=values,
        z=[0]*len(values),
        dx=0.5,
        dy=1,
        dz=values,
        marker=dict(color=values, colorscale='Viridis', showscale=True),
    )])

    fig.update_layout(
        title="3D Leadership Quality Visualization",
        scene=dict(
            xaxis=dict(title="Leadership Attributes"),
            yaxis=dict(title="Score"),
            zaxis=dict(title="Depth"),
            aspectmode="cube"
        )
    )
    return fig

# Explainable AI function - Displaying score breakdown
def explain_score_breakdown(scores):
    breakdown_df = pd.DataFrame(scores).T  # Convert scores dictionary to DataFrame
    breakdown_df["Contribution (%)"] = (breakdown_df["score"] / breakdown_df["score"].sum()) * 100
    breakdown_df = breakdown_df.sort_values(by="Contribution (%)", ascending=False)

    st.write("### Score Contribution Breakdown")
    st.write("This breakdown shows the relative contribution of each leadership trait to the overall score.")
    st.dataframe(breakdown_df)

# Analyze SWOT and categorize scores
@st.cache_data
def analyze_swot(strengths, weaknesses, opportunities, threats):
    scores = {
        "Leadership": 0,
        "Creativity": 0,
        "Communication": 0,
        "Adaptability": 0,
        "Problem-Solving": 0,
        "Analytical": 0,
        "Empathy": 0,
        "Strategic Thinking": 0,
        "Technical Skills": 0,
        "Decision Making": 0
    }

    # Scoring based on strengths
    if "leadership" in strengths.lower():
        scores["Leadership"] += 3
    if "creativity" in strengths.lower():
        scores["Creativity"] += 3
    if "communication" in strengths.lower():
        scores["Communication"] += 3
    if "adaptability" in strengths.lower():
        scores["Adaptability"] += 3
    if "problem-solving" in strengths.lower():
        scores["Problem-Solving"] += 3
    if "analytical" in strengths.lower():
        scores["Analytical"] += 3
    if "empathy" in strengths.lower():
        scores["Empathy"] += 3
    if "strategic thinking" in strengths.lower():
        scores["Strategic Thinking"] += 3
    if "decision making" in strengths.lower():
        scores["Decision Making"] += 3

    # Weaknesses reduce scores
    if "time management" in weaknesses.lower():
        scores["Adaptability"] -= 1
    if "communication" in weaknesses.lower():
        scores["Communication"] -= 2
    if "confidence" in weaknesses.lower():
        scores["Leadership"] -= 2

    # Opportunities boost scores
    if "networking" in opportunities.lower():
        scores["Communication"] += 2
    if "mentorship" in opportunities.lower():
        scores["Leadership"] += 2
    if "skill development" in opportunities.lower():
        scores["Adaptability"] += 2

    # Threats reduce scores
    if "competition" in threats.lower():
        scores["Adaptability"] -= 1
    if "job insecurity" in threats.lower():
        scores["Adaptability"] -= 1

    # Scale scores to a 1-10 range
    scaler = MinMaxScaler(feature_range=(1, 10))
    score_values = list(scores.values())
    scaled_scores = scaler.fit_transform(np.array(score_values).reshape(-1, 1)).flatten()

    categorized_scores = {}
    for idx, skill in enumerate(scores.keys()):
        score = scaled_scores[idx]
        level = "High" if score >= 7 else "Moderate" if score >= 4 else "Low"
        categorized_scores[skill] = {"score": round(score, 2), "level": level}

    return categorized_scores

# Streamlit app layout
st.title("🌟 Enhanced SWOT Analysis with Advanced Leadership Assessment 🌟")
st.write("**Upload your SWOT PDF or enter your SWOT details below to get insights and career recommendations.**")

# File uploader for PDF or text input
uploaded_file = st.file_uploader("Upload a PDF SWOT file (or leave blank to type manually):", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
        st.write("Extracted Text:", extracted_text)
        strengths, weaknesses, opportunities, threats = extracted_text.split('\n')[:4]  # Assumes line-per-SWOT format
else:
    strengths = st.text_area("Enter your Strengths:", "")
    weaknesses = st.text_area("Enter your Weaknesses:", "")
    opportunities = st.text_area("Enter your Opportunities:", "")
    threats = st.text_area("Enter your Threats:", "")

if st.button("Analyze"):
    with st.spinner("Analyzing your SWOT..."):
        scores = analyze_swot(strengths, weaknesses, opportunities, threats)
        radar_fig = generate_3d_radar_chart(scores)
        leadership_quality, leadership_category, category_color = assess_leadership_quality(scores)
        leadership_fig = generate_3d_leadership_chart(scores)

    st.subheader("🌐 3D SWOT Analysis Visualization")
    st.plotly_chart(radar_fig, use_container_width=True)

    st.subheader("🏅 Leadership Quality Assessment")
    st.write(f"### Leadership Quality Score: {leadership_quality:.2f}% - **{leadership_category}**")
    st.progress(leadership_quality / 100)

    st.subheader("📊 3D Leadership Attributes Visualization")
    st.plotly_chart(leadership_fig, use_container_width=True)

    # Explainable AI: Score Breakdown
    st.subheader("📈 Explainable AI (XAI): Score Breakdown")
    explain_score_breakdown(scores)

    st.success("Analysis complete! Rotate the 3D chart, view role suggestions, and explore insights.")

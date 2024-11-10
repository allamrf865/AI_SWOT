# app.py

import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import plotly.graph_objects as go
import io
import PyPDF2

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

# Function to analyze SWOT
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

    if "time management" in weaknesses.lower():
        scores["Adaptability"] -= 1
    if "communication" in weaknesses.lower():
        scores["Communication"] -= 2
    if "confidence" in weaknesses.lower():
        scores["Leadership"] -= 2

    if "networking" in opportunities.lower():
        scores["Communication"] += 2
    if "mentorship" in opportunities.lower():
        scores["Leadership"] += 2
    if "skill development" in opportunities.lower():
        scores["Adaptability"] += 2

    if "competition" in threats.lower():
        scores["Adaptability"] -= 1
    if "job insecurity" in threats.lower():
        scores["Adaptability"] -= 1

    scaler = MinMaxScaler(feature_range=(1, 10))
    score_values = list(scores.values())
    scaled_scores = scaler.fit_transform(np.array(score_values).reshape(-1, 1)).flatten()

    categorized_scores = {}
    for idx, skill in enumerate(scores.keys()):
        score = scaled_scores[idx]
        level = "High" if score >= 7 else "Moderate" if score >= 4 else "Low"
        categorized_scores[skill] = {"score": round(score, 2), "level": level}

    return categorized_scores

# Display recommendations based on scores for a wide range of roles
def display_recommendations(scores):
    role_suggestions = {
        "Leadership": ["Chief Executive Officer", "Director", "Operations Manager", "Human Resources Manager"],
        "Creativity": ["Art Director", "Content Creator", "Product Designer", "Marketing Specialist"],
        "Communication": ["Public Relations Specialist", "Sales Manager", "Customer Service Representative", "Event Planner"],
        "Adaptability": ["Consultant", "Entrepreneur", "Field Agent", "Social Worker"],
        "Problem-Solving": ["Engineer", "Detective", "Consultant", "Project Manager"],
        "Analytical": ["Data Analyst", "Research Scientist", "Economist", "Financial Analyst"],
        "Empathy": ["Psychologist", "Counselor", "Nurse", "Social Worker"],
        "Strategic Thinking": ["Chief Strategy Officer", "Business Analyst", "Policy Advisor", "Urban Planner"],
        "Technical Skills": ["Software Developer", "Network Engineer", "Data Scientist", "IT Specialist"],
        "Decision Making": ["Judge", "Chief Operations Officer", "Surgeon", "Pilot"]
    }
    
    for skill, data in scores.items():
        st.write(f"### {skill} - Score: {data['score']} (Level: {data['level']})")
        if data["level"] == "High":
            roles = role_suggestions.get(skill, [])
            st.success(f"Strong {skill}. Suitable roles include: {', '.join(roles)}.")
        elif data["level"] == "Moderate":
            st.info(f"{skill} is moderate. Consider strengthening through additional training.")
        else:
            st.warning(f"{skill} is currently low. Improvement is recommended.")

# Streamlit app layout
st.title("🌟 Enhanced SWOT Analysis with 3D Visualization 🌟")
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

    st.subheader("🌐 3D SWOT Analysis Visualization")
    st.plotly_chart(radar_fig, use_container_width=True)

    st.subheader("🏅 Role-Based Recommendations")
    display_recommendations(scores)

    st.success("Analysis complete! Rotate the 3D chart, view role suggestions, and explore insights.")

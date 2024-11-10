# app.py

import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import expit  # Sigmoid function
import plotly.express as px
import plotly.graph_objects as go

# Define weights (eigenvalues) for each SWOT element
w_S = np.array([1.5, 1.4, 1.3, 1.2])
w_W = np.array([1.2, 1.1, 1.3, 1.0])
w_O = np.array([1.4, 1.3, 1.5, 1.2])
w_T = np.array([1.1, 1.2, 1.0, 1.3])

# Normalize scores safely
def normalize_scores(scores, weights):
    scores_array = np.array(scores)
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

# Streamlit app layout
st.title("🌟 Advanced SWOT-Based Leadership Viability Assessment 🌟")
st.write("**AI Created by Allam Rafi FKUI 2022**")
st.markdown("Analyze your suitability for leadership with structured SWOT inputs.")

# Input fields for SWOT descriptions
st.subheader("📝 Enter Your SWOT Scores (1-10 for Confidence Level)")

# Strengths
st.write("### Strengths (1-10 Confidence Level)")
S = [st.slider(f"Strength {i+1}", 1, 10, 5) for i in range(4)]

# Weaknesses
st.write("### Weaknesses (1-10 Confidence Level)")
W = [st.slider(f"Weakness {i+1}", 1, 10, 5) for i in range(4)]

# Opportunities
st.write("### Opportunities (1-10 Confidence Level)")
O = [st.slider(f"Opportunity {i+1}", 1, 10, 5) for i in range(4)]

# Threats
st.write("### Threats (1-10 Confidence Level)")
T = [st.slider(f"Threat {i+1}", 1, 10, 5) for i in range(4)]

# Button for Analyze
if st.button("Analyze"):
    # Normalize scores and calculate entropies
    S_norm = normalize_scores(S, w_S)
    W_norm = normalize_scores(W, w_W)
    O_norm = normalize_scores(O, w_O)
    T_norm = normalize_scores(T, w_T)

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

    # Visualization: Radar, Bar, 3D Scatter, and Surface
    scores_df = pd.DataFrame({
        "Qualities": ["Strengths", "Weaknesses", "Opportunities", "Threats"],
        "Scores": [np.mean(S), np.mean(W), np.mean(O), np.mean(T)]
    })

    # Radar Chart
    fig_radar = px.line_polar(scores_df, r="Scores", theta="Qualities", line_close=True, title="Radar Chart of SWOT")
    fig_radar.update_traces(fill='toself')
    st.plotly_chart(fig_radar)

    # Bar Chart
    fig_bar = px.bar(scores_df, x="Qualities", y="Scores", color="Scores", title="Bar Chart of SWOT Scores")
    st.plotly_chart(fig_bar)

    # 3D Scatter Plot
    fig_scatter = go.Figure(data=[go.Scatter3d(x=["Strengths", "Weaknesses", "Opportunities", "Threats"],
                                               y=[np.mean(S), np.mean(W), np.mean(O), np.mean(T)],
                                               z=[H_S, H_W, H_O, H_T],
                                               mode='markers',
                                               marker=dict(size=10))])
    fig_scatter.update_layout(title="3D Scatter Plot of SWOT and Entropy")
    st.plotly_chart(fig_scatter)

    # Surface Plot of SWOT Interaction Matrix
    fig_surface = go.Figure(data=[go.Surface(z=np.outer(S_norm, O_norm), x=["Strengths", "Weaknesses", "Opportunities", "Threats"], y=["Strengths", "Weaknesses", "Opportunities", "Threats"])])
    fig_surface.update_layout(title="Surface Plot of SWOT Interaction Matrix")
    st.plotly_chart(fig_surface)

else:
    st.write("Please adjust each SWOT element's confidence level and click 'Analyze'.")

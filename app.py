# app.py

import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit  # Sigmoid function
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

# Define keywords for each variable
KEYWORDS = {
    "Leadership": ["leadership", "kepemimpinan"],
    "Influence": ["influence", "pengaruh"],
    "Vision": ["vision", "visi"],
    "Communication": ["communication", "komunikasi"],
    "Empathy": ["empathy", "empati"],
    "Teamwork": ["teamwork", "kerja sama"],
    "Conflict Resolution": ["conflict resolution", "resolusi konflik"],
    "Strategic Thinking": ["strategic thinking", "berpikir strategis"],
    "Problem-Solving": ["problem-solving", "pemecahan masalah"],
    "Decision-Making": ["decision-making", "pengambilan keputusan"],
    "Risk Management": ["risk management", "manajemen risiko"],
    "Goal Orientation": ["goal orientation", "orientasi tujuan"],
    "Time Management": ["time management", "manajemen waktu"],
    "Accountability": ["accountability", "tanggung jawab"],
    "Resource Management": ["resource management", "manajemen sumber daya"],
    "Integrity": ["integrity", "integritas"],
    "Resilience": ["resilience", "ketahanan"],
    "Adaptability": ["adaptability", "adaptasi"],
    "Confidence": ["confidence", "kepercayaan diri"]
}

# Calculate entropy for each element
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

# Input fields for SWOT values
st.subheader("📝 Input Your SWOT Values")
st.write("Enter values for each sub-factor from 1 to 10.")
S = np.array([st.slider(f"Strength {i+1}", 1, 10, 5) for i in range(4)])
W = np.array([st.slider(f"Weakness {i+1}", 1, 10, 5) for i in range(4)])
O = np.array([st.slider(f"Opportunity {i+1}", 1, 10, 5) for i in range(4)])
T = np.array([st.slider(f"Threat {i+1}", 1, 10, 5) for i in range(4)])

# Step-by-Step Calculation
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

# 3. 3D Scatter Plot
fig_scatter = go.Figure(data=[go.Scatter3d(
    x=components_df["Components"],
    y=components_df["Normalized Values"],
    z=components_df["Entropy"],
    mode='markers',
    marker=dict(
        size=12,
        color=components_df["Normalized Values"],
        colorscale='Viridis',
        opacity=0.8
    )
)])
fig_scatter.update_layout(
    title="3D Scatter Plot of SWOT Components",
    scene=dict(
        xaxis_title="Components",
        yaxis_title="Normalized Values",
        zaxis_title="Entropy"
    )
)
st.plotly_chart(fig_scatter)

# 4. 3D Surface Plot for Interaction Impact
x, y = np.meshgrid(range(4), range(4))
z = D  # Using D matrix for surface plot
fig_surface = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale="Viridis")])
fig_surface.update_layout(
    title="3D Surface Plot of Dynamic Amplification Matrix",
    scene=dict(
        xaxis_title="SWOT Index",
        yaxis_title="SWOT Index",
        zaxis_title="Interaction Impact"
    )
)
st.plotly_chart(fig_surface)

# 5. Heatmap of SWOT Interaction
fig_heatmap = px.imshow(D, text_auto=True, color_continuous_scale="Inferno", 
                        title="Heatmap of SWOT Interaction Matrix")
fig_heatmap.update_xaxes(title="SWOT Elements")
fig_heatmap.update_yaxes(title="SWOT Elements")
st.plotly_chart(fig_heatmap)

# Display Detailed Data in Table Format
st.subheader("📊 Detailed SWOT Analysis with Impact Factor")
components_df["Impact Factor (%)"] = [np.sum(w_S), np.sum(w_W), np.sum(w_O), np.sum(w_T)]
st.table(components_df)

# Footer note
st.write("**AI Created by Allam Rafi FKUI 2022**")

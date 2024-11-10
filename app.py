import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import expit  # Sigmoid function
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import plotly.graph_objects as go

# Load transformer model for semantic similarity
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
weights = {
    "S": np.array([1.5, 1.4, 1.3, 1.2]),
    "W": np.array([1.2, 1.1, 1.3, 1.0]),
    "O": np.array([1.4, 1.3, 1.5, 1.2]),
    "T": np.array([1.1, 1.2, 1.0, 1.3])
}

# Normalize scores with a stable approach
def normalize_scores(scores, weights):
    scores_array = np.array(scores)
    if len(scores_array) < len(weights):
        scores_array = np.pad(scores_array, (0, len(weights) - len(scores_array)), 'constant')
    elif len(scores_array) > len(weights):
        scores_array = scores_array[:len(weights)]
    return expit(scores_array * weights)

# Leadership viability calculation with simplified stable approach
def calculate_kp(S_norm, W_norm, O_norm, T_norm, H_S, H_W, H_O, H_T):
    epsilon = 1e-9  # Small value to prevent division by zero
    numerator = np.sum(S_norm) * H_S + np.sum(O_norm) * H_O
    denominator = np.sum(W_norm) * H_W + np.sum(T_norm) * H_T + epsilon
    kp_score = np.log(numerator / denominator + epsilon)
    return kp_score

# Calculate semantic similarity scores with confidence adjustment
def calculate_leadership_scores(swot_text, model, qualities, confidence):
    scores = {}
    for quality, description in qualities.items():
        if swot_text.strip():  # Only calculate if there's valid input
            quality_embedding = model.encode(description, convert_to_tensor=True)
            swot_embedding = model.encode(swot_text, convert_to_tensor=True)
            similarity_score = util.pytorch_cos_sim(swot_embedding, quality_embedding).item()
            scores[quality] = similarity_score * 100 * (confidence / 10) if similarity_score > 0 else 0
        else:
            scores[quality] = 0
    return scores

# Streamlit app layout
st.title("🌟 Advanced SWOT-Based Leadership Viability Assessment 🌟")
st.write("**AI Created by Allam Rafi FKUI 2022**")
st.markdown("Analyze your suitability for leadership with NLP and mathematical modeling.")

# Input fields for SWOT descriptions and confidence levels
st.subheader("📝 Enter Your SWOT Descriptions with Confidence Levels (1-10)")

# Input function for each category with min/max check
def input_swot_category(category_name):
    st.write(f"### {category_name}")
    entries = []
    min_entries, max_entries = 3, 5
    num_entries = st.number_input(f"Number of entries for {category_name} (min {min_entries}, max {max_entries})", 
                                  min_value=min_entries, max_value=max_entries, value=min_entries)
    for i in range(num_entries):
        text = st.text_area(f"{category_name} aspect #{i + 1}", key=f"{category_name}_text_{i}")
        confidence = st.slider(f"Confidence level for {category_name} aspect #{i + 1}", 1, 10, 5, key=f"{category_name}_conf_{i}")
        entries.append((text, confidence))
    return entries

# Getting entries for each SWOT category
strengths_entries = input_swot_category("Strengths")
weaknesses_entries = input_swot_category("Weaknesses")
opportunities_entries = input_swot_category("Opportunities")
threats_entries = input_swot_category("Threats")

# Process if user clicks "Analyze"
if st.button("Analyze"):
    # Combine text entries and calculate scores
    scores_dict = {}
    impact_factors = {}
    for category, entries, weights_key in [("Strengths", strengths_entries, "S"), 
                                           ("Weaknesses", weaknesses_entries, "W"),
                                           ("Opportunities", opportunities_entries, "O"), 
                                           ("Threats", threats_entries, "T")]:
        combined_text = " ".join([entry[0] for entry in entries if entry[0]])  # Only use non-empty entries
        avg_confidence = np.mean([entry[1] for entry in entries]) if entries else 5  # Default confidence level if no input
        scores_dict[category] = calculate_leadership_scores(combined_text, model, LEADERSHIP_QUALITIES, avg_confidence)
        
        # Calculate Impact Factor for each entry
        category_impact = []
        for text, confidence in entries:
            if text.strip():
                impact_score = calculate_leadership_scores(text, model, LEADERSHIP_QUALITIES, confidence)
                category_impact.append(impact_score)
        impact_factors[category] = category_impact

    # Normalize scores and calculate entropies
    S_norm = normalize_scores(list(scores_dict["Strengths"].values()), weights["S"])
    W_norm = normalize_scores(list(scores_dict["Weaknesses"].values()), weights["W"])
    O_norm = normalize_scores(list(scores_dict["Opportunities"].values()), weights["O"])
    T_norm = normalize_scores(list(scores_dict["Threats"].values()), weights["T"])

    H_S, H_W, H_O, H_T = np.mean(S_norm), np.mean(W_norm), np.mean(O_norm), np.mean(T_norm)

    # Calculate KP score
    kp_score = calculate_kp(S_norm, W_norm, O_norm, T_norm, H_S, H_W, H_O, H_T)
    st.subheader("🏆 Leadership Viability Score")
    st.write(f"Your Viability Score: **{kp_score:.2f}**")

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

    # Visualization Code
    st.subheader("🔍 Visual Analysis of SWOT Impact on Leadership Qualities")

    # Convert scores to DataFrame for visualizations
    scores_df = pd.DataFrame({
        "Qualities": list(LEADERSHIP_QUALITIES.keys()),
        "Strengths": list(scores_dict["Strengths"].values()),
        "Weaknesses": list(scores_dict["Weaknesses"].values()),
        "Opportunities": list(scores_dict["Opportunities"].values()),
        "Threats": list(scores_dict["Threats"].values())
    })

    # Radar Charts for Each SWOT Category
    for category in ["Strengths", "Weaknesses", "Opportunities", "Threats"]:
        fig = px.line_polar(scores_df, r=category, theta="Qualities", line_close=True, title=f"Radar Chart of {category}")
        fig.update_traces(fill='toself')
        st.plotly_chart(fig)

    # Bar Charts for Each SWOT Category
    for category in ["Strengths", "Weaknesses", "Opportunities", "Threats"]:
        fig = px.bar(scores_df, x="Qualities", y=category, title=f"Bar Chart of {category}")
        st.plotly_chart(fig)

    # 3D Scatter Plot
    fig_scatter = go.Figure(data=[go.Scatter3d(
        x=scores_df["Strengths"], y=scores_df["Weaknesses"], z=scores_df["Opportunities"],
        mode='markers', marker=dict(size=5)
    )])
    fig_scatter.update_layout(title="3D Scatter Plot of Strengths, Weaknesses, and Opportunities")
    st.plotly_chart(fig_scatter)

    # 3D Surface Plot
    fig_surface = go.Figure(data=[go.Surface(z=scores_df.values[:, 1:], x=scores_df["Qualities"], y=scores_df.columns[1:])])
    fig_surface.update_layout(title="3D Surface Plot of SWOT Interaction")
    st.plotly_chart(fig_surface)

    # Heatmap
    fig_heatmap = px.imshow(scores_df.values[:, 1:], title="Heatmap of SWOT Scores")
    st.plotly_chart(fig_heatmap)

    # Pie Charts for Impact Factors
    for category, impacts in impact_factors.items():
        if impacts:
            for i, impact_score in enumerate(impacts):
                impact_df = pd.DataFrame(impact_score.items(), columns=["Quality", "Impact"])
                fig_pie = px.pie(impact_df, values="Impact", names="Quality", title=f"{category} Impact Factor - Entry #{i + 1}")
                st.plotly_chart(fig_pie)

    # Line Chart Comparing Average Scores
    avg_scores = scores_df[["Strengths", "Weaknesses", "Opportunities", "Threats"]].mean()
    fig_line = px.line(x=avg_scores.index, y=avg_scores.values, markers=True, title="Average Score Comparison Across Categories")
    fig_line.update_xaxes(title="SWOT Category")
    fig_line.update_yaxes(title="Average Score")
    st.plotly_chart(fig_line)

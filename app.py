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

# Define expanded leadership qualities dictionary with 50 qualities across Strengths, Weaknesses, Opportunities, and Threats
LEADERSHIP_QUALITIES = {
    # Strengths
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
    "Confidence": "Confidence in abilities",
    "Perseverance": "Staying determined and focused under pressure",

    # Weaknesses
    "Procrastination": "Tendency to delay important tasks",
    "Indecisiveness": "Difficulty in making quick and effective decisions",
    "Overconfidence": "Excessive belief in own abilities, leading to risk",
    "Impatience": "Inability to wait, leading to rushed decisions",
    "Micromanagement": "Inability to delegate tasks effectively",
    "Poor Communication": "Lack of clarity and precision in communication",
    "Conflict Avoidance": "Avoids conflicts, leading to unresolved issues",
    "Inflexibility": "Reluctance to adapt to changes",
    "Perfectionism": "Obsession with perfection, slowing progress",
    "Short-term Focus": "Focuses too much on short-term gains over long-term goals",
    "Poor Time Management": "Struggles to manage time effectively",
    "Unaccountability": "Avoids taking responsibility for actions",
    "Risk Aversion": "Fear of taking necessary risks",
    "Low Self-Confidence": "Lacks confidence in abilities",
    "Low Resilience": "Struggles to recover from setbacks",
    "Dishonesty": "Tendency to be dishonest or hide information",
    "Inconsistent": "Lacks consistency in actions or decisions",
    "Lack of Initiative": "Reluctant to take proactive actions",
    "Low Empathy": "Struggles to understand and share others' feelings",
    "Poor Problem-Solving": "Difficulty in finding effective solutions",

    # Opportunities
    "Networking": "Ability to build and maintain professional relationships",
    "Professional Development": "Opportunities to gain new skills and knowledge",
    "Mentorship": "Access to experienced mentors for guidance",
    "Innovative Ideas": "Ability to generate and implement new ideas",
    "Technological Advancement": "Utilizes technology to enhance effectiveness",
    "Market Expansion": "Opportunities to reach new audiences or markets",
    "Improving Communication": "Opportunity to strengthen communication skills",
    "Cultural Awareness": "Understanding diverse cultures and perspectives",
    "Conflict Management Training": "Opportunities to improve conflict resolution",
    "Leadership Workshops": "Participating in training sessions for leaders",
    "Strategic Partnerships": "Building alliances that strengthen resources",
    "Access to Resources": "Increased access to resources for projects",
    "Time Management Tools": "Use of tools to improve productivity",
    "Public Speaking": "Opportunities to improve public speaking skills",
    "Personal Branding": "Building a positive personal image",
    "Improving Flexibility": "Becoming more adaptable to change",
    "Learning New Technologies": "Keeping up-to-date with relevant technologies",
    "Building Resilience": "Opportunities to develop emotional resilience",
    "Expanding Team Collaboration": "Opportunities to work with diverse teams",
    "Improving Decision-Making": "Developing faster and more effective decisions",

    # Threats
    "Market Competition": "Presence of competitors in the same market",
    "Economic Downturn": "Impact of an economic recession on leadership success",
    "Time Constraints": "Lack of sufficient time to accomplish goals",
    "Resource Limitations": "Insufficient resources to achieve objectives",
    "Technological Disruptions": "Risk of technology rendering skills obsolete",
    "Health Issues": "Personal health issues that affect performance",
    "Changing Regulations": "New regulations that hinder progress",
    "High Turnover": "Frequent staff changes disrupting team dynamics",
    "Public Criticism": "Negative public feedback affecting reputation",
    "Stress and Burnout": "Risk of exhaustion due to workload",
    "Political Instability": "External political factors affecting the environment",
    "Changing Consumer Preferences": "Adapting to new consumer demands",
    "Cybersecurity Risks": "Threats related to digital security",
    "Environmental Concerns": "Impact of environmental issues on operations",
    "Supply Chain Disruptions": "Interruptions in the supply chain affecting projects",
    "Internal Conflicts": "Conflicts within the team or organization",
    "Lack of Support": "Absence of sufficient support from stakeholders",
    "Budget Constraints": "Insufficient budget for planned initiatives",
    "Negative Public Perception": "Challenges due to poor public image",
    "Legal Issues": "Legal challenges that impact leadership effectiveness"
}

# Function to calculate semantic similarity scores with confidence adjustment
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
st.subheader("📝 Enter Your SWOT Descriptions with Confidence Levels (1-10)")

# Input function for each category with min/max check
def input_swot_category(category_name):
    entries = []
    min_entries, max_entries = 3, 5
    num_entries = st.number_input(f"Number of entries for {category_name} (min {min_entries}, max {max_entries})", 
                                  min_value=min_entries, max_value=max_entries, value=min_entries)
    for i in range(num_entries):
        text = st.text_area(f"{category_name} aspect #{i + 1}", key=f"{category_name}_text_{i}")
        confidence = st.slider(f"Confidence level for {category_name} aspect #{i + 1}", 1, 10, 5, key=f"{category_name}_conf_{i}")
        entries.append((text, confidence))
    return entries

# Collect entries for each category
strengths_entries = input_swot_category("Strengths")
weaknesses_entries = input_swot_category("Weaknesses")
opportunities_entries = input_swot_category("Opportunities")
threats_entries = input_swot_category("Threats")

# Analysis on button click
if st.button("Analyze"):
    impact_factors = {}
    for category, entries in [("Strengths", strengths_entries), 
                              ("Weaknesses", weaknesses_entries), 
                              ("Opportunities", opportunities_entries), 
                              ("Threats", threats_entries)]:
        impact_factors[category] = []
        for text, confidence in entries:
            if text.strip():
                impact_score = calculate_leadership_scores(text, model, LEADERSHIP_QUALITIES, confidence)
                impact_factors[category].append({"Input Text": text, "Impact Score": impact_score})

    # Display the impact factors with input text
    st.subheader("📊 Impact Factors for Each SWOT Entry")
    for category, impacts in impact_factors.items():
        st.write(f"### {category}")
        for i, impact in enumerate(impacts):
            st.write(f"**Entry #{i + 1} - Input Text:** {impact['Input Text']}")
            impact_df = pd.DataFrame(list(impact['Impact Score'].items()), columns=["Quality", "Impact"])
            fig = px.bar(impact_df, x="Quality", y="Impact", title=f"Impact Factor for {category} Entry #{i + 1}")
            st.plotly_chart(fig)

    # Combine all impact scores into one DataFrame for comprehensive visualizations
    all_scores = {category: [] for category in ["Strengths", "Weaknesses", "Opportunities", "Threats"]}
    for category, impacts in impact_factors.items():
        for impact in impacts:
            all_scores[category].append(pd.DataFrame(list(impact['Impact Score'].items()), columns=["Quality", "Impact"]))
    
    # Visualize each SWOT category comprehensively with multiple charts
    for category in ["Strengths", "Weaknesses", "Opportunities", "Threats"]:
        category_scores_df = pd.concat(all_scores[category])
        st.subheader(f"📊 Detailed Visualizations for {category}")

        # Radar Chart
        radar_fig = px.line_polar(category_scores_df, r="Impact", theta="Quality", line_close=True, title=f"{category} - Radar Chart")
        radar_fig.update_traces(fill='toself')
        st.plotly_chart(radar_fig)

        # 3D Scatter Plot
        scatter_fig = go.Figure(data=[go.Scatter3d(
            x=category_scores_df["Quality"],
            y=category_scores_df["Impact"],
            z=category_scores_df["Impact"],  # Duplicate impact for z-axis just for visualization
            mode='markers',
            marker=dict(size=5)
        )])
        scatter_fig.update_layout(title=f"{category} - 3D Scatter Plot")
        st.plotly_chart(scatter_fig)

        # 2D Bar Chart (Grouped by Quality)
        bar_fig = px.bar(category_scores_df, x="Quality", y="Impact", title=f"{category} - 2D Bar Chart")
        st.plotly_chart(bar_fig)

        # Heatmap for Quality Impact
        heatmap_fig = px.imshow([category_scores_df["Impact"].values], labels=dict(x="Quality", y="Impact"), 
                                x=category_scores_df["Quality"].values, title=f"{category} - Heatmap of Impact")
        st.plotly_chart(heatmap_fig)

        # Pie Chart for Distribution of Impact Across Qualities
        pie_fig = px.pie(category_scores_df, values="Impact", names="Quality", title=f"{category} - Pie Chart of Impact")
        st.plotly_chart(pie_fig)

    # Overall 3D Surface Plot for All Categories
    strengths_df = pd.concat(all_scores["Strengths"], ignore_index=True).rename(columns={"Impact": "Strengths"})
    weaknesses_df = pd.concat(all_scores["Weaknesses"], ignore_index=True).rename(columns={"Impact": "Weaknesses"})
    opportunities_df = pd.concat(all_scores["Opportunities"], ignore_index=True).rename(columns={"Impact": "Opportunities"})
    threats_df = pd.concat(all_scores["Threats"], ignore_index=True).rename(columns={"Impact": "Threats"})
    
    combined_df = pd.concat([strengths_df["Quality"], strengths_df["Strengths"], weaknesses_df["Weaknesses"], 
                             opportunities_df["Opportunities"], threats_df["Threats"]], axis=1)

    surface_fig = go.Figure(data=[go.Surface(z=combined_df[["Strengths", "Weaknesses", "Opportunities", "Threats"]].values,
                                             x=combined_df["Quality"], 
                                             y=combined_df.columns[1:])])
    surface_fig.update_layout(title="3D Surface Plot of All SWOT Categories")
    st.plotly_chart(surface_fig)

    # Line Chart Comparing Average Scores Across Categories
    avg_scores = {
        "Strengths": strengths_df["Strengths"].mean(),
        "Weaknesses": weaknesses_df["Weaknesses"].mean(),
        "Opportunities": opportunities_df["Opportunities"].mean(),
        "Threats": threats_df["Threats"].mean()
    }
    avg_scores_df = pd.DataFrame(list(avg_scores.items()), columns=["Category", "Average Impact"])
    line_chart_fig = px.line(avg_scores_df, x="Category", y="Average Impact", markers=True, title="Average Impact Comparison Across Categories")
    st.plotly_chart(line_chart_fig)

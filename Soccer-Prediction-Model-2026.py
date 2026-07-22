import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

st.set_page_config(
    page_title="2026 FIFA World Cup Explorer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== DATA ====================
teams_data = {
    'Team': [
        'Spain', 'France', 'Argentina', 'Brazil', 'England', 'Germany', 'Portugal', 
        'Belgium', 'Netherlands', 'Morocco', 'Uruguay', 'Colombia', 'USA', 'Japan', 
        'Mexico', 'Croatia', 'Senegal', 'Switzerland', 'Canada', 'Australia', 
        'South Korea', 'Saudi Arabia', 'Iran', 'Iraq', 'Uzbekistan', 'Jordan', 
        'Qatar', 'Tunisia', 'Algeria', 'Egypt', 'Ghana', 'Ivory Coast', 
        'South Africa', 'Cabo Verde', 'DR Congo', 'Haiti', 'Panama', 'Curaçao', 
        'Paraguay', 'Ecuador', 'New Zealand', 'Austria', 'Sweden', 'Norway', 
        'Bosnia and Herzegovina', 'Czechia', 'Türkiye'
    ],
    'Confederation': [
        'UEFA', 'UEFA', 'CONMEBOL', 'CONMEBOL', 'UEFA', 'UEFA', 'UEFA',
        'UEFA', 'UEFA', 'CAF', 'CONMEBOL', 'CONMEBOL', 'CONCACAF', 'AFC',
        'CONCACAF', 'UEFA', 'CAF', 'UEFA', 'CONCACAF', 'AFC',
        'AFC', 'AFC', 'AFC', 'AFC', 'AFC', 'AFC',
        'AFC', 'CAF', 'CAF', 'CAF', 'CAF', 'CAF',
        'CAF', 'CAF', 'CAF', 'CONCACAF', 'CONCACAF', 'CONCACAF',
        'CONMEBOL', 'CONMEBOL', 'OFC', 'UEFA', 'UEFA', 'UEFA',
        'UEFA', 'UEFA', 'UEFA'
    ],
    'Appearances_pre_2026': [16,16,18,22,16,20,8,14,11,6,14,6,11,7,17,6,3,12,2,6,11,6,6,1,0,0,1,6,5,3,4,3,3,0,1,1,1,0,8,4,2,8,12,3,1,9,3],
    'Best_Result_pre': [
        'Winners (2010)', 'Winners (1998, 2018)', 'Winners (x3)', 'Winners (x5)', 
        'Winners (1966)', 'Winners (x4)', 'Third place (1966)', 'Third place (2018)',
        'Runners-up (x3)', 'Fourth place (2022)', 'Winners (x2)', 'Quarter-finals (2014)',
        'Third place (1930)', 'Round of 16', 'Quarter-finals (1970, 1986)', 
        'Runners-up (2018)', 'Quarter-finals (2002)', 'Quarter-finals', 'Group stage',
        'Round of 16', 'Fourth place (2002)', 'Round of 16 (1994)', 'Group stage',
        'Group stage (1986)', 'Debut', 'Debut', 'Group stage (2022)', 'Group stage',
        'Round of 16 (2014)', 'Group stage', 'Quarter-finals (2010)', 'Group stage',
        'Group stage', 'Debut', 'Group stage (1974)', 'Group stage (1974)', 
        'Group stage (2018)', 'Debut', 'Quarter-finals (2010)', 'Round of 16 (2006)', 
        'Group stage', 'Third place (1954)', 'Runners-up (1958)', 'Round of 16 (1998)', 
        'Group stage (2014)', 'Runners-up (1934, 1962)', 'Third place (2002)'
    ],
    'Pre_Tournament_Odds': [450,450,950,800,800,1050,1000,1500,2150,5000,3000,3250,5000,6500,6500,3000,5000,8000,20000,20000,8000,15000,20000,30000,40000,40000,20000,15000,5000,10000,8000,10000,20000,50000,30000,40000,30000,50000,15000,15000,50000,15000,5000,3000,20000,15000,8000],
    'Group_Stage_Result_2026': [
        'Advanced', 'Advanced', 'Advanced', 'Advanced', 'Advanced', 'Advanced', 'Advanced',
        'Advanced', 'Advanced', 'Advanced', 'Advanced', 'Advanced', 'Advanced', 'Advanced',
        'Topped Group', 'Advanced', 'Advanced', 'Topped Group', 'Topped Group', 'Advanced',
        'Did not advance', 'Advanced', 'Advanced', 'Advanced', 'Advanced', 'Advanced',
        'Did not advance', 'Advanced', 'Advanced', 'Did not advance', 'Advanced', 'Advanced',
        'Advanced', 'Advanced', 'Advanced', 'Advanced', 'Advanced', 'Advanced', 'Advanced',
        'Advanced', 'Advanced', 'Advanced', 'Advanced', 'Advanced', 'Advanced', 'Advanced',
        'Advanced', 'Advanced'
    ],
    'Actual_2026_Performance': [
        'Champions', 'Semifinals', 'Runners-up', 'Round of 32', 'Quarterfinals', 'Round of 16',
        'Round of 16', 'Quarterfinals', 'Round of 16', 'Quarterfinals', 'Round of 32',
        'Round of 32', 'Round of 16', 'Round of 32', 'Round of 16', 'Round of 16',
        'Round of 32', 'Round of 16', 'Round of 16', 'Round of 32', 'Group stage',
        'Round of 32', 'Round of 32', 'Round of 32', 'Round of 32', 'Round of 32',
        'Group stage', 'Round of 32', 'Round of 32', 'Group stage', 'Round of 32',
        'Round of 32', 'Round of 32', 'Round of 32', 'Round of 32', 'Round of 32',
        'Round of 32', 'Round of 32', 'Round of 32', 'Round of 32', 'Round of 32',
        'Round of 32', 'Round of 16', 'Round of 16', 'Round of 16', 'Round of 32',
        'Round of 32', 'Round of 32'
    ]
}

df = pd.DataFrame(teams_data)

# Calculate metrics
df['Implied_Prob'] = 100 / (100 + df['Pre_Tournament_Odds'])
df['Implied_Prob_Pct'] = (df['Implied_Prob'] * 100).round(1)

# Success score mapping
def get_success_score(result):
    if 'Winners' in result:
        return 5
    elif 'Runners-up' in result:
        return 4
    elif 'Quarter' in result or 'Fourth' in result:
        return 3
    elif 'Round of 16' in result:
        return 2
    else:
        return 1

df['Success_Score'] = df['Best_Result_pre'].apply(get_success_score)

# ==================== SIDEBAR ====================
st.sidebar.title("⚽ 2026 World Cup App")
st.sidebar.markdown("**Interactive Explorer + Predictive Model** (Altair Edition)")

weight_success = st.sidebar.slider("Weight: Historical Success", 0.0, 1.0, 0.40, 0.05)
weight_appearances = st.sidebar.slider("Weight: Appearances", 0.0, 1.0, 0.30, 0.05)
weight_odds = st.sidebar.slider("Weight: Betting Odds", 0.0, 1.0, 0.30, 0.05)

# Normalize weights
total = weight_success + weight_appearances + weight_odds
if total > 0:
    weight_success /= total
    weight_appearances /= total
    weight_odds /= total

# Dynamic Predictive Score
df['Predictive_Score'] = (
    df['Success_Score'] * weight_success * 20 +
    df['Appearances_pre_2026'] * weight_appearances * 2 +
    df['Implied_Prob_Pct'] * weight_odds
).round(2)

df = df.sort_values('Predictive_Score', ascending=False).reset_index(drop=True)

# ==================== MAIN CONTENT ====================
st.title("⚽ 2026 FIFA World Cup Team Explorer & Predictive Model")
st.markdown("### Pre-tournament data • Dynamic predictive scoring • **Altair** visualizations")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Team Data", "🔮 Predictive Model", "📈 Visualizations", "🔍 Team Deep Dive"])

# ==================== TAB 1: DATA ====================
with tab1:
    st.header("All 48 Teams - Complete Dataset")

    col1, col2, col3 = st.columns(3)
    with col1:
        conf_filter = st.multiselect("Confederation", options=df['Confederation'].unique(), default=list(df['Confederation'].unique()))
    with col2:
        adv_filter = st.selectbox("Group Stage Result", ["All", "Advanced", "Did not advance", "Topped Group"])
    with col3:
        min_app = st.slider("Min Appearances (pre-2026)", 0, 25, 0)

    filtered = df[df['Confederation'].isin(conf_filter)]
    if adv_filter != "All":
        filtered = filtered[filtered['Group_Stage_Result_2026'] == adv_filter]
    filtered = filtered[filtered['Appearances_pre_2026'] >= min_app]

    st.dataframe(
        filtered[['Team', 'Confederation', 'Appearances_pre_2026', 'Best_Result_pre', 
                  'Pre_Tournament_Odds', 'Implied_Prob_Pct', 'Predictive_Score',
                  'Group_Stage_Result_2026', 'Actual_2026_Performance']],
        use_container_width=True,
        hide_index=True
    )

# ==================== TAB 2: PREDICTIVE MODEL ====================
with tab2:
    st.header("🔮 Dynamic Predictive Model (Altair)")

    st.markdown(f"""
    **Model**: `Predictive Score = (Success × {weight_success:.2f}) + (Appearances × {weight_appearances:.2f}) + (Implied Prob × {weight_odds:.2f})`
    """)

    st.subheader("Top 15 Teams by Predictive Score")

    top15 = df.head(15).copy()

    # Altair Horizontal Bar Chart
    chart = alt.Chart(top15).mark_bar().encode(
        x=alt.X('Predictive_Score:Q', title='Predictive Score'),
        y=alt.Y('Team:N', sort='-x', title='Team'),
        color=alt.Color('Implied_Prob_Pct:Q', scale=alt.Scale(scheme='viridis'), title='Implied Win %'),
        tooltip=['Team', 'Predictive_Score', 'Implied_Prob_Pct', 'Pre_Tournament_Odds']
    ).properties(
        height=550,
        title="Top Predicted Contenders (Adjust weights in sidebar)"
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

# ==================== TAB 3: VISUALIZATIONS ====================
with tab3:
    st.header("📈 Interactive Visualizations (Altair)")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Implied Win Probability (Top 20)")
        top20 = df.head(20)

        bar_chart = alt.Chart(top20).mark_bar().encode(
            x=alt.X('Implied_Prob_Pct:Q', title='Implied Win Probability (%)'),
            y=alt.Y('Team:N', sort='-x'),
            color=alt.Color('Confederation:N', scale=alt.Scale(scheme='category10')),
            tooltip=['Team', 'Implied_Prob_Pct', 'Pre_Tournament_Odds']
        ).properties(height=500).interactive()

        st.altair_chart(bar_chart, use_container_width=True)

    with col2:
        st.subheader("Experience vs History (Bubble Chart)")
        scatter = alt.Chart(df).mark_circle(size=80).encode(
            x=alt.X('Appearances_pre_2026:Q', title='Appearances before 2026'),
            y=alt.Y('Success_Score:Q', title='Historical Success Score'),
            size=alt.Size('Implied_Prob_Pct:Q', title='Implied Win %', scale=alt.Scale(range=[50, 600])),
            color=alt.Color('Predictive_Score:Q', scale=alt.Scale(scheme='redyellowgreen')),
            tooltip=['Team', 'Appearances_pre_2026', 'Success_Score', 'Implied_Prob_Pct', 'Predictive_Score']
        ).properties(
            height=500,
            title="Bubble Size = Betting Strength | Color = Overall Predictive Score"
        ).interactive()

        st.altair_chart(scatter, use_container_width=True)

# ==================== TAB 4: TEAM DEEP DIVE ====================
with tab4:
    st.header("🔍 Team Deep Dive")

    selected = st.selectbox("Select a team", df['Team'].tolist())
    row = df[df['Team'] == selected].iloc[0]

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Predictive Score", f"{row['Predictive_Score']:.2f}")
        st.metric("Implied Win Probability", f"{row['Implied_Prob_Pct']}%")
        st.metric("Pre-Tournament Odds", f"+{row['Pre_Tournament_Odds']}")
    with c2:
        st.write(f"**Confederation:** {row['Confederation']}")
        st.write(f"**Appearances before 2026:** {row['Appearances_pre_2026']}")
        st.write(f"**Best historical result:** {row['Best_Result_pre']}")
        st.write(f"**Group Stage 2026:** {row['Group_Stage_Result_2026']}")
        st.write(f"**Final 2026 Performance:** {row['Actual_2026_Performance']}")

st.markdown("---")
st.caption("Built with Streamlit + Altair • Fully dynamic predictive model • 2026 World Cup Edition")

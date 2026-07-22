import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson  # pip install scipy if not already installed

st.set_page_config(
    page_title="⚽ Soccer Prediction Model 2026",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Soccer Prediction Model 2026")
st.markdown("**Completely Updated & Robust Version** — Fixed DataFrame Error + Improved Predictions")

# ====================== DATA LOADING (ROBUST) ======================
@st.cache_data(ttl=3600)
def load_team_data():
    """
    Replace this function with your real data source.
    Options:
    - pd.read_csv("teams.csv")
    - API call (e.g. football-data.org, API-Football)
    - Web scraping (BeautifulSoup / requests)
    """
    # ==================== SAMPLE DATA (Replace with your real data) ====================
    sample_teams = [
        {"team": "Manchester City", "league": "Premier League", "matches": 38, "wins": 28, "draws": 7, "losses": 3,
         "goals_for": 96, "goals_against": 34, "points": 91, "form_last_5": "WWWDW"},
        {"team": "Arsenal", "league": "Premier League", "matches": 38, "wins": 26, "draws": 6, "losses": 6,
         "goals_for": 88, "goals_against": 43, "points": 84, "form_last_5": "WDWWW"},
        {"team": "Liverpool", "league": "Premier League", "matches": 38, "wins": 25, "draws": 8, "losses": 5,
         "goals_for": 92, "goals_against": 40, "points": 83, "form_last_5": "DWWLW"},
        {"team": "Real Madrid", "league": "La Liga", "matches": 38, "wins": 29, "draws": 6, "losses": 3,
         "goals_for": 95, "goals_against": 32, "points": 93, "form_last_5": "WWWWW"},
        {"team": "Barcelona", "league": "La Liga", "matches": 38, "wins": 26, "draws": 7, "losses": 5,
         "goals_for": 85, "goals_against": 38, "points": 85, "form_last_5": "WDWLW"},
        # Add as many teams as you want — no length restrictions anymore!
    ]
    
    # ==================== ROBUST DATAFRAME CREATION ====================
    # This is the key fix — list of dicts instead of dict of lists
    df = pd.DataFrame(sample_teams)
    
    # Calculate derived stats safely
    df["goal_difference"] = df["goals_for"] - df["goals_against"]
    df["win_rate"] = (df["wins"] / df["matches"] * 100).round(1)
    df["attack_strength"] = (df["goals_for"] / df["matches"]).round(2)
    df["defense_strength"] = (df["goals_against"] / df["matches"]).round(2)
    
    return df

df = load_team_data()

# ====================== SIDEBAR ======================
st.sidebar.header("⚙️ Settings")
league_filter = st.sidebar.multiselect(
    "Filter by League",
    options=df["league"].unique(),
    default=df["league"].unique()
)

filtered_df = df[df["league"].isin(league_filter)]

st.sidebar.success(f"Loaded {len(filtered_df)} teams")

# ====================== MAIN TABS ======================
tab1, tab2, tab3 = st.tabs(["📊 Team Stats", "🔮 Match Predictor", "📈 Model Insights"])

with tab1:
    st.subheader("Team Statistics")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    
    st.download_button(
        "Download Team Data as CSV",
        filtered_df.to_csv(index=False),
        "soccer_teams_2026.csv",
        "text/csv"
    )

with tab2:
    st.subheader("Match Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox("Home Team", filtered_df["team"].tolist(), index=0)
    with col2:
        away_team = st.selectbox("Away Team", filtered_df["team"].tolist(), index=1)
    
    if home_team == away_team:
        st.warning("Please select two different teams!")
    else:
        home_data = filtered_df[filtered_df["team"] == home_team].iloc[0]
        away_data = filtered_df[filtered_df["team"] == away_team].iloc[0]
        
        # Simple but effective prediction model
        home_attack = home_data["attack_strength"]
        away_defense = away_data["defense_strength"]
        away_attack = away_data["attack_strength"]
        home_defense = home_data["defense_strength"]
        
        # Expected goals (basic model)
        lambda_home = (home_attack * 1.3) / (away_defense + 0.8)   # Home advantage
        lambda_away = (away_attack * 0.9) / (home_defense + 0.8)
        
        # Simulate many matches
        n_simulations = 10000
        home_goals = np.random.poisson(lambda_home, n_simulations)
        away_goals = np.random.poisson(lambda_away, n_simulations)
        
        home_wins = np.sum(home_goals > away_goals)
        draws = np.sum(home_goals == away_goals)
        away_wins = np.sum(home_goals < away_goals)
        
        prob_home = home_wins / n_simulations * 100
        prob_draw = draws / n_simulations * 100
        prob_away = away_wins / n_simulations * 100
        
        st.markdown("### Predicted Outcome")
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Home Win", f"{prob_home:.1f}%")
        col_b.metric("Draw", f"{prob_draw:.1f}%")
        col_c.metric("Away Win", f"{prob_away:.1f}%")
        
        st.progress(int(max(prob_home, prob_draw, prob_away)))
        
        # Most likely score
        most_likely_home = int(np.median(home_goals))
        most_likely_away = int(np.median(away_goals))
        st.success(f"**Most likely score:** {home_team} {most_likely_home} - {most_likely_away} {away_team}")

with tab3:
    st.subheader("Model Insights")
    st.write("**Attack vs Defense Strength**")
    st.bar_chart(filtered_df.set_index("team")[["attack_strength", "defense_strength"]])
    
    st.write("**Current League Table (sorted by points)**")
    st.dataframe(filtered_df.sort_values("points", ascending=False)[["team", "points", "goal_difference", "win_rate"]])

# ====================== FOOTER ======================
st.markdown("---")
st.caption("Updated July 2026 • Robust DataFrame handling • Easy to extend with real data sources")

# Optional: Show raw data button for debugging
if st.checkbox("Show raw team data for debugging"):
    st.json(filtered_df.to_dict(orient="records"))

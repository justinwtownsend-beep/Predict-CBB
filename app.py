import datetime as dt
import re
from io import StringIO
import numpy as np
import pandas as pd
import requests
import streamlit as st

UA_HEADERS = {"User-Agent": "Mozilla/5.0"}
DEFAULT_POSSESSIONS = 68.0

# -------------------------------
# Helpers
# -------------------------------
def clean(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def resolve(name, lookup):
    key = clean(name)
    if key in lookup:
        return lookup[key]
    for k, v in lookup.items():
        if key in k or k in key:
            return v
    raise ValueError(f"Could not resolve team name: {name}")

def pick(*args):
    for a in args:
        if a and str(a).strip():
            return a
    return None


# -------------------------------
# Torvik ratings
# -------------------------------
@st.cache_data(ttl=3600)
def load_torvik(year):
    url = f"https://barttorvik.com/{year}_team_results.csv"
    r = requests.get(url, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), on_bad_lines="skip")

    team_col = next((c for c in df.columns if "team" in c.lower()), df.columns[0])
    df = df.rename(columns={team_col: "TEAM"})
    df["TEAM"] = df["TEAM"].str.strip()

    oe = next(c for c in df.columns if "oe" in c.lower())
    de = next(c for c in df.columns if "de" in c.lower())
    tempo = next((c for c in df.columns if "tempo" in c.lower() or "adj" in c.lower()), None)

    df["ADJ_OE"] = pd.to_numeric(df[oe], errors="coerce")
    df["ADJ_DE"] = pd.to_numeric(df[de], errors="coerce")
    if tempo:
        df["TEMPO"] = pd.to_numeric(df[tempo], errors="coerce")
    else:
        df["TEMPO"] = DEFAULT_POSSESSIONS

    return df[["TEAM", "ADJ_OE", "ADJ_DE", "TEMPO"]]


# -------------------------------
# ESPN slate
# -------------------------------
@st.cache_data(ttl=600)
def get_espn_slate(date):
    date_str = date.strftime("%Y%m%d")
    url = (
        "https://site.web.api.espn.com/apis/v2/sports/basketball/"
        "leagues/mens-college-basketball/scoreboard"
        f"?dates={date_str}&limit=300"
    )

    r = requests.get(url, headers=UA_HEADERS, timeout=30)
    debug = {"status": r.status_code, "url": url}

    if r.status_code != 200:
        return pd.DataFrame(), debug

    data = r.json()
    events = data.get("events", [])
    debug["events"] = len(events)

    rows = []
    for ev in events:
        comp = ev["competitions"][0]
        neutral = comp.get("neutralSite", False)

        away = home = None
        for c in comp["competitors"]:
            team = c["team"]
            name = pick(team.get("shortDisplayName"), team.get("displayName"), team.get("name"))
            if c["homeAway"] == "away":
                away = name
            else:
                home = name

        if away and home:
            rows.append({
                "Away": away,
                "Home": home,
                "Neutral": neutral,
                "Matchup": f"{away} vs {home}" if neutral else f"{away} at {home}"
            })

    return pd.DataFrame(rows), debug


# -------------------------------
# Prediction engine
# -------------------------------
def predict(df, away, home, neutral, hca, sims, sd):
    lookup = {clean(t): t for t in df["TEAM"]}
    away = resolve(away, lookup)
    home = resolve(home, lookup)

    A = df[df.TEAM == away].iloc[0]
    H = df[df.TEAM == home].iloc[0]

    poss = np.mean([A.TEMPO, H.TEMPO])
    away_pts = ((A.ADJ_OE + H.ADJ_DE) / 2) / 100 * poss
    home_pts = ((H.ADJ_OE + A.ADJ_DE) / 2) / 100 * poss

    if not neutral:
        home_pts += hca / 2
        away_pts -= hca / 2

    margin = home_pts - away_pts
    draws = np.random.normal(margin, sd, sims)

    return {
        "Away": away,
        "Home": home,
        "Proj Away": away_pts,
        "Proj Home": home_pts,
        "Proj Total": away_pts + home_pts,
        "Proj Margin (Home)": margin,
        "Home Win %": (draws > 0).mean() * 100
    }


# -------------------------------
# Run slate
# -------------------------------
def run(year, date, sims, hca, sd):
    teams = load_torvik(year)
    slate, debug = get_espn_slate(date)

    if slate.empty:
        return pd.DataFrame(), debug

    out = []
    for _, g in slate.iterrows():
        try:
            p = predict(teams, g.Away, g.Home, g.Neutral, hca, sims, sd)
            p["Matchup"] = g.Matchup
            out.append(p)
        except Exception as e:
            out.append({"Matchup": g.Matchup, "Error": str(e)})

    df = pd.DataFrame(out)
    df["Abs Margin"] = df["Proj Margin (Home)"].abs()
    return df, debug


# -------------------------------
# UI
# -------------------------------
st.set_page_config(layout="wide")
st.title("CBB Slate Predictor (ESPN slate + Torvik ratings)")

with st.sidebar:
    year = st.number_input("Season year (2026 = 2025-26)", value=2026)
    date = st.date_input("Slate date", value=dt.date.today())
    sims = st.slider("Simulations", 1000, 20000, 10000, 1000)
    hca = st.slider("Home court advantage", 0.0, 5.0, 2.5, .5)
    sd = st.slider("Margin SD", 6.0, 18.0, 11.0, .5)
    run_btn = st.button("Run")

if run_btn:
    df, debug = run(year, date, sims, hca, sd)

    st.json(debug)

    if df.empty:
        st.error("No games returned from ESPN for this date")
    else:
        st.dataframe(df.sort_values("Abs Margin", ascending=False), use_container_width=True)

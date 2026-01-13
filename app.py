import datetime as dt
import re
import numpy as np
import pandas as pd
import streamlit as st
import requests
from io import StringIO

# ----------------------------
# Torvik Ratings (safe loader)
# ----------------------------
@st.cache_data(ttl=3600)
def load_torvik_team_results(year):
    url = f"https://barttorvik.com/{year}_team_results.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    text = r.text
    if text.lstrip().startswith("<"):
        raise ValueError("Torvik returned HTML instead of CSV")

    df = pd.read_csv(StringIO(text), engine="python", on_bad_lines="skip")

    # Normalize team column
    team_col = None
    for c in df.columns:
        if c.strip().lower() in {"team", "teams", "teamname"} or c.upper() == "TEAM":
            team_col = c
            break
    if team_col is None:
        team_col = df.columns[0]

    df = df.rename(columns={team_col: "TEAM"})
    df["TEAM"] = df["TEAM"].astype(str).str.strip()
    return df


def find_col(df, patterns):
    for p in patterns:
        rx = re.compile(p, re.IGNORECASE)
        for c in df.columns:
            if rx.search(str(c)):
                return c
    raise KeyError(f"Column not found: {patterns}")


def standardize(df):
    df = df.copy()
    df = df.rename(columns={
        find_col(df, ["AdjOE", "OE"]): "ADJ_OE",
        find_col(df, ["AdjDE", "DE"]): "ADJ_DE",
    })
    try:
        df = df.rename(columns={find_col(df, ["AdjT", "Tempo", "Pace", "Poss"]): "TEMPO"})
    except:
        pass

    for c in ["ADJ_OE", "ADJ_DE", "TEMPO"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ----------------------------
# Schedule (correct table)
# ----------------------------
@st.cache_data(ttl=600)
def get_slate(date):
    url = f"https://barttorvik.com/schedule.php?date={date}&conlimit="
    tables = pd.read_html(url, flavor="bs4")

    slate = None
    for t in tables:
        t.columns = [str(c) for c in t.columns]
        if any("Matchup" in c for c in t.columns):
            slate = t
            break

    if slate is None:
        return pd.DataFrame(), url

    matchup_col = [c for c in slate.columns if "Matchup" in c][0]
    slate = slate.rename(columns={matchup_col: "Matchup"})
    slate = slate[slate["Matchup"].astype(str).str.contains(" at | vs ", regex=True)]
    return slate.reset_index(drop=True), url


# ----------------------------
# Team name matching
# ----------------------------
def clean(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def resolve(name, lookup):
    c = clean(name)
    if c in lookup:
        return lookup[c]
    for k in lookup:
        if c in k or k in c:
            return lookup[k]
    raise ValueError(name)


# ----------------------------
# Prediction engine
# ----------------------------
def predict(df, away, home, neutral, hca, n, sd):
    lookup = {clean(t): t for t in df["TEAM"]}
    a = df[df["TEAM"] == resolve(away, lookup)].iloc[0]
    h = df[df["TEAM"] == resolve(home, lookup)].iloc[0]

    poss = np.nanmean([a.get("TEMPO", 68), h.get("TEMPO", 68)])
    if np.isnan(poss):
        poss = 68

    away_pp = (a["ADJ_OE"] + h["ADJ_DE"]) / 2
    home_pp = (h["ADJ_OE"] + a["ADJ_DE"]) / 2

    away_pts = away_pp / 100 * poss
    home_pts = home_pp / 100 * poss

    if not neutral:
        home_pts += hca / 2
        away_pts -= hca / 2

    margin = home_pts - away_pts
    sims = np.random.normal(margin, sd, n)

    return {
        "Away": away,
        "Home": home,
        "Proj_Away": away_pts,
        "Proj_Home": home_pts,
        "Proj_Total": away_pts + home_pts,
        "Proj_Margin": margin,
        "Home_Win_%": (sims > 0).mean() * 100,
    }


# ----------------------------
# Run slate
# ----------------------------
def run(year, date, n, hca, sd):
    df = standardize(load_torvik_team_results(year))
    slate, url = get_slate(date)

    if slate.empty:
        return pd.DataFrame(), url

    rows = []
    for m in slate["Matchup"]:
        if " vs " in m:
            a, h = m.split(" vs ")
            neutral = True
        else:
            a, h = m.split(" at ")
            neutral = False

        try:
            r = predict(df, a.strip(), h.strip(), neutral, hca, n, sd)
            r["Matchup"] = m
            rows.append(r)
        except:
            pass

    return pd.DataFrame(rows), url


# ----------------------------
# UI
# ----------------------------
st.set_page_config(layout="wide")
st.title("CBB Torvik Slate Predictor")

with st.sidebar:
    season = st.number_input("Season year (2026 = 2025–26)", 2026)
    date = st.date_input("Slate date", dt.date.today()).strftime("%Y%m%d")
    sims = st.slider("Simulations", 1000, 20000, 10000)
    hca = st.slider("Home court advantage", 0.0, 5.0, 2.5)
    sd = st.slider("Margin SD", 6.0, 18.0, 11.0)

data, url = run(season, date, sims, hca, sd)
st.caption(url)

if data.empty:
    st.warning("No games found for this date.")
else:
    st.dataframe(data.sort_values("Proj_Margin", ascending=False))

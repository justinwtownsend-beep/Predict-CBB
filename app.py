# ================================
# CBB PREDICTOR + DRAFTKINGS BET FINDER
# ================================

import datetime as dt
import json
import os
import re
import math
from io import StringIO
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ================================
# CONFIG
# ================================

DATA_DIR = "data"
HISTORY_PATH = os.path.join(DATA_DIR, "history.csv")
STATE_PATH = os.path.join(DATA_DIR, "model_state.json")

UA_HEADERS = {"User-Agent": "Mozilla/5.0"}
ODDS_API_HOST = "https://api.the-odds-api.com"


DEFAULTS = {
    "NATIONAL_AVG_OE": 107.0,
    "LEAGUE_AVG_TEMPO": 68.0,
    "HCA": 2.5,
    "BASE_SD": 11.0,
    "K_CAL": 1.0,
}


# ================================
# STORAGE
# ================================

def ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(STATE_PATH):
        with open(STATE_PATH, "w") as f:
            json.dump(DEFAULTS, f)

    if not os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "w") as f:
            f.write("")


def load_state():
    ensure_storage()
    try:
        with open(STATE_PATH) as f:
            d = json.load(f)
    except:
        d = dict(DEFAULTS)

    for k, v in DEFAULTS.items():
        d.setdefault(k, v)
    return d


def save_state(d):
    ensure_storage()
    with open(STATE_PATH, "w") as f:
        json.dump(d, f)


# ================================
# TEAM NAME MAPPING
# ================================

def clean_name(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s.strip()


def similarity(a, b):
    return SequenceMatcher(None, clean_name(a), clean_name(b)).ratio()


def auto_map_team(name, candidates):
    best = max(candidates, key=lambda t: similarity(name, t))
    score = similarity(name, best)
    return best, score, 0.0


# ================================
# TORVIK RATINGS
# ================================

@st.cache_data(ttl=3600)
def load_torvik(year):
    url = f"https://barttorvik.com/{year}_team_results.csv"
    r = requests.get(url, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    df.columns = [c.upper() for c in df.columns]

    df = df.rename(columns={
        "TEAM": "TEAM",
        "ADJOE": "ADJ_OE",
        "ADJDE": "ADJ_DE",
        "ADJT": "TEMPO"
    })

    return df[["TEAM", "ADJ_OE", "ADJ_DE", "TEMPO"]]


# ================================
# ESPN SLATE
# ================================

@st.cache_data(ttl=600)
def get_espn(date_val):
    d = date_val.strftime("%Y%m%d")
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={d}"
    r = requests.get(url, headers=UA_HEADERS)
    data = r.json()

    rows = []
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        teams = comp["competitors"]

        home = next(t for t in teams if t["homeAway"] == "home")
        away = next(t for t in teams if t["homeAway"] == "away")

        rows.append({
            "Away_ESPN": away["team"]["shortDisplayName"],
            "Home_ESPN": home["team"]["shortDisplayName"],
            "Neutral": int(comp.get("neutralSite", False)),
            "Matchup": f'{away["team"]["shortDisplayName"]} at {home["team"]["shortDisplayName"]}',
        })

    return pd.DataFrame(rows)


# ================================
# MODEL
# ================================

def possessions(a_t, h_t):
    return np.nanmean([a_t, h_t])


def sd_scaled(base, poss, avg=68):
    return base * math.sqrt(poss / avg)


def pp100(oe, opp_de, nat):
    return oe + opp_de - nat


# ================================
# ODDS API (DraftKings)
# ================================

@st.cache_data(ttl=120)
def fetch_dk(date_val):
    key = st.secrets.get("ODDS_API_KEY", "")
    if not key:
        return pd.DataFrame()

    start = dt.datetime(date_val.year, date_val.month, date_val.day, tzinfo=dt.timezone.utc)
    end = start + dt.timedelta(days=1)

    url = f"{ODDS_API_HOST}/v4/sports/basketball_ncaab/odds"

    params = {
        "apiKey": key,
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "american",
        "bookmakers": "draftkings",
        "commenceTimeFrom": start.isoformat().replace("+00:00", "Z"),
        "commenceTimeTo": end.isoformat().replace("+00:00", "Z"),
    }

    r = requests.get(url, params=params)
    data = r.json()

    rows = []

    for g in data:
        home = g["home_team"]
        away = g["away_team"]

        bk = g["bookmakers"][0]

        spread_home = None
        total = None
        odds_h = odds_a = odds_o = odds_u = None

        for m in bk["markets"]:
            if m["key"] == "spreads":
                for o in m["outcomes"]:
                    if o["name"] == home:
                        spread_home = o["point"]
                        odds_h = o["price"]
                    else:
                        odds_a = o["price"]

            if m["key"] == "totals":
                for o in m["outcomes"]:
                    total = o["point"]
                    if o["name"].lower() == "over":
                        odds_o = o["price"]
                    else:
                        odds_u = o["price"]

        rows.append({
            "Home_Line_Team": home,
            "Away_Line_Team": away,
            "Book_Spread_Home": spread_home,
            "Book_Total": total,
            "Odds_H": odds_h,
            "Odds_A": odds_a,
            "Odds_O": odds_o,
            "Odds_U": odds_u,
        })

    return pd.DataFrame(rows)


# ================================
# BET MATH
# ================================

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def p_cover(margin, sd, spread):
    target = abs(spread) if spread < 0 else -abs(spread)
    z = (target - margin) / sd
    return 1 - norm_cdf(z)


def p_over(total, sd, line):
    z = (line - total) / sd
    return 1 - norm_cdf(z)


def breakeven(odds):
    dec = 1 + (100 / abs(odds) if odds < 0 else odds / 100)
    return 1 / dec


def kelly(p, odds, frac=0.25):
    dec = 1 + (100 / abs(odds) if odds < 0 else odds / 100)
    b = dec - 1
    q = 1 - p
    k = (b * p - q) / b
    return max(0, k * frac)


# ================================
# UI
# ================================

st.set_page_config(layout="wide")
st.title("🏀 CBB Predictor + DraftKings Best Bets")

state = load_state()

year = st.sidebar.number_input("Season Year", 2020, 2030, 2026)
date_val = st.sidebar.date_input("Date", dt.date.today())

if st.button("Run Model"):

    teams = load_torvik(year)
    slate = get_espn(date_val)

    preds = []

    for _, g in slate.iterrows():

        away, _ = auto_map_team(g["Away_ESPN"], teams["TEAM"])
        home, _ = auto_map_team(g["Home_ESPN"], teams["TEAM"])

        a = teams[teams["TEAM"] == away].iloc[0]
        h = teams[teams["TEAM"] == home].iloc[0]

        poss = possessions(a["TEMPO"], h["TEMPO"])
        sd = sd_scaled(state["BASE_SD"], poss)

        a_pts = pp100(a["ADJ_OE"], h["ADJ_DE"], state["NATIONAL_AVG_OE"]) / 100 * poss
        h_pts = pp100(h["ADJ_OE"], a["ADJ_DE"], state["NATIONAL_AVG_OE"]) / 100 * poss

        if not g["Neutral"]:
            h_pts += state["HCA"] / 2
            a_pts -= state["HCA"] / 2

        preds.append({
            "Matchup": g["Matchup"],
            "Away_ESPN": g["Away_ESPN"],
            "Home_ESPN": g["Home_ESPN"],
            "Proj_Away": a_pts,
            "Proj_Home": h_pts,
            "Proj_Total": a_pts + h_pts,
            "Proj_Margin_Home": h_pts - a_pts,
            "SD_Game": sd,
        })

    preds = pd.DataFrame(preds)

    # ============================
    # GET DRAFTKINGS LINES
    # ============================

    dk = fetch_dk(date_val)

    if not dk.empty:
        espn_teams = list(preds["Away_ESPN"]) + list(preds["Home_ESPN"])

        dk["Home_ESPN"], _, _ = zip(*dk["Home_Line_Team"].apply(lambda x: auto_map_team(x, espn_teams)))
        dk["Away_ESPN"], _, _ = zip(*dk["Away_Line_Team"].apply(lambda x: auto_map_team(x, espn_teams)))

        preds = preds.merge(dk, on=["Home_ESPN", "Away_ESPN"], how="left")

    # ============================
    # BET CALCULATIONS
    # ============================

    preds["P_Home_Cover"] = preds.apply(
        lambda r: p_cover(r["Proj_Margin_Home"], r["SD_Game"], r["Book_Spread_Home"])
        if pd.notna(r["Book_Spread_Home"]) else np.nan,
        axis=1
    )

    preds["P_Over"] = preds.apply(
        lambda r: p_over(r["Proj_Total"], r["SD_Game"] * 1.6, r["Book_Total"])
        if pd.notna(r["Book_Total"]) else np.nan,
        axis=1
    )

    preds["Edge_Spread"] = preds.apply(
        lambda r: r["P_Home_Cover"] - breakeven(r["Odds_H"])
        if pd.notna(r["P_Home_Cover"]) else np.nan,
        axis=1
    )

    preds["Edge_Total"] = preds.apply(
        lambda r: r["P_Over"] - breakeven(r["Odds_O"])
        if pd.notna(r["P_Over"]) else np.nan,
        axis=1
    )

    preds["Kelly_Spread"] = preds.apply(
        lambda r: kelly(r["P_Home_Cover"], r["Odds_H"])
        if pd.notna(r["P_Home_Cover"]) else np.nan,
        axis=1
    )

    preds["Kelly_Total"] = preds.apply(
        lambda r: kelly(r["P_Over"], r["Odds_O"])
        if pd.notna(r["P_Over"]) else np.nan,
        axis=1
    )

    st.subheader("All Games")
    st.dataframe(preds, use_container_width=True)

    # ============================
    # BEST BETS
    # ============================

    min_edge = st.slider("Min Edge (%)", 0.0, 10.0, 3.0) / 100

    best = preds[
        (preds["Edge_Spread"] >= min_edge) |
        (preds["Edge_Total"] >= min_edge)
    ].sort_values(["Edge_Spread", "Edge_Total"], ascending=False)

    st.subheader("🔥 Best Bets — DraftKings")
    st.dataframe(best, use_container_width=True)
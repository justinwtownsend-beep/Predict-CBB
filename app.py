# ================================
# CBB PREDICTOR + DRAFTKINGS BET FINDER (ALL-IN-ONE)
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

    # IMPORTANT: create a header-only file so pd.read_csv never throws EmptyDataError
    if not os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "w") as f:
            f.write("date,away,home,book_spread_home,book_total,proj_margin_home,proj_total,result_home_margin,result_total\n")


def load_state():
    ensure_storage()
    try:
        with open(STATE_PATH) as f:
            d = json.load(f)
    except Exception:
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
    """
    Returns (best_match, score, bonus)
    """
    # candidates can be a list-like of strings or list of dicts {"team": ...}
    if len(candidates) == 0:
        return None, 0.0, 0.0

    # If it's a pandas Series/Index of strings
    if not isinstance(candidates[0], dict):
        cand_list = list(candidates)
        best = max(cand_list, key=lambda t: similarity(name, t))
        score = similarity(name, best)
        return best, score, 0.0

    # If it's list of dicts with "team"
    best = max(candidates, key=lambda t: similarity(name, t["team"]))
    score = similarity(name, best["team"])
    return best["team"], score, 0.0


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
    r = requests.get(url, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()
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
    return float(np.nanmean([a_t, h_t]))


def sd_scaled(base, poss, avg=68.0):
    return float(base) * math.sqrt(float(poss) / float(avg))


def pp100(team_oe, opp_de, nat_avg):
    # Expected_OE = Team_AdjOE + Opp_AdjDE - National_Avg_OE
    return float(team_oe) + float(opp_de) - float(nat_avg)


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

    r = requests.get(url, params=params, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    rows = []
    for g in data:
        home = g.get("home_team")
        away = g.get("away_team")
        bks = g.get("bookmakers") or []
        if not home or not away or not bks:
            continue

        bk = bks[0]  # DK due to filter
        spread_home = None
        total = None
        odds_h = odds_a = odds_o = odds_u = None

        for m in bk.get("markets", []):
            if m.get("key") == "spreads":
                outs = m.get("outcomes", [])
                for o in outs:
                    if o.get("name") == home:
                        spread_home = o.get("point")
                        odds_h = o.get("price")
                    elif o.get("name") == away:
                        odds_a = o.get("price")

            if m.get("key") == "totals":
                outs = m.get("outcomes", [])
                for o in outs:
                    total = o.get("point")
                    nm = (o.get("name") or "").lower()
                    if nm == "over":
                        odds_o = o.get("price")
                    elif nm == "under":
                        odds_u = o.get("price")

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
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def p_cover_home(model_margin_home, sd, home_spread):
    """
    home_spread is the book's "home spread" (e.g., -4.5 means home must win by > 4.5)
    """
    m = float(model_margin_home)
    sd = max(float(sd), 1e-9)
    s = float(home_spread)

    cover_threshold = abs(s) if s < 0 else -abs(s)
    z = (cover_threshold - m) / sd
    return 1.0 - norm_cdf(z)


def p_over_total(model_total, sd_total, total_line):
    t = float(model_total)
    sd = max(float(sd_total), 1e-9)
    L = float(total_line)
    z = (L - t) / sd
    return 1.0 - norm_cdf(z)


def breakeven(odds):
    odds = float(odds)
    dec = 1.0 + (100.0 / abs(odds) if odds < 0 else odds / 100.0)
    return 1.0 / dec


def kelly(p, odds, frac=0.25):
    odds = float(odds)
    p = float(p)
    dec = 1.0 + (100.0 / abs(odds) if odds < 0 else odds / 100.0)
    b = dec - 1.0
    q = 1.0 - p
    f = (b * p - q) / max(b, 1e-9)
    return max(0.0, f) * float(frac)


# ================================
# APP
# ================================

st.set_page_config(layout="wide")
st.title("🏀 CBB Predictor + DraftKings Best Bets")

state = load_state()

st.sidebar.header("Settings")
year = st.sidebar.number_input("Season Year (e.g., 2026 = 2025-26)", 2020, 2030, 2026)
date_val = st.sidebar.date_input("Date", dt.date.today())

base_sd = st.sidebar.slider("Base Margin SD", 6.0, 18.0, float(state["BASE_SD"]), 0.5)
hca_pts = st.sidebar.slider("Home Court Advantage (pts)", 0.0, 6.0, float(state["HCA"]), 0.25)
sd_total_mult = st.sidebar.slider("Total SD multiplier", 1.1, 2.2, 1.6, 0.05)
kelly_frac = st.sidebar.slider("Kelly fraction", 0.05, 0.50, 0.25, 0.05)
min_edge = st.sidebar.slider("Min Edge (%)", 0.0, 10.0, 3.0, 0.5) / 100.0

# keep state updated in memory (and save)
state["BASE_SD"] = float(base_sd)
state["HCA"] = float(hca_pts)
save_state(state)

run = st.button("Run Model")


if run:
    teams = load_torvik(year)
    slate = get_espn(date_val)

    if slate.empty:
        st.warning("No games found from ESPN for this date.")
        st.stop()

    preds = []
    team_list = teams["TEAM"].tolist()

    for _, g in slate.iterrows():
        # ✅ FIX: mapper returns 3 values (team, score, bonus)
        away_team, _, _ = auto_map_team(g["Away_ESPN"], team_list)
        home_team, _, _ = auto_map_team(g["Home_ESPN"], team_list)

        if away_team is None or home_team is None:
            continue

        a = teams.loc[teams["TEAM"] == away_team].iloc[0]
        h = teams.loc[teams["TEAM"] == home_team].iloc[0]

        poss = possessions(a["TEMPO"], h["TEMPO"])
        sd = sd_scaled(state["BASE_SD"], poss, avg=state["LEAGUE_AVG_TEMPO"])

        a_pp100 = pp100(a["ADJ_OE"], h["ADJ_DE"], state["NATIONAL_AVG_OE"])
        h_pp100 = pp100(h["ADJ_OE"], a["ADJ_DE"], state["NATIONAL_AVG_OE"])

        a_pts = (a_pp100 / 100.0) * poss
        h_pts = (h_pp100 / 100.0) * poss

        if int(g["Neutral"]) == 0:
            h_pts += state["HCA"] / 2.0
            a_pts -= state["HCA"] / 2.0

        preds.append({
            "Matchup": g["Matchup"],
            "Away_ESPN": g["Away_ESPN"],
            "Home_ESPN": g["Home_ESPN"],
            "Neutral": int(g["Neutral"]),
            "Proj_Away": float(a_pts),
            "Proj_Home": float(h_pts),
            "Proj_Total": float(a_pts + h_pts),
            "Proj_Margin_Home": float(h_pts - a_pts),
            "SD_Game": float(sd),
        })

    preds = pd.DataFrame(preds)

    # DraftKings lines
    dk = fetch_dk(date_val)

    if dk.empty:
        st.warning("No DraftKings lines returned (Odds API). Check your key, date window, and quota.")
    else:
        espn_teams = list(pd.unique(pd.concat([preds["Away_ESPN"], preds["Home_ESPN"]], ignore_index=True)))

        dk["Home_ESPN_Map"], _, _ = zip(*dk["Home_Line_Team"].apply(lambda x: auto_map_team(x, espn_teams)))
        dk["Away_ESPN_Map"], _, _ = zip(*dk["Away_Line_Team"].apply(lambda x: auto_map_team(x, espn_teams)))

        preds = preds.merge(
            dk,
            left_on=["Home_ESPN", "Away_ESPN"],
            right_on=["Home_ESPN_Map", "Away_ESPN_Map"],
            how="left"
        )

    # Fill default odds if missing
    for col in ["Odds_H", "Odds_A", "Odds_O", "Odds_U"]:
        if col in preds.columns:
            preds[col] = preds[col].fillna(-110)

    # Bet probs
    preds["SD_Total"] = preds["SD_Game"] * float(sd_total_mult)

    preds["P_Home_Cover"] = preds.apply(
        lambda r: p_cover_home(r["Proj_Margin_Home"], r["SD_Game"], r["Book_Spread_Home"])
        if pd.notna(r.get("Book_Spread_Home")) else np.nan,
        axis=1
    )
    preds["P_Away_Cover"] = 1.0 - preds["P_Home_Cover"]

    preds["P_Over"] = preds.apply(
        lambda r: p_over_total(r["Proj_Total"], r["SD_Total"], r["Book_Total"])
        if pd.notna(r.get("Book_Total")) else np.nan,
        axis=1
    )
    preds["P_Under"] = 1.0 - preds["P_Over"]

    # Edges
    preds["BE_H"] = preds["Odds_H"].apply(breakeven)
    preds["BE_A"] = preds["Odds_A"].apply(breakeven)
    preds["BE_O"] = preds["Odds_O"].apply(breakeven)
    preds["BE_U"] = preds["Odds_U"].apply(breakeven)

    preds["Edge_Home_Spread"] = preds["P_Home_Cover"] - preds["BE_H"]
    preds["Edge_Away_Spread"] = preds["P_Away_Cover"] - preds["BE_A"]
    preds["Edge_Over"] = preds["P_Over"] - preds["BE_O"]
    preds["Edge_Under"] = preds["P_Under"] - preds["BE_U"]

    # Kelly sizes
    preds["Kelly_Home_Spread"] = preds.apply(
        lambda r: kelly(r["P_Home_Cover"], r["Odds_H"], frac=kelly_frac)
        if pd.notna(r["P_Home_Cover"]) else np.nan,
        axis=1
    )
    preds["Kelly_Away_Spread"] = preds.apply(
        lambda r: kelly(r["P_Away_Cover"], r["Odds_A"], frac=kelly_frac)
        if pd.notna(r["P_Away_Cover"]) else np.nan,
        axis=1
    )
    preds["Kelly_Over"] = preds.apply(
        lambda r: kelly(r["P_Over"], r["Odds_O"], frac=kelly_frac)
        if pd.notna(r["P_Over"]) else np.nan,
        axis=1
    )
    preds["Kelly_Under"] = preds.apply(
        lambda r: kelly(r["P_Under"], r["Odds_U"], frac=kelly_frac)
        if pd.notna(r["P_Under"]) else np.nan,
        axis=1
    )

    st.subheader("All Games")
    st.dataframe(preds, use_container_width=True)

    # Best bets table
    st.subheader("🔥 Best Bets — DraftKings")

    best = preds.copy()
    best["Best_Edge"] = best[["Edge_Home_Spread", "Edge_Away_Spread", "Edge_Over", "Edge_Under"]].max(axis=1)

    best = best[best["Best_Edge"] >= float(min_edge)].sort_values("Best_Edge", ascending=False)

    show_cols = [
        "Matchup",
        "Book_Spread_Home", "Book_Total",
        "P_Home_Cover", "P_Away_Cover", "P_Over", "P_Under",
        "Edge_Home_Spread", "Edge_Away_Spread", "Edge_Over", "Edge_Under",
        "Kelly_Home_Spread", "Kelly_Away_Spread", "Kelly_Over", "Kelly_Under",
    ]

    # keep only cols that exist
    show_cols = [c for c in show_cols if c in best.columns]

    st.dataframe(best[show_cols], use_container_width=True)
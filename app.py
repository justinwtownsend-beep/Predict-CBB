# ================================
# CBB TORVIK PREDICTOR (NO ODDS API)
# - ESPN slate (FULL D1 via groups=50) + Torvik ratings
# - Model spread/total + win%
# - Optional: enter DK lines directly in-table (no upload) to compute edge/Kelly
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
# CONFIG / STORAGE
# ================================

DATA_DIR = "data"
STATE_PATH = os.path.join(DATA_DIR, "model_state.json")
UA_HEADERS = {"User-Agent": "Mozilla/5.0"}

DEFAULTS = {
    "NATIONAL_AVG_OE": 107.0,   # approx D1 avg points/100 poss
    "LEAGUE_AVG_TEMPO": 68.0,   # avg tempo (poss/game)
    "HCA": 2.5,                # pts
    "BASE_SD": 11.0,           # baseline margin SD at avg tempo
    "TOTAL_SD_MULT": 1.6,      # SD_total ≈ SD_margin * 1.6
}


def ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(STATE_PATH):
        with open(STATE_PATH, "w") as f:
            json.dump(DEFAULTS, f)


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
# NAME NORMALIZATION + MAPPING
# ================================

def clean_name(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, clean_name(a), clean_name(b)).ratio()


def auto_map_team(name: str, candidates: list[str]) -> tuple[str | None, float, float]:
    """Returns (best_match, score, bonus)."""
    if not candidates:
        return None, 0.0, 0.0
    best = max(candidates, key=lambda t: similarity(name, t))
    score = similarity(name, best)
    return best, score, 0.0


# ================================
# TORVIK RATINGS
# ================================

@st.cache_data(ttl=3600)
def load_torvik(year: int) -> pd.DataFrame:
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

    out = df[["TEAM", "ADJ_OE", "ADJ_DE", "TEMPO"]].copy()
    out["TEAM"] = out["TEAM"].astype(str)
    return out


# ================================
# ESPN SLATE (FIXED: FULL D1)
# ================================

@st.cache_data(ttl=600)
def get_espn_slate_full_d1(date_val: dt.date) -> tuple[pd.DataFrame, dict]:
    """
    ESPN defaults to a small set unless you pass groups=50 for D1.
    This returns (df, debug).
    """
    d = date_val.strftime("%Y%m%d")
    base_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
    params = {
        "dates": d,
        "groups": 50,   # <-- FULL D1
        "limit": 300    # ESPN often caps at 100, but setting higher doesn't hurt
    }

    r = requests.get(base_url, params=params, headers=UA_HEADERS, timeout=30)
    status = r.status_code

    debug = {
        "url": r.url,
        "status": status,
        "text_len": len(r.text or ""),
        "events_count": None
    }

    if status != 200:
        return pd.DataFrame(), debug

    data = r.json()
    events = data.get("events", []) or []
    debug["events_count"] = len(events)

    rows = []
    for ev in events:
        try:
            comp = ev["competitions"][0]
            teams = comp["competitors"]

            home = next(t for t in teams if t.get("homeAway") == "home")
            away = next(t for t in teams if t.get("homeAway") == "away")

            rows.append({
                "Away_ESPN": away["team"]["shortDisplayName"],
                "Home_ESPN": home["team"]["shortDisplayName"],
                "Neutral": int(comp.get("neutralSite", False)),
                "Matchup": f'{away["team"]["shortDisplayName"]} at {home["team"]["shortDisplayName"]}',
            })
        except Exception:
            # skip any weird malformed event
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["Away_ESPN", "Home_ESPN", "Neutral"])

    return df, debug


# ================================
# MODEL MATH
# ================================

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def possessions(a_tempo: float, h_tempo: float) -> float:
    return float(np.nanmean([a_tempo, h_tempo]))


def sd_scaled(base_sd: float, poss: float, avg_poss: float) -> float:
    return float(base_sd) * math.sqrt(float(poss) / float(avg_poss))


def expected_pp100(team_oe: float, opp_de: float, nat_avg_oe: float) -> float:
    # Expected_OE = Team_AdjOE + Opp_AdjDE - National_Avg_OE
    return float(team_oe) + float(opp_de) - float(nat_avg_oe)


def breakeven_prob_from_american(odds: float) -> float:
    odds = float(odds)
    dec = 1.0 + (100.0 / abs(odds) if odds < 0 else odds / 100.0)
    return 1.0 / dec


def kelly_fraction(p: float, odds: float, frac: float = 0.25) -> float:
    odds = float(odds)
    p = float(p)
    dec = 1.0 + (100.0 / abs(odds) if odds < 0 else odds / 100.0)
    b = dec - 1.0
    q = 1.0 - p
    f = (b * p - q) / max(b, 1e-9)
    return max(0.0, f) * float(frac)


def p_home_covers(model_margin_home: float, sd_margin: float, home_spread: float) -> float:
    # home_spread: -4.5 means home must win by > 4.5
    m = float(model_margin_home)
    sd = max(float(sd_margin), 1e-9)
    s = float(home_spread)
    cover_threshold = abs(s) if s < 0 else -abs(s)
    z = (cover_threshold - m) / sd
    return 1.0 - normal_cdf(z)


def p_over(model_total: float, sd_total: float, total_line: float) -> float:
    t = float(model_total)
    sd = max(float(sd_total), 1e-9)
    L = float(total_line)
    z = (L - t) / sd
    return 1.0 - normal_cdf(z)


# ================================
# STREAMLIT UI
# ================================

st.set_page_config(layout="wide")
st.title("🏀 CBB Predictor (Torvik) — No Odds API")

state = load_state()

st.sidebar.header("Model settings")
season_year = st.sidebar.number_input("Season year (e.g., 2026 = 2025–26)", 2020, 2030, 2026)
date_val = st.sidebar.date_input("Slate date", dt.date.today())

state["HCA"] = st.sidebar.slider("Home-court advantage (pts)", 0.0, 6.0, float(state["HCA"]), 0.25)
state["BASE_SD"] = st.sidebar.slider("Base margin SD", 6.0, 18.0, float(state["BASE_SD"]), 0.5)
state["TOTAL_SD_MULT"] = st.sidebar.slider("Total SD multiplier", 1.1, 2.2, float(state["TOTAL_SD_MULT"]), 0.05)
save_state(state)

st.sidebar.header("Betting (optional)")
edge_min = st.sidebar.slider("Min edge (%)", 0.0, 10.0, 3.0, 0.5) / 100.0
kelly_frac = st.sidebar.slider("Kelly fraction", 0.05, 0.50, 0.25, 0.05)
default_odds = st.sidebar.selectbox("Default odds", [-110, -105, -115, -120], index=0)

show_debug = st.sidebar.checkbox("Show debug", value=True)

run = st.button("Run Model", type="primary")

if run:
    torvik = load_torvik(int(season_year))
    slate, debug = get_espn_slate_full_d1(date_val)

    if show_debug:
        with st.expander("Debug (ESPN request)"):
            st.json(debug)

    if slate.empty:
        st.warning("No games returned from ESPN for this date (or ESPN returned an empty slate).")
        st.stop()

    team_list = torvik["TEAM"].tolist()

    preds = []
    for _, g in slate.iterrows():
        away_team, away_score, _ = auto_map_team(g["Away_ESPN"], team_list)
        home_team, home_score, _ = auto_map_team(g["Home_ESPN"], team_list)

        if away_team is None or home_team is None:
            continue

        a = torvik.loc[torvik["TEAM"] == away_team].iloc[0]
        h = torvik.loc[torvik["TEAM"] == home_team].iloc[0]

        poss = possessions(a["TEMPO"], h["TEMPO"])
        sd_margin = sd_scaled(state["BASE_SD"], poss, state["LEAGUE_AVG_TEMPO"])
        sd_total = sd_margin * float(state["TOTAL_SD_MULT"])

        a_pp100 = expected_pp100(a["ADJ_OE"], h["ADJ_DE"], state["NATIONAL_AVG_OE"])
        h_pp100 = expected_pp100(h["ADJ_OE"], a["ADJ_DE"], state["NATIONAL_AVG_OE"])

        a_pts = (a_pp100 / 100.0) * poss
        h_pts = (h_pp100 / 100.0) * poss

        if int(g["Neutral"]) == 0:
            h_pts += state["HCA"] / 2.0
            a_pts -= state["HCA"] / 2.0

        proj_margin_home = h_pts - a_pts
        proj_total = h_pts + a_pts

        model_spread_home = -proj_margin_home
        model_total = proj_total
        home_win_pct = 1.0 - normal_cdf((0.0 - proj_margin_home) / max(sd_margin, 1e-9))

        preds.append({
            "Matchup": g["Matchup"],
            "Away_ESPN": g["Away_ESPN"],
            "Home_ESPN": g["Home_ESPN"],
            "Neutral": int(g["Neutral"]),
            "Proj_Away": float(a_pts),
            "Proj_Home": float(h_pts),
            "Proj_Total": float(proj_total),
            "Proj_Margin_Home": float(proj_margin_home),
            "Model_Spread_Home": float(model_spread_home),
            "Model_Total": float(model_total),
            "Home_Win_%": float(100.0 * home_win_pct),
            "SD_Margin": float(sd_margin),
            "SD_Total": float(sd_total),
            "MapScore_Away": float(away_score),
            "MapScore_Home": float(home_score),
            "Book_Spread_Home": np.nan,
            "Book_Total": np.nan,
            "Odds_Spread": float(default_odds),
            "Odds_Total": float(default_odds),
        })

    df = pd.DataFrame(preds)

    st.subheader(f"Model projections (games parsed: {len(df)})")
    st.dataframe(
        df[[
            "Matchup", "Proj_Home", "Proj_Away", "Proj_Total", "Proj_Margin_Home",
            "Model_Spread_Home", "Model_Total", "Home_Win_%", "SD_Margin",
            "MapScore_Home", "MapScore_Away"
        ]].sort_values("Matchup"),
        use_container_width=True
    )

    st.divider()
    st.subheader("Optional: type/paste lines (no uploads)")
    st.caption("Fill Book_Spread_Home and/or Book_Total. Odds default to your sidebar selection.")

    editable_cols = ["Book_Spread_Home", "Book_Total", "Odds_Spread", "Odds_Total"]
    edited = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Book_Spread_Home": st.column_config.NumberColumn("Book Spread (Home)", step=0.5),
            "Book_Total": st.column_config.NumberColumn("Book Total", step=0.5),
            "Odds_Spread": st.column_config.NumberColumn("Odds (Spread)", step=1),
            "Odds_Total": st.column_config.NumberColumn("Odds (Total)", step=1),
        },
        disabled=[c for c in df.columns if c not in editable_cols],
        key="lines_editor"
    )

    out = edited.copy()

    out["P_Home_Cover"] = out.apply(
        lambda r: p_home_covers(r["Proj_Margin_Home"], r["SD_Margin"], r["Book_Spread_Home"])
        if pd.notna(r["Book_Spread_Home"]) else np.nan,
        axis=1
    )
    out["P_Over"] = out.apply(
        lambda r: p_over(r["Proj_Total"], r["SD_Total"], r["Book_Total"])
        if pd.notna(r["Book_Total"]) else np.nan,
        axis=1
    )

    out["BE_Spread"] = out["Odds_Spread"].apply(breakeven_prob_from_american)
    out["BE_Total"] = out["Odds_Total"].apply(breakeven_prob_from_american)

    out["Edge_Spread_Home"] = out["P_Home_Cover"] - out["BE_Spread"]
    out["Edge_Total_Over"] = out["P_Over"] - out["BE_Total"]

    out["Kelly_Spread"] = out.apply(
        lambda r: kelly_fraction(r["P_Home_Cover"], r["Odds_Spread"], frac=kelly_frac)
        if pd.notna(r["P_Home_Cover"]) else np.nan,
        axis=1
    )
    out["Kelly_Total"] = out.apply(
        lambda r: kelly_fraction(r["P_Over"], r["Odds_Total"], frac=kelly_frac)
        if pd.notna(r["P_Over"]) else np.nan,
        axis=1
    )

    st.divider()
    st.subheader("Best Bets (only rows where you entered a line)")

    best = out.copy()
    best["Best_Edge"] = best[["Edge_Spread_Home", "Edge_Total_Over"]].max(axis=1)
    best = best[pd.notna(best["Best_Edge"])]
    best = best[best["Best_Edge"] >= edge_min].sort_values("Best_Edge", ascending=False)

    if best.empty:
        st.info("Enter at least one Book_Spread_Home or Book_Total above to generate bets.")
    else:
        st.dataframe(
            best[[
                "Matchup",
                "Book_Spread_Home", "Odds_Spread", "P_Home_Cover", "Edge_Spread_Home", "Kelly_Spread",
                "Book_Total", "Odds_Total", "P_Over", "Edge_Total_Over", "Kelly_Total",
                "Model_Spread_Home", "Model_Total"
            ]],
            use_container_width=True
        )
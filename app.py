import datetime as dt
import re
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st

UA_HEADERS = {"User-Agent": "Mozilla/5.0"}
DEFAULT_POSSESSIONS = 68.0


# ----------------------------
# Helpers
# ----------------------------
def clean(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def resolve(name: str, lookup: dict[str, str]) -> str:
    key = clean(name)
    if key in lookup:
        return lookup[key]
    # fallback contains-match
    for k, v in lookup.items():
        if key in k or k in key:
            return v
    raise ValueError(f"Could not resolve team name: {name}")


def pick_first(*vals):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None


# ----------------------------
# Torvik Ratings (safe loader)
# ----------------------------
@st.cache_data(ttl=3600)
def load_torvik_team_results(year: int) -> pd.DataFrame:
    """
    year=2026 corresponds to the 2025-26 season file: https://barttorvik.com/2026_team_results.csv
    """
    url = f"https://barttorvik.com/{year}_team_results.csv"
    r = requests.get(url, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()

    text = r.text
    if text.lstrip().startswith("<"):
        raise ValueError(f"Expected CSV but got HTML from {url}")

    df = pd.read_csv(StringIO(text), engine="python", on_bad_lines="skip")

    # Normalize TEAM column
    team_col = None
    for c in df.columns:
        if c.strip().lower() in {"team", "teams", "teamname"} or c.strip().upper() == "TEAM":
            team_col = c
            break
    if team_col is None:
        team_col = df.columns[0]

    df = df.rename(columns={team_col: "TEAM"})
    df["TEAM"] = df["TEAM"].astype(str).str.strip()
    return df


def find_col(df: pd.DataFrame, patterns: list[str]) -> str:
    cols = list(df.columns)
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for c in cols:
            if rx.search(str(c)):
                return c
    raise KeyError(f"Could not find column matching: {patterns}. Columns: {cols}")


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Torvik columns vary slightly by year; this is a robust finder
    df = df.rename(columns={
        find_col(df, ["AdjOE", r"\bOE\b"]): "ADJ_OE",
        find_col(df, ["AdjDE", r"\bDE\b"]): "ADJ_DE",
    })

    try:
        df = df.rename(columns={find_col(df, ["AdjT", "Tempo", "Pace", "Poss"]): "TEMPO"})
    except Exception:
        pass

    for c in ["ADJ_OE", "ADJ_DE", "TEMPO"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# ----------------------------
# Slate source (ESPN scoreboard)  ✅ FIXED DOMAIN
# ----------------------------
@st.cache_data(ttl=600)
def fetch_espn_scoreboard(date_yyyymmdd: str):
    """
    IMPORTANT:
    Use site.api.espn.com (works), not site.web.api.espn.com (often 404).
    groups=50 pulls the full D1 slate.
    """
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        f"?dates={date_yyyymmdd}&groups=50&limit=500"
    )
    r = requests.get(url, headers=UA_HEADERS, timeout=30)
    return url, r.status_code, r.text


@st.cache_data(ttl=600)
def get_espn_daily_slate(date_val: dt.date) -> tuple[pd.DataFrame, str, dict]:
    date_yyyymmdd = date_val.strftime("%Y%m%d")
    url, status, text = fetch_espn_scoreboard(date_yyyymmdd)

    debug = {"status": status, "url": url, "text_len": len(text)}

    if status != 200:
        return pd.DataFrame(), url, debug

    try:
        data = requests.models.complexjson.loads(text)
    except Exception as e:
        debug["json_error"] = str(e)
        return pd.DataFrame(), url, debug

    events = data.get("events", [])
    debug["events_count"] = len(events) if isinstance(events, list) else 0

    rows = []
    if isinstance(events, list):
        for ev in events:
            comps = ev.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]

            neutral = bool(comp.get("neutralSite", False))

            competitors = comp.get("competitors", [])
            if not isinstance(competitors, list) or len(competitors) < 2:
                continue

            away = None
            home = None
            for c in competitors:
                ha = (c.get("homeAway") or "").lower()
                team = c.get("team", {}) if isinstance(c.get("team"), dict) else {}
                name = pick_first(team.get("shortDisplayName"), team.get("displayName"), team.get("name"))
                if ha == "away":
                    away = name
                elif ha == "home":
                    home = name

            if not away or not home:
                continue

            matchup = f"{away} vs {home}" if neutral else f"{away} at {home}"
            rows.append({"Away": away, "Home": home, "Neutral": int(neutral), "Matchup": matchup})

    if rows:
        debug["first_row"] = rows[0]

    df = pd.DataFrame(rows).drop_duplicates()
    return df.reset_index(drop=True), url, debug


# ----------------------------
# Prediction engine
# ----------------------------
def predict_game(df_teams: pd.DataFrame, away: str, home: str, neutral: bool, hca: float, n: int, sd: float) -> dict:
    lookup = {clean(t): t for t in df_teams["TEAM"].dropna().unique()}

    away_team = resolve(away, lookup)
    home_team = resolve(home, lookup)

    a = df_teams[df_teams["TEAM"] == away_team].iloc[0]
    h = df_teams[df_teams["TEAM"] == home_team].iloc[0]

    poss = float(np.nanmean([a.get("TEMPO", np.nan), h.get("TEMPO", np.nan)]))
    if np.isnan(poss):
        poss = DEFAULT_POSSESSIONS

    away_pp100 = (float(a["ADJ_OE"]) + float(h["ADJ_DE"])) / 2.0
    home_pp100 = (float(h["ADJ_OE"]) + float(a["ADJ_DE"])) / 2.0

    away_pts = (away_pp100 / 100.0) * poss
    home_pts = (home_pp100 / 100.0) * poss

    if not neutral:
        home_pts += hca / 2.0
        away_pts -= hca / 2.0

    margin = home_pts - away_pts
    sims = np.random.default_rng(7).normal(loc=margin, scale=sd, size=n)

    return {
        "Away": away_team,
        "Home": home_team,
        "Neutral": bool(neutral),
        "Proj_Away": away_pts,
        "Proj_Home": home_pts,
        "Proj_Total": away_pts + home_pts,
        "Proj_Margin_Home": margin,
        "Home_Win_%": float((sims > 0).mean() * 100.0),
    }


def run_slate(season_year: int, date_val: dt.date, n: int, hca: float, sd: float):
    teams = standardize(load_torvik_team_results(season_year))
    slate, slate_url, debug = get_espn_daily_slate(date_val)

    if slate.empty:
        return pd.DataFrame(), slate_url, debug

    rows = []
    for _, row in slate.iterrows():
        away = str(row["Away"]).strip()
        home = str(row["Home"]).strip()
        neutral = bool(int(row.get("Neutral", 0)))

        try:
            pred = predict_game(teams, away, home, neutral, hca, n, sd)
            pred["Matchup"] = row.get("Matchup", f"{away} at {home}")
            rows.append(pred)
        except Exception as e:
            rows.append({"Matchup": row.get("Matchup", f"{away} at {home}"), "Error": str(e)})

    out = pd.DataFrame(rows)
    if not out.empty and "Proj_Margin_Home" in out.columns:
        out["Abs_Margin"] = out["Proj_Margin_Home"].abs()
        out["Close_Game_Score"] = (out["Home_Win_%"] - 50).abs()

    return out, slate_url, debug


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="CBB Slate Predictor", layout="wide")
st.title("CBB Slate Predictor (ESPN slate + Torvik ratings)")

with st.sidebar:
    st.header("Settings")
    season = st.number_input("Season year (2026 = 2025–26)", value=2026, step=1)
    date_val = st.date_input("Slate date", value=dt.date.today())
    sims = st.slider("Simulations", 1000, 20000, 10000, step=1000)
    hca = st.slider("Home-court advantage (pts)", 0.0, 5.0, 2.5, step=0.5)
    sd = st.slider("Margin SD", 6.0, 18.0, 11.0, step=0.5)
    auto_run = st.checkbox("Run automatically", value=True)
    show_debug = st.checkbox("Show debug", value=True)
    run_btn = st.button("Run slate")

if auto_run or run_btn:
    data, slate_url, debug = run_slate(int(season), date_val, int(sims), float(hca), float(sd))
    st.caption(f"Slate source: {slate_url}")

    if show_debug:
        with st.expander("Debug (what ESPN feed returned)"):
            st.json(debug)

    if data.empty:
        st.warning("No games parsed from the ESPN feed for this date.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Biggest projected margins")
        st.dataframe(data.sort_values("Abs_Margin", ascending=False).head(25), use_container_width=True)
    with c2:
        st.subheader("Closest games")
        st.dataframe(data.sort_values("Close_Game_Score", ascending=True).head(25), use_container_width=True)
    with c3:
        st.subheader("Highest projected totals")
        st.dataframe(data.sort_values("Proj_Total", ascending=False).head(25), use_container_width=True)

    st.subheader("All games")
    st.dataframe(data.sort_values("Matchup"), use_container_width=True)
else:
    st.info("Enable **Run automatically** or click **Run slate**.")

import datetime as dt
import re
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ----------------------------
# Torvik Ratings (safe loader)
# ----------------------------
@st.cache_data(ttl=3600)
def load_torvik_team_results(year: int) -> pd.DataFrame:
    url = f"https://barttorvik.com/{year}_team_results.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers, timeout=30)
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

    # Try to be robust to Torvik column naming differences
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
# Schedule (FIXED: requests -> parse HTML)
# ----------------------------
@st.cache_data(ttl=600)
def get_slate(date_yyyymmdd: str) -> tuple[pd.DataFrame, str]:
    url = f"https://barttorvik.com/schedule.php?date={date_yyyymmdd}&conlimit="

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    }

    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        html = r.text
    except Exception:
        return pd.DataFrame(), url

    # Parse tables from the HTML string (not directly from URL)
    try:
        tables = pd.read_html(StringIO(html), flavor="bs4")
    except ValueError:
        # "No tables found"
        return pd.DataFrame(), url
    except Exception:
        return pd.DataFrame(), url

    if not tables:
        return pd.DataFrame(), url

    # Pick the table containing a "Matchup" column (case-insensitive)
    slate = None
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if any("matchup" in c for c in cols):
            slate = t.copy()
            break

    if slate is None:
        return pd.DataFrame(), url

    slate.columns = [str(c).strip() for c in slate.columns]
    matchup_col = None
    for c in slate.columns:
        if "matchup" in str(c).lower():
            matchup_col = c
            break

    if matchup_col is None:
        return pd.DataFrame(), url

    slate = slate.rename(columns={matchup_col: "Matchup"})
    slate["Matchup"] = slate["Matchup"].astype(str)

    # Keep only real games
    slate = slate[slate["Matchup"].str.contains(r"\s(at|vs)\s", case=False, regex=True)]
    return slate.reset_index(drop=True), url


# ----------------------------
# Team name matching
# ----------------------------
def clean(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def resolve(name: str, lookup: dict[str, str]) -> str:
    key = clean(name)
    if key in lookup:
        return lookup[key]
    # fallback: contains match
    for k, v in lookup.items():
        if key in k or k in key:
            return v
    raise ValueError(f"Could not resolve team name: {name}")


# ----------------------------
# Prediction engine
# ----------------------------
def predict(df: pd.DataFrame, away: str, home: str, neutral: bool, hca: float, n: int, sd: float) -> dict:
    lookup = {clean(t): t for t in df["TEAM"].dropna().unique()}

    away_team = resolve(away, lookup)
    home_team = resolve(home, lookup)

    a = df[df["TEAM"] == away_team].iloc[0]
    h = df[df["TEAM"] == home_team].iloc[0]

    poss = float(np.nanmean([a.get("TEMPO", np.nan), h.get("TEMPO", np.nan)]))
    if np.isnan(poss):
        poss = 68.0

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
        "Neutral": neutral,
        "Proj_Away": away_pts,
        "Proj_Home": home_pts,
        "Proj_Total": away_pts + home_pts,
        "Proj_Margin_Home": margin,
        "Home_Win_%": float((sims > 0).mean() * 100.0),
    }


def run(year: int, date_yyyymmdd: str, n: int, hca: float, sd: float) -> tuple[pd.DataFrame, str]:
    df = standardize(load_torvik_team_results(year))
    slate, url = get_slate(date_yyyymmdd)

    if slate.empty:
        return pd.DataFrame(), url

    rows = []
    for m in slate["Matchup"]:
        m = str(m)
        if " vs " in m:
            a, h = m.split(" vs ", 1)
            neutral = True
        elif " at " in m:
            a, h = m.split(" at ", 1)
            neutral = False
        else:
            continue

        try:
            r = predict(df, a.strip(), h.strip(), neutral, hca, n, sd)
            r["Matchup"] = m
            rows.append(r)
        except Exception as e:
            rows.append({"Matchup": m, "Error": str(e)})

    out = pd.DataFrame(rows)
    if "Proj_Margin_Home" in out.columns:
        out["Abs_Margin"] = out["Proj_Margin_Home"].abs()
        out["Close_Game_Score"] = (out["Home_Win_%"] - 50).abs()

    return out, url


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="CBB Torvik Slate Predictor", layout="wide")
st.title("CBB Torvik Slate Predictor")

with st.sidebar:
    season = st.number_input("Season year (2026 = 2025–26)", value=2026, step=1)
    date_val = st.date_input("Slate date", value=dt.date.today())
    sims = st.slider("Simulations", 1000, 20000, 10000, step=1000)
    hca = st.slider("Home-court advantage (pts)", 0.0, 5.0, 2.5, step=0.5)
    sd = st.slider("Margin SD", 6.0, 18.0, 11.0, step=0.5)
    auto_run = st.checkbox("Run automatically", value=True)
    run_btn = st.button("Run slate")

date_yyyymmdd = date_val.strftime("%Y%m%d")
should_run = auto_run or run_btn

if should_run:
    data, url = run(int(season), date_yyyymmdd, int(sims), float(hca), float(sd))
    st.caption(f"Slate source: {url}")

    if data.empty:
        st.warning("No games found for this date (or Torvik returned no readable schedule table).")
        st.stop()

    if "Error" in data.columns and data.shape[1] <= 3:
        st.error("Most matchups failed to resolve.")
        st.dataframe(data, use_container_width=True)
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
    st.info("Use the sidebar and click **Run slate** (or enable **Run automatically**).")

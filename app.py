import datetime as dt
import re
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ----------------------------
# Helpers
# ----------------------------
UA_HEADERS = {"User-Agent": "Mozilla/5.0"}


def clean(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def resolve(name: str, lookup: dict[str, str]) -> str:
    key = clean(name)
    if key in lookup:
        return lookup[key]
    for k, v in lookup.items():
        if key in k or k in key:
            return v
    raise ValueError(f"Could not resolve team name: {name}")


# ----------------------------
# Torvik Ratings (safe loader)
# ----------------------------
@st.cache_data(ttl=3600)
def load_torvik_team_results(year: int) -> pd.DataFrame:
    url = f"https://barttorvik.com/{year}_team_results.csv"
    r = requests.get(url, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()

    text = r.text
    if text.lstrip().startswith("<"):
        raise ValueError(f"Expected CSV but got HTML from {url}")

    df = pd.read_csv(StringIO(text), engine="python", on_bad_lines="skip")

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
# Slate (use schedule.php&csv=1, no HTML tables)
# ----------------------------
@st.cache_data(ttl=600)
def get_slate(date_yyyymmdd: str) -> tuple[pd.DataFrame, str]:
    """
    Pull slate as CSV from schedule.php using &csv=1.
    This avoids pandas read_html (No tables found) and Torvik HTML changes.
    """
    url = f"https://barttorvik.com/schedule.php?date={date_yyyymmdd}&conlimit=&csv=1"

    try:
        r = requests.get(url, headers=UA_HEADERS, timeout=30)
        r.raise_for_status()
        text = r.text
    except Exception:
        return pd.DataFrame(), url

    # If Torvik returns an HTML "verifying your browser" page, it will start with "<"
    if text.lstrip().startswith("<"):
        return pd.DataFrame(), url

    # Try reading as CSV
    try:
        df = pd.read_csv(StringIO(text))
    except Exception:
        return pd.DataFrame(), url

    df.columns = [str(c).strip() for c in df.columns]

    # Need a matchup column (Torvik usually provides Matchup)
    matchup_col = None
    for c in df.columns:
        if "matchup" in str(c).lower():
            matchup_col = c
            break
    if matchup_col is None:
        return pd.DataFrame(), url

    df = df.rename(columns={matchup_col: "Matchup"})
    df["Matchup"] = df["Matchup"].astype(str)

    # Keep only real games
    df = df[df["Matchup"].str.contains(r"\s(at|vs)\s", case=False, regex=True)].reset_index(drop=True)
    return df, url


def parse_matchup(matchup: str):
    m = str(matchup).strip()
    if re.search(r"\s+vs\s+", m, flags=re.IGNORECASE):
        a, h = re.split(r"\s+vs\s+", m, flags=re.IGNORECASE)
        return a.strip(), h.strip(), True
    if re.search(r"\s+at\s+", m, flags=re.IGNORECASE):
        a, h = re.split(r"\s+at\s+", m, flags=re.IGNORECASE)
        return a.strip(), h.strip(), False
    return None, None, None


# ----------------------------
# Prediction
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
    teams = standardize(load_torvik_team_results(year))
    slate, url = get_slate(date_yyyymmdd)

    if slate.empty:
        return pd.DataFrame(), url

    rows = []
    for _, row in slate.iterrows():
        away, home, neutral = parse_matchup(row["Matchup"])
        if away is None:
            continue

        try:
            r = predict(teams, away, home, neutral, hca, n, sd)
            r["Matchup"] = row["Matchup"]
            rows.append(r)
        except Exception as e:
            rows.append({"Matchup": row["Matchup"], "Error": str(e)})

    out = pd.DataFrame(rows)
    if not out.empty and "Proj_Margin_Home" in out.columns:
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
        st.warning("No games found for this date (or Torvik blocked the CSV output).")
        st.caption("If you KNOW there are games, open the slate source link above. If it shows an HTML verification page, Torvik is blocking Streamlit Cloud.")
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

import datetime as dt
import re
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# Data pull + standardization
# ----------------------------
@st.cache_data(ttl=60 * 60)
def load_torvik_team_results(year: int) -> pd.DataFrame:
    url = f"https://barttorvik.com/{year}_team_results.csv"
    df = pd.read_csv(url)

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


def _find_col(df: pd.DataFrame, patterns: list[str]) -> str:
    cols = list(df.columns)
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for c in cols:
            if rx.fullmatch(c) or rx.search(c):
                return c
    raise KeyError(f"Missing column for {patterns}. Columns: {cols}")


def standardize_torvik_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    col_adj_oe = _find_col(out, [r"AdjOE", r"ADJ.*OE", r"OE.*Adj"])
    col_adj_de = _find_col(out, [r"AdjDE", r"ADJ.*DE", r"DE.*Adj"])

    try:
        col_tempo = _find_col(out, [r"AdjT", r"Tempo", r"Pace", r"Poss"])
    except KeyError:
        col_tempo = None

    rename_map = {col_adj_oe: "ADJ_OE", col_adj_de: "ADJ_DE"}
    if col_tempo:
        rename_map[col_tempo] = "TEMPO"

    out = out.rename(columns=rename_map)

    for c in ["ADJ_OE", "ADJ_DE", "TEMPO"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def build_team_lookup(df: pd.DataFrame) -> dict[str, str]:
    def simp(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9 ]+", "", s)
        s = re.sub(r"\s+", " ", s)
        return s

    lookup = {}
    for t in df["TEAM"].dropna().unique():
        lookup[simp(t)] = t
    return lookup


def resolve_team_name(user_input: str, lookup: dict[str, str]) -> str:
    def simp(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^a-z0-9 ]+", "", s)
        s = re.sub(r"\s+", " ", s)
        return s

    key = simp(user_input)
    if key in lookup:
        return lookup[key]

    candidates = [(k, v) for k, v in lookup.items() if key in k or k in key]
    if len(candidates) == 1:
        return candidates[0][1]
    if len(candidates) > 1:
        candidates.sort(key=lambda kv: len(kv[0]))
        return candidates[0][1]

    raise ValueError(f"Could not resolve team: '{user_input}'")


# ----------------------------
# Slate pull (UPDATED - picks correct table)
# ----------------------------
@st.cache_data(ttl=10 * 60)
def get_torvik_daily_slate(date_yyyymmdd: str) -> tuple[pd.DataFrame, str]:
    """
    Pull Torvik daily schedule for a given date.
    Fix: Torvik pages can have multiple tables. We select the one that contains a Matchup column.
    """
    url = f"https://barttorvik.com/schedule.php?conlimit=&date={date_yyyymmdd}&sort=time"

    try:
        tables = pd.read_html(url, flavor="bs4")
    except Exception:
        return pd.DataFrame(), url

    if not tables:
        return pd.DataFrame(), url

    # ✅ pick the table that actually contains a matchup column
    slate_df = None
    for t in tables:
        t = t.copy()
        t.columns = [str(c).strip() for c in t.columns]
        if any("matchup" in str(c).lower() for c in t.columns):
            slate_df = t
            break

    if slate_df is None:
        return pd.DataFrame(), url

    # Normalize matchup column
    matchup_col = [c for c in slate_df.columns if "matchup" in str(c).lower()][0]
    slate_df = slate_df.rename(columns={matchup_col: "Matchup"})

    # Keep only rows that look like real games
    slate_df["Matchup"] = slate_df["Matchup"].astype(str)
    slate_df = slate_df[slate_df["Matchup"].str.contains(r"\s(at|vs)\s", case=False, regex=True)].reset_index(drop=True)

    return slate_df, url


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
def predict_cbb_game(df_teams, away, home, neutral, hca_points, n_sims, margin_sd):
    lookup = build_team_lookup(df_teams)
    away_team = resolve_team_name(away, lookup)
    home_team = resolve_team_name(home, lookup)

    a = df_teams.loc[df_teams["TEAM"] == away_team].iloc[0]
    h = df_teams.loc[df_teams["TEAM"] == home_team].iloc[0]

    poss = float(np.nanmean([a.get("TEMPO", np.nan), h.get("TEMPO", np.nan)]))
    if np.isnan(poss):
        poss = 68.0

    away_pp100 = (float(a["ADJ_OE"]) + float(h["ADJ_DE"])) / 2.0
    home_pp100 = (float(h["ADJ_OE"]) + float(a["ADJ_DE"])) / 2.0

    away_pts = (away_pp100 / 100.0) * poss
    home_pts = (home_pp100 / 100.0) * poss

    if not neutral:
        home_pts += hca_points / 2.0
        away_pts -= hca_points / 2.0

    mean_margin = home_pts - away_pts

    rng = np.random.default_rng(7)
    sims_margin = rng.normal(loc=mean_margin, scale=margin_sd, size=n_sims)
    home_win_prob = float(np.mean(sims_margin > 0))

    return {
        "Away": away_team,
        "Home": home_team,
        "Neutral": neutral,
        "Proj_Away": away_pts,
        "Proj_Home": home_pts,
        "Proj_Total": home_pts + away_pts,
        "Proj_Margin_Home": mean_margin,
        "Home_Win_%": home_win_prob * 100,
    }


def run_slate(year, date_yyyymmdd, n_sims, hca_points, margin

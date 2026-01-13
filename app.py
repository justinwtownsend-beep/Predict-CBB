import datetime as dt
import re
from io import StringIO
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================
# Config
# ============================
UA_HEADERS = {"User-Agent": "Mozilla/5.0"}
DEFAULT_POSSESSIONS = 68.0


# ============================
# Text normalization + matching
# ============================
STOPWORDS = {
    "university", "college", "the", "of", "and", "at", "a", "an",
    "mens", "men", "womens", "women"
}

ABBREV_MAP = {
    "st": "state",
    "st.": "state",
    "ste": "stephen",
    "ft": "fort",
    "ft.": "fort",
    "mt": "mount",
    "mt.": "mount",
    "u": "university",
    "univ": "university",
    "tech": "tech",
    "a&m": "am",
    "aandm": "am",
    "&": "and",
}

DIRECTION_MAP = {
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
}

# Common ESPN shortenings that benefit from expansion
SPECIAL_PHRASES = [
    ("md", "maryland"),
    ("ar", "arkansas"),
    ("la", "louisiana"),
    ("tx", "texas"),
    ("nm", "new mexico"),
    ("nc", "north carolina"),
    ("sc", "south carolina"),
]


def _basic_clean(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace("\u2019", "'")
    s = re.sub(r"[\(\)\[\]\{\}]", " ", s)
    s = re.sub(r"[^a-z0-9&\.\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(name: str) -> list[str]:
    s = _basic_clean(name)

    # replace some punctuation that matters in team names
    s = s.replace("-", " ")
    s = s.replace(".", "")
    s = s.replace("&", " & ")

    parts = [p for p in s.split() if p]

    out = []
    for p in parts:
        if p in DIRECTION_MAP:
            out.append(DIRECTION_MAP[p])
            continue
        if p in ABBREV_MAP:
            out.append(ABBREV_MAP[p])
            continue
        out.append(p)

    # expand special phrases at token level (e.g., "md" -> "maryland")
    expanded = []
    for t in out:
        replaced = False
        for a, b in SPECIAL_PHRASES:
            if t == a:
                expanded.extend(b.split())
                replaced = True
                break
        if not replaced:
            expanded.append(t)

    # drop stopwords
    expanded = [t for t in expanded if t not in STOPWORDS]

    return expanded


def normalize_key(name: str) -> str:
    toks = tokenize(name)
    return "".join(toks)


def jaccard(a_tokens: set[str], b_tokens: set[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union else 0.0


def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def score_match(query_name: str, cand) -> float:
    """
    cand is a dict with:
      - key: normalized string
      - tokens: set[str]
    """
    q_key = normalize_key(query_name)
    q_tokens = set(tokenize(query_name))

    # blend character similarity + token overlap
    s1 = seq_ratio(q_key, cand["key"])
    s2 = jaccard(q_tokens, cand["tokens"])
    return 0.65 * s1 + 0.35 * s2


def generate_query_variants(name: str) -> list[str]:
    """
    Create a handful of variants so ESPN short names match Torvik more often.
    """
    s = _basic_clean(name)

    variants = {s}

    # common shortenings
    variants.add(s.replace(" st ", " state "))
    variants.add(s.replace(" st.", " state "))
    variants.add(s.replace(" & ", " and "))
    variants.add(s.replace(" a&m ", " am "))

    # directional expansions
    for k, v in DIRECTION_MAP.items():
        variants.add(re.sub(rf"\b{k}\b", v, s))

    # remove "the"
    variants.add(re.sub(r"\bthe\b", "", s).strip())

    # remove punctuation again
    variants = {re.sub(r"\s+", " ", v).strip() for v in variants if v.strip()}

    return list(variants)


@st.cache_data(ttl=3600)
def build_torvik_candidate_index(torvik_teams: list[str]):
    """
    Precompute candidate keys/tokens for fast matching.
    """
    cands = []
    for t in torvik_teams:
        cands.append({
            "team": t,
            "key": normalize_key(t),
            "tokens": set(tokenize(t)),
        })
    return cands


def auto_map_team(name: str, candidates, force_pick: bool = True):
    """
    Returns: (mapped_team, confidence_score, second_best_score)
    If force_pick=True, always returns the best candidate (even low confidence).
    """
    best_team = None
    best_score = -1.0
    second_score = -1.0

    for variant in generate_query_variants(name):
        for cand in candidates:
            s = score_match(variant, cand)
            if s > best_score:
                second_score = best_score
                best_score = s
                best_team = cand["team"]
            elif s > second_score:
                second_score = s

    if best_team is None:
        if force_pick and candidates:
            return candidates[0]["team"], 0.0, 0.0
        raise ValueError(f"Could not map team: {name}")

    return best_team, float(best_score), float(second_score)


# ============================
# Torvik loader (robust)
# ============================
def find_col(df: pd.DataFrame, patterns: list[str]) -> str:
    cols = list(df.columns)
    for pat in patterns:
        rx = re.compile(pat, re.IGNORECASE)
        for c in cols:
            if rx.search(str(c)):
                return c
    raise KeyError(f"Could not find column matching: {patterns}. Columns={cols}")


@st.cache_data(ttl=3600)
def load_torvik_team_results(year: int) -> pd.DataFrame:
    url = f"https://barttorvik.com/{year}_team_results.csv"
    r = requests.get(url, headers=UA_HEADERS, timeout=30)
    r.raise_for_status()

    text = r.text
    if text.lstrip().startswith("<"):
        raise ValueError(f"Expected CSV but got HTML from {url}")

    df = pd.read_csv(StringIO(text), engine="python", on_bad_lines="skip")

    # TEAM
    team_col = None
    for c in df.columns:
        if c.strip().lower() in {"team", "teams", "teamname"} or c.strip().upper() == "TEAM":
            team_col = c
            break
    if team_col is None:
        team_col = df.columns[0]
    df = df.rename(columns={team_col: "TEAM"})
    df["TEAM"] = df["TEAM"].astype(str).str.strip()

    # Ratings
    df = df.rename(columns={
        find_col(df, ["AdjOE", r"\bOE\b"]): "ADJ_OE",
        find_col(df, ["AdjDE", r"\bDE\b"]): "ADJ_DE",
    })

    try:
        df = df.rename(columns={find_col(df, ["AdjT", "Tempo", "Pace", "Poss"]): "TEMPO"})
    except Exception:
        df["TEMPO"] = np.nan

    for c in ["ADJ_OE", "ADJ_DE", "TEMPO"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # if tempo missing, fill later in predict
    return df[["TEAM", "ADJ_OE", "ADJ_DE", "TEMPO"]].copy()


# ============================
# ESPN slate (correct endpoint)
# ============================
@st.cache_data(ttl=600)
def fetch_espn_scoreboard(date_yyyymmdd: str):
    # ✅ Use site.api.espn.com (works) + group 50 for D1
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        f"?dates={date_yyyymmdd}&groups=50&limit=500"
    )
    r = requests.get(url, headers=UA_HEADERS, timeout=30)
    return url, r.status_code, r.text


@st.cache_data(ttl=600)
def get_espn_daily_slate(date_val: dt.date):
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
                name = team.get("shortDisplayName") or team.get("displayName") or team.get("name")
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


# ============================
# Prediction engine
# ============================
def predict_game(df_teams: pd.DataFrame, candidates, away: str, home: str, neutral: bool,
                 hca: float, n: int, sd: float):
    away_team, away_conf, away_second = auto_map_team(away, candidates, force_pick=True)
    home_team, home_conf, home_second = auto_map_team(home, candidates, force_pick=True)

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
        "Away_ESPN": away,
        "Home_ESPN": home,
        "Away_Torvik": away_team,
        "Home_Torvik": home_team,
        "Map_Conf_Away": away_conf,
        "Map_Conf_Home": home_conf,
        "Neutral": bool(neutral),
        "Proj_Away": away_pts,
        "Proj_Home": home_pts,
        "Proj_Total": away_pts + home_pts,
        "Proj_Margin_Home": margin,
        "Home_Win_%": float((sims > 0).mean() * 100.0),
    }


def run_slate(season_year: int, date_val: dt.date, n: int, hca: float, sd: float):
    teams = load_torvik_team_results(season_year)
    torvik_team_list = teams["TEAM"].dropna().unique().tolist()
    candidates = build_torvik_candidate_index(torvik_team_list)

    slate, slate_url, debug = get_espn_daily_slate(date_val)
    if slate.empty:
        return pd.DataFrame(), slate_url, debug

    rows = []
    for _, row in slate.iterrows():
        try:
            pred = predict_game(
                teams,
                candidates,
                away=str(row["Away"]).strip(),
                home=str(row["Home"]).strip(),
                neutral=bool(int(row.get("Neutral", 0))),
                hca=float(hca),
                n=int(n),
                sd=float(sd),
            )
            pred["Matchup"] = row.get("Matchup")
            rows.append(pred)
        except Exception as e:
            rows.append({
                "Matchup": row.get("Matchup"),
                "Error": str(e),
                "Away_ESPN": row.get("Away"),
                "Home_ESPN": row.get("Home"),
            })

    out = pd.DataFrame(rows)

    if "Proj_Margin_Home" in out.columns:
        out["Abs_Margin"] = out["Proj_Margin_Home"].abs()
        out["Close_Game_Score"] = (out["Home_Win_%"] - 50).abs()

    # helpful flags: low-confidence mappings
    if "Map_Conf_Away" in out.columns and "Map_Conf_Home" in out.columns:
        out["Low_Conf_Map"] = (out["Map_Conf_Away"] < 0.78) | (out["Map_Conf_Home"] < 0.78)

    return out, slate_url, debug


# ============================
# UI
# ============================
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
    show_debug = st.checkbox("Show debug", value=False)
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

    # Split out errors vs good rows
    has_error = data.columns.to_list()
    err_rows = data[data.get("Error").notna()] if "Error" in data.columns else pd.DataFrame()
    ok_rows = data[data.get("Error").isna()] if "Error" in data.columns else data

    if not err_rows.empty:
        st.warning(f"{len(err_rows)} games had errors (still shown below).")

    # Top panels for ok rows only
    if not ok_rows.empty and "Abs_Margin" in ok_rows.columns:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Biggest projected margins")
            st.dataframe(ok_rows.sort_values("Abs_Margin", ascending=False).head(25), use_container_width=True)
        with c2:
            st.subheader("Closest games")
            st.dataframe(ok_rows.sort_values("Close_Game_Score", ascending=True).head(25), use_container_width=True)
        with c3:
            st.subheader("Highest projected totals")
            st.dataframe(ok_rows.sort_values("Proj_Total", ascending=False).head(25), use_container_width=True)

    st.subheader("All games")
    # Sort with low confidence at top so you can eyeball it (no maintenance required)
    if "Low_Conf_Map" in data.columns:
        st.dataframe(
            data.sort_values(["Low_Conf_Map", "Matchup"], ascending=[False, True]),
            use_container_width=True
        )
    else:
        st.dataframe(data.sort_values("Matchup"), use_container_width=True)

else:
    st.info("Enable **Run automatically** or click **Run slate**.")

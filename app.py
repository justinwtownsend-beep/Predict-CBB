import datetime as dt
import json
import os
import re
from io import StringIO
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================
# Storage
# ============================
DATA_DIR = "data"
HISTORY_PATH = os.path.join(DATA_DIR, "history.csv")
STATE_PATH = os.path.join(DATA_DIR, "model_state.json")


# ============================
# Defaults (will be learned)
# ============================
DEFAULTS = {
    "NATIONAL_AVG_OE": 107.0,   # points per 100 possessions
    "LEAGUE_AVG_TEMPO": 68.0,   # possessions
    "HCA": 2.5,                # points
    "BASE_SD": 11.0,           # margin SD at league avg tempo
    "K_CAL": 1.0,              # win% calibration scale on z = margin/sd
    "MIN_MAP_CONF": 0.78,      # for flagging low-confidence team matches
}

UA_HEADERS = {"User-Agent": "Mozilla/5.0"}
DEFAULT_POSSESSIONS = 68.0


# ============================
# Team auto-mapping (no manual aliases)
# ============================
STOPWORDS = {"university", "college", "the", "of", "and", "at", "a", "an", "mens", "men", "womens", "women"}

ABBREV_MAP = {
    "st": "state", "st.": "state",
    "ste": "stephen",
    "ft": "fort", "ft.": "fort",
    "mt": "mount", "mt.": "mount",
    "u": "university", "univ": "university",
    "a&m": "am", "aandm": "am",
    "&": "and",
}

DIRECTION_MAP = {
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
}

SPECIAL_PHRASES = [
    ("md", "maryland"), ("ar", "arkansas"), ("la", "louisiana"),
    ("tx", "texas"), ("nm", "new mexico"), ("nc", "north carolina"),
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
    s = s.replace("-", " ").replace(".", "")
    s = s.replace("&", " & ")
    parts = [p for p in s.split() if p]

    out = []
    for p in parts:
        if p in DIRECTION_MAP:
            out.append(DIRECTION_MAP[p])
        elif p in ABBREV_MAP:
            out.append(ABBREV_MAP[p])
        else:
            out.append(p)

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

    expanded = [t for t in expanded if t not in STOPWORDS]
    return expanded


def normalize_key(name: str) -> str:
    return "".join(tokenize(name))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def seq_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def score_match(query_name: str, cand: dict) -> float:
    q_key = normalize_key(query_name)
    q_tokens = set(tokenize(query_name))
    s1 = seq_ratio(q_key, cand["key"])
    s2 = jaccard(q_tokens, cand["tokens"])
    return 0.65 * s1 + 0.35 * s2


def generate_query_variants(name: str) -> list[str]:
    s = _basic_clean(name)
    variants = {s}
    variants.add(s.replace(" st ", " state "))
    variants.add(s.replace(" st.", " state "))
    variants.add(s.replace(" & ", " and "))
    variants.add(s.replace(" a&m ", " am "))

    for k, v in DIRECTION_MAP.items():
        variants.add(re.sub(rf"\b{k}\b", v, s))

    variants.add(re.sub(r"\bthe\b", "", s).strip())
    variants = {re.sub(r"\s+", " ", v).strip() for v in variants if v.strip()}
    return list(variants)


@st.cache_data(ttl=3600)
def build_torvik_candidate_index(torvik_teams: list[str]):
    return [{"team": t, "key": normalize_key(t), "tokens": set(tokenize(t))} for t in torvik_teams]


def auto_map_team(name: str, candidates, force_pick: bool = True):
    best_team, best_score, second_score = None, -1.0, -1.0

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
# Torvik loader
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

    # TEAM column
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

    return df[["TEAM", "ADJ_OE", "ADJ_DE", "TEMPO"]].copy()


# ============================
# ESPN slate + results
# ============================
@st.cache_data(ttl=600)
def fetch_espn_scoreboard(date_yyyymmdd: str):
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        f"?dates={date_yyyymmdd}&groups=50&limit=500"
    )
    r = requests.get(url, headers=UA_HEADERS, timeout=30)
    return url, r.status_code, r.text


def parse_espn_events(text: str):
    data = requests.models.complexjson.loads(text)
    return data.get("events", [])


def get_games_from_events(events):
    """
    Returns rows with stable ESPN identifiers:
      - event_id
      - competition_id
    plus team names and neutral flag
    """
    rows = []
    if not isinstance(events, list):
        return rows

    for ev in events:
        event_id = ev.get("id")
        comps = ev.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        comp_id = comp.get("id")
        neutral = bool(comp.get("neutralSite", False))

        competitors = comp.get("competitors", [])
        if not isinstance(competitors, list) or len(competitors) < 2:
            continue

        away = home = None
        away_score = home_score = None

        for c in competitors:
            ha = (c.get("homeAway") or "").lower()
            team = c.get("team", {}) if isinstance(c.get("team"), dict) else {}
            name = team.get("shortDisplayName") or team.get("displayName") or team.get("name")
            score = c.get("score", None)

            if ha == "away":
                away = name
                away_score = score
            elif ha == "home":
                home = name
                home_score = score

        if not away or not home:
            continue

        status = comp.get("status", {}).get("type", {})
        state = status.get("state")          # e.g., "pre", "in", "post"
        completed = bool(status.get("completed", False))

        matchup = f"{away} vs {home}" if neutral else f"{away} at {home}"

        rows.append({
            "event_id": event_id,
            "competition_id": comp_id,
            "Away_ESPN": away,
            "Home_ESPN": home,
            "Neutral": int(neutral),
            "Matchup": matchup,
            "state": state,
            "completed": int(completed),
            "Away_score": away_score,
            "Home_score": home_score,
        })

    return rows


@st.cache_data(ttl=600)
def get_espn_daily(date_val: dt.date):
    date_yyyymmdd = date_val.strftime("%Y%m%d")
    url, status, text = fetch_espn_scoreboard(date_yyyymmdd)
    debug = {"status": status, "url": url, "text_len": len(text)}
    if status != 200:
        return pd.DataFrame(), url, debug

    try:
        events = parse_espn_events(text)
        rows = get_games_from_events(events)
        debug["events_count"] = len(events) if isinstance(events, list) else 0
        debug["games_count"] = len(rows)
        if rows:
            debug["first_row"] = rows[0]
        return pd.DataFrame(rows), url, debug
    except Exception as e:
        debug["parse_error"] = str(e)
        return pd.DataFrame(), url, debug


# ============================
# Model state + history
# ============================
def ensure_storage():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(STATE_PATH):
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(DEFAULTS, f, indent=2)
    if not os.path.exists(HISTORY_PATH):
        pd.DataFrame().to_csv(HISTORY_PATH, index=False)


def load_state():
    ensure_storage()
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        d = json.load(f)
    # fill any missing defaults
    for k, v in DEFAULTS.items():
        d.setdefault(k, v)
    return d


def save_state(d: dict):
    ensure_storage()
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def load_history() -> pd.DataFrame:
    ensure_storage()
    if os.path.exists(HISTORY_PATH) and os.path.getsize(HISTORY_PATH) > 0:
        return pd.read_csv(HISTORY_PATH)
    return pd.DataFrame()


def append_history(df_new: pd.DataFrame):
    ensure_storage()
    hist = load_history()
    if hist.empty:
        df_new.to_csv(HISTORY_PATH, index=False)
        return
    combined = pd.concat([hist, df_new], ignore_index=True)
    # de-dupe by competition_id if present
    if "competition_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["competition_id"], keep="last")
    combined.to_csv(HISTORY_PATH, index=False)


# ============================
# Pro formulas (learned params)
# ============================
def game_possessions(a_tempo: float, h_tempo: float) -> float:
    poss = float(np.nanmean([a_tempo, h_tempo]))
    if np.isnan(poss) or poss <= 0:
        poss = DEFAULT_POSSESSIONS
    return poss


def tempo_scaled_sd(base_sd: float, poss: float, league_avg_tempo: float) -> float:
    return float(base_sd * np.sqrt(max(poss, 1.0) / float(league_avg_tempo)))


def expected_pp100(team_oe: float, opp_de: float, nat_avg_oe: float) -> float:
    return float(team_oe + opp_de - nat_avg_oe)


def logistic(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# ============================
# Predict + Save
# ============================
def predict_slate(season_year: int, date_val: dt.date, n_sims: int, state: dict, show_debug=False):
    teams = load_torvik_team_results(season_year)
    torvik_team_list = teams["TEAM"].dropna().unique().tolist()
    candidates = build_torvik_candidate_index(torvik_team_list)

    slate, slate_url, debug = get_espn_daily(date_val)
    if slate.empty:
        return pd.DataFrame(), slate_url, debug

    nat_avg = float(state["NATIONAL_AVG_OE"])
    hca = float(state["HCA"])
    base_sd = float(state["BASE_SD"])
    league_avg_tempo = float(state["LEAGUE_AVG_TEMPO"])
    k_cal = float(state["K_CAL"])

    rows = []
    for _, g in slate.iterrows():
        away_name = str(g["Away_ESPN"]).strip()
        home_name = str(g["Home_ESPN"]).strip()
        neutral = bool(int(g.get("Neutral", 0)))

        try:
            away_team, away_conf, _ = auto_map_team(away_name, candidates, force_pick=True)
            home_team, home_conf, _ = auto_map_team(home_name, candidates, force_pick=True)

            a = teams[teams["TEAM"] == away_team].iloc[0]
            h = teams[teams["TEAM"] == home_team].iloc[0]

            poss = game_possessions(a.get("TEMPO", np.nan), h.get("TEMPO", np.nan))
            sd_game = tempo_scaled_sd(base_sd, poss, league_avg_tempo)

            away_pp100 = expected_pp100(a["ADJ_OE"], h["ADJ_DE"], nat_avg)
            home_pp100 = expected_pp100(h["ADJ_OE"], a["ADJ_DE"], nat_avg)

            away_pts = (away_pp100 / 100.0) * poss
            home_pts = (home_pp100 / 100.0) * poss

            if not neutral:
                home_pts += hca / 2.0
                away_pts -= hca / 2.0

            margin = home_pts - away_pts
            total = home_pts + away_pts

            # MC for distribution-y stats if you want them later
            sims = np.random.default_rng(7).normal(loc=margin, scale=sd_game, size=int(n_sims))
            home_win_mc = float((sims > 0).mean() * 100.0)

            # Calibrated analytic win prob (fast + stable)
            z = (margin / sd_game) if sd_game > 0 else 0.0
            p_home = float(logistic(np.array([k_cal * z]))[0] * 100.0)

            low_conf = (away_conf < state["MIN_MAP_CONF"]) or (home_conf < state["MIN_MAP_CONF"])

            rows.append({
                "date": date_val.strftime("%Y-%m-%d"),
                "season_year": int(season_year),

                "event_id": g.get("event_id"),
                "competition_id": g.get("competition_id"),

                "Matchup": g.get("Matchup"),
                "Neutral": int(neutral),

                "Away_ESPN": away_name,
                "Home_ESPN": home_name,
                "Away_Torvik": away_team,
                "Home_Torvik": home_team,
                "Map_Conf_Away": away_conf,
                "Map_Conf_Home": home_conf,
                "Low_Conf_Map": int(low_conf),

                "Possessions": poss,
                "SD_Game": sd_game,
                "Proj_Away": away_pts,
                "Proj_Home": home_pts,
                "Proj_Total": total,
                "Proj_Margin_Home": margin,
                "Home_Win_%_MC": home_win_mc,
                "Home_Win_%": p_home,   # calibrated analytic

                # placeholders for results
                "Final_Away": np.nan,
                "Final_Home": np.nan,
                "Final_Total": np.nan,
                "Final_Margin_Home": np.nan,
                "Completed": 0,
            })
        except Exception as e:
            rows.append({
                "date": date_val.strftime("%Y-%m-%d"),
                "season_year": int(season_year),
                "event_id": g.get("event_id"),
                "competition_id": g.get("competition_id"),
                "Matchup": g.get("Matchup"),
                "Error": str(e),
            })

    out = pd.DataFrame(rows)
    return out, slate_url, debug


# ============================
# Update results (self-learning loop)
# ============================
def update_results_from_espn_for_date(date_val: dt.date):
    """
    Fetch ESPN scoreboard for date, return a DF keyed by competition_id with final scores when completed.
    """
    slate, _, debug = get_espn_daily(date_val)
    if slate.empty:
        return pd.DataFrame(), debug

    # Only completed games with scores
    done = slate[(slate["completed"] == 1)].copy()
    if done.empty:
        return pd.DataFrame(), debug

    # cast scores to numeric
    done["Away_score"] = pd.to_numeric(done["Away_score"], errors="coerce")
    done["Home_score"] = pd.to_numeric(done["Home_score"], errors="coerce")
    done = done.dropna(subset=["Away_score", "Home_score"])

    done["Final_Away"] = done["Away_score"]
    done["Final_Home"] = done["Home_score"]
    done["Final_Total"] = done["Final_Away"] + done["Final_Home"]
    done["Final_Margin_Home"] = done["Final_Home"] - done["Final_Away"]
    done["Completed"] = 1

    cols = ["competition_id", "Final_Away", "Final_Home", "Final_Total", "Final_Margin_Home", "Completed"]
    return done[cols].copy(), debug


def update_history_with_results(history: pd.DataFrame) -> pd.DataFrame:
    """
    For any rows missing Completed=1, fetch results for their date and fill.
    """
    if history.empty or "date" not in history.columns:
        return history

    history = history.copy()
    if "Completed" not in history.columns:
        history["Completed"] = 0

    # dates that still need updates
    need = history[(history["Completed"].fillna(0).astype(int) == 0) & history["competition_id"].notna()]
    if need.empty:
        return history

    dates = sorted(set(pd.to_datetime(need["date"]).dt.date.tolist()))
    for d in dates:
        finals, _dbg = update_results_from_espn_for_date(d)
        if finals.empty:
            continue
        history = history.merge(finals, on="competition_id", how="left", suffixes=("", "_new"))

        # fill only where new exists
        for col in ["Final_Away", "Final_Home", "Final_Total", "Final_Margin_Home", "Completed"]:
            newc = f"{col}_new"
            if newc in history.columns:
                history[col] = history[col].where(history[newc].isna(), history[newc])
                history.drop(columns=[newc], inplace=True)

    return history


# ============================
# Learning (parameter tuning)
# ============================
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return np.inf
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def learn_parameters(history: pd.DataFrame, state: dict, max_rows: int = 800):
    """
    Grid-search a few parameters to minimize RMSE on margin and total, plus calibrate win%.
    Uses only completed games.
    """
    hist = history.copy()
    if hist.empty:
        return state, {"msg": "No history to learn from."}

    if "Completed" not in hist.columns:
        return state, {"msg": "History missing Completed column."}

    done = hist[hist["Completed"].fillna(0).astype(int) == 1].copy()
    # require predictions + finals
    need_cols = ["Proj_Margin_Home", "Final_Margin_Home", "Proj_Total", "Final_Total", "Possessions"]
    for c in need_cols:
        if c not in done.columns:
            return state, {"msg": f"History missing column: {c}"}

    done = done.dropna(subset=["Proj_Margin_Home", "Final_Margin_Home", "Proj_Total", "Final_Total", "Possessions"])
    if done.empty:
        return state, {"msg": "No completed rows with both predictions and finals."}

    done = done.tail(int(max_rows))

    # We tune by correcting systematic bias:
    # - nat_avg_oe shifts both team scoring expectations
    # - hca shifts margins in non-neutral games
    # - base_sd affects calibration + MC spread; here we fit sd to margin residuals scaled by tempo
    #
    # We'll optimize nat_avg + hca by minimizing RMSE on margin and total.
    # Then base_sd by matching residual SD vs tempo.

    # Precompute some pieces to allow fast evaluation:
    # Our model already produced Proj_* values using current state.
    # For tuning nat_avg/hca, we'd ideally recompute projections.
    # But we can approximate adjustments:
    # - changing nat_avg_oe shifts both away_pp100/home_pp100 by (-delta_nat_avg),
    #   so total shifts by (-2 * delta_nat_avg / 100 * poss) and margin does NOT change from nat_avg alone.
    # Actually, nat_avg affects both teams equally in our formula, so margin stays mostly unchanged.
    # It *does* affect total strongly. Good.
    #
    # For HCA: margin shifts by delta_hca for non-neutral games (because home +hca/2, away -hca/2 => +hca margin).
    #
    # So:
    #   new_total = old_total + (-2*delta_nat_avg/100)*poss
    #   new_margin = old_margin + delta_hca*(1-neutral)
    #
    # This makes tuning very stable without refetching Torvik.

    poss = done["Possessions"].astype(float).to_numpy()
    neutral = done.get("Neutral", 0)
    if "Neutral" in done.columns:
        neutral = done["Neutral"].fillna(0).astype(int).to_numpy()
    else:
        neutral = np.zeros(len(done), dtype=int)

    old_total = done["Proj_Total"].astype(float).to_numpy()
    old_margin = done["Proj_Margin_Home"].astype(float).to_numpy()
    y_total = done["Final_Total"].astype(float).to_numpy()
    y_margin = done["Final_Margin_Home"].astype(float).to_numpy()

    nat0 = float(state["NATIONAL_AVG_OE"])
    hca0 = float(state["HCA"])

    nat_grid = np.arange(104.0, 110.5, 0.5)     # safe range
    hca_grid = np.arange(0.0, 5.25, 0.25)

    best = None
    for nat in nat_grid:
        delta_nat = nat - nat0
        new_total = old_total + (-2.0 * delta_nat / 100.0) * poss

        for hca in hca_grid:
            delta_hca = hca - hca0
            new_margin = old_margin + delta_hca * (1 - neutral)

            rmse_m = rmse(new_margin, y_margin)
            rmse_t = rmse(new_total, y_total)

            score = rmse_m + 0.35 * rmse_t  # emphasize margin, still care about totals
            if (best is None) or (score < best["score"]):
                best = {
                    "nat": float(nat),
                    "hca": float(hca),
                    "rmse_margin": rmse_m,
                    "rmse_total": rmse_t,
                    "score": score,
                }

    # Fit BASE_SD from residuals with tempo scaling:
    # We expect residual SD ~ base_sd * sqrt(poss/avg_tempo)
    # => base_sd ~ std(residual / sqrt(poss/avg_tempo))
    league_avg_tempo = float(state["LEAGUE_AVG_TEMPO"])
    scale = np.sqrt(np.maximum(poss, 1.0) / league_avg_tempo)
    resid = (y_margin - (old_margin + (best["hca"] - hca0) * (1 - neutral))).astype(float)
    resid_scaled = resid / scale
    base_sd_hat = float(np.nanstd(resid_scaled, ddof=1))
    base_sd_hat = float(np.clip(base_sd_hat, 6.0, 18.0))

    # Calibrate win probability scale K on z = margin/sd_game:
    # y = 1 if home wins else 0
    y_win = (y_margin > 0).astype(float)

    # use current per-game SD from history if available; otherwise recompute from poss + base_sd_hat
    if "SD_Game" in done.columns and done["SD_Game"].notna().any():
        sd_game = done["SD_Game"].astype(float).to_numpy()
        # adjust sd_game to reflect base_sd_hat vs old base_sd
        old_base_sd = float(state["BASE_SD"])
        sd_game = sd_game * (base_sd_hat / old_base_sd) if old_base_sd > 0 else sd_game
    else:
        sd_game = base_sd_hat * scale

    z = (old_margin + (best["hca"] - hca0) * (1 - neutral)) / np.maximum(sd_game, 1e-6)

    k_grid = np.arange(0.6, 2.41, 0.05)
    best_k = None
    for k in k_grid:
        p = logistic(k * z)
        ll = logloss(y_win, p)
        if (best_k is None) or (ll < best_k["ll"]):
            best_k = {"k": float(k), "ll": float(ll)}

    new_state = dict(state)
    new_state["NATIONAL_AVG_OE"] = best["nat"]
    new_state["HCA"] = best["hca"]
    new_state["BASE_SD"] = base_sd_hat
    new_state["K_CAL"] = best_k["k"]

    metrics = {
        "n_games_used": int(len(done)),
        "best_nat_avg_oe": best["nat"],
        "best_hca": best["hca"],
        "fit_base_sd": base_sd_hat,
        "fit_k_cal": best_k["k"],
        "rmse_margin": best["rmse_margin"],
        "rmse_total": best["rmse_total"],
        "logloss_win": best_k["ll"],
    }
    return new_state, metrics


# ============================
# UI
# ============================
st.set_page_config(page_title="CBB Slate Predictor (Self-Learning)", layout="wide")
st.title("CBB Slate Predictor — Self-Learning (ESPN slate + Torvik ratings)")

state = load_state()
history = load_history()

with st.sidebar:
    st.header("Model controls")

    season = st.number_input("Season year (2026 = 2025–26)", value=2026, step=1)
    date_val = st.date_input("Slate date", value=dt.date.today())

    st.divider()
    n_sims = st.slider("MC simulations (for extras)", 1000, 30000, 10000, step=1000)

    st.divider()
    st.subheader("Current learned parameters")
    st.write({
        "National Avg OE": state["NATIONAL_AVG_OE"],
        "HCA": state["HCA"],
        "Base SD": state["BASE_SD"],
        "K calibration": state["K_CAL"],
    })

    st.divider()
    auto_update = st.checkbox("Auto-update results & learn on load", value=False)
    show_debug = st.checkbox("Show ESPN debug", value=False)

    run_btn = st.button("Run slate (and save predictions)")
    update_btn = st.button("Update results (ESPN finals) + Learn")


def show_health(history_df: pd.DataFrame):
    if history_df.empty:
        st.info("No saved history yet. Run a slate to start learning.")
        return
    n = len(history_df)
    done = int((history_df.get("Completed", 0).fillna(0).astype(int) == 1).sum()) if "Completed" in history_df.columns else 0
    st.caption(f"History rows: {n} | Completed with finals: {done} | Pending: {n - done}")


# optional auto loop
if auto_update and not history.empty:
    history = update_history_with_results(history)
    history.to_csv(HISTORY_PATH, index=False)
    new_state, metrics = learn_parameters(history, state)
    state = new_state
    save_state(state)

st.subheader("History status")
show_health(history)

# Run slate
if run_btn:
    preds, slate_url, dbg = predict_slate(int(season), date_val, int(n_sims), state)
    st.caption(f"Slate source: {slate_url}")

    if show_debug:
        with st.expander("ESPN debug"):
            st.json(dbg)

    if preds.empty:
        st.warning("No games found from ESPN for this date.")
    else:
        append_history(preds)
        history = load_history()

        st.success(f"Saved {len(preds)} predictions to history.")
        # show slate outputs
        ok = preds[preds.get("Error").isna()] if "Error" in preds.columns else preds
        st.subheader("All games (new predictions)")
        st.dataframe(ok.sort_values(["Low_Conf_Map", "Matchup"], ascending=[False, True]), use_container_width=True)

# Update results + learn
if update_btn:
    if history.empty:
        st.warning("No history to update. Run a slate first.")
    else:
        history = update_history_with_results(history)
        history.to_csv(HISTORY_PATH, index=False)

        st.success("Updated history with ESPN finals where available.")
        show_health(history)

        new_state, metrics = learn_parameters(history, state)
        state = new_state
        save_state(state)

        st.subheader("Learning metrics")
        st.json(metrics)

        st.subheader("Updated learned parameters")
        st.write({
            "National Avg OE": state["NATIONAL_AVG_OE"],
            "HCA": state["HCA"],
            "Base SD": state["BASE_SD"],
            "K calibration": state["K_CAL"],
        })

# Show latest completed performance
if not history.empty and "Completed" in history.columns:
    done = history[history["Completed"].fillna(0).astype(int) == 1].copy()
    if not done.empty and all(c in done.columns for c in ["Proj_Margin_Home", "Final_Margin_Home", "Proj_Total", "Final_Total"]):
        done = done.dropna(subset=["Proj_Margin_Home", "Final_Margin_Home", "Proj_Total", "Final_Total"])
        if not done.empty:
            st.subheader("Model performance (completed games)")
            rmse_m = rmse(done["Proj_Margin_Home"], done["Final_Margin_Home"])
            rmse_t = rmse(done["Proj_Total"], done["Final_Total"])
            st.write({"RMSE margin": rmse_m, "RMSE total": rmse_t})
            st.dataframe(done.tail(50), use_container_width=True)
"""
Texas Longhorns - 2026 Predicted Home Attendance
Streamlit prediction engine using hierarchical regression coefficients (R-squared = 0.7148).
"""

import json
import pathlib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# 0. Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Texas Attendance Predictor",
    page_icon="\U0001F918",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Burnt orange accent bar under header */
    .stTitle { border-bottom: 3px solid #BF5700; padding-bottom: 8px; }

    /* Metric label styling - smaller labels, bigger numbers */
    [data-testid="stMetricLabel"] p { font-size: 0.82rem; color: #666; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; }

    /* Section headers */
    .stMarkdown h3 { color: #333F48; letter-spacing: 0.02em; }

    /* Home-game attendance highlight */
    .att-home { color: #BF5700; font-size: 1.05em; font-weight: 700; }

    /* Away-game greyed cells */
    .att-away { color: #D6D2C4; }

    /* Spacing between major sections */
    .section-spacer { margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------
CAPACITY = 100_119          # DKR-Texas Memorial Stadium
STADIUM_RECORD = 105_215   # Attendance record (vs Georgia, 2024)
RMSE = 0.0827               # In-sample RMSE from training
Z95 = 1.96                  # For approximate 95% prediction interval

# Texas brand palette
COLOR_PRIMARY = "#BF5700"   # burnt orange - 2026 predicted
COLOR_SECONDARY = "#333F48" # charcoal - 2024 actual
COLOR_GREY = "#D6D2C4"      # warm grey - CI lines / accents
COLOR_COMPARE = "#4A90A4"   # steel blue - comparison scenario

# Prior-season constants (2024 Texas season: 13-3 overall incl. postseason)
PRIOR_SEASON_WIN_PCT = 13 / 16  # 0.8125
PRIOR_SEASON_ATT_RATE = 1.023   # Approximate 2024 home attendance rate

# Coach flags
NEW_COACH = 0
INTERIM_COACH = 0
NEW_CONFERENCE = 0  # Year 2 in SEC

# Historical average high temperatures (deg F) for Austin by month
AUSTIN_TEMP_DEFAULTS = {
    9: 88,   # September
    10: 79,  # October
    11: 65,  # November
}

# ---------------------------------------------------------------------------
# 2. Load coefficients
# ---------------------------------------------------------------------------
COEFF_PATH = pathlib.Path(__file__).parent / "model_coefficients.json"
with open(COEFF_PATH) as f:
    COEFFS = json.load(f)

# ---------------------------------------------------------------------------
# 3. Texas 2026 full schedule (12 games)
# ---------------------------------------------------------------------------
FULL_SCHEDULE = [
    {"week": 1,  "opponent": "Texas State",       "location": "home",    "conference_game": 0},
    {"week": 2,  "opponent": "Ohio State",         "location": "home",    "conference_game": 0},
    {"week": 3,  "opponent": "UTSA",               "location": "home",    "conference_game": 0},
    {"week": 4,  "opponent": "Tennessee",           "location": "away",    "conference_game": 1},
    {"week": 5,  "opponent": "Oklahoma",            "location": "neutral", "conference_game": 1},
    {"week": 7,  "opponent": "Florida",             "location": "home",    "conference_game": 1},
    {"week": 8,  "opponent": "Ole Miss",            "location": "home",    "conference_game": 1},
    {"week": 9,  "opponent": "Mississippi State",   "location": "home",    "conference_game": 1},
    {"week": 10, "opponent": "Missouri",            "location": "away",    "conference_game": 1},
    {"week": 11, "opponent": "LSU",                 "location": "away",    "conference_game": 1},
    {"week": 12, "opponent": "Arkansas",            "location": "home",    "conference_game": 1},
    {"week": 13, "opponent": "Texas A&M",           "location": "away",    "conference_game": 1},
]

# Home-game metadata (7 home games)
HOME_GAMES = {
    1:  {"opponent": "Texas State",       "date": "Sep 5",  "month": 9,  "conference_game": 0, "rivalry": 0, "cupcake": 1, "marquee_noncon": 0, "opponent_away_draw": 0.885, "default_opp_ranked": False, "default_tv": False,  "default_kickoff": "Afternoon"},
    2:  {"opponent": "Ohio State",        "date": "Sep 12", "month": 9,  "conference_game": 0, "rivalry": 0, "cupcake": 0, "marquee_noncon": 1, "opponent_away_draw": 0.977, "default_opp_ranked": True,  "default_tv": True,   "default_kickoff": "Evening"},
    3:  {"opponent": "UTSA",              "date": "Sep 19", "month": 9,  "conference_game": 0, "rivalry": 0, "cupcake": 1, "marquee_noncon": 0, "opponent_away_draw": 0.885, "default_opp_ranked": False, "default_tv": False,  "default_kickoff": "Afternoon"},
    7:  {"opponent": "Florida",           "date": "Oct 17", "month": 10, "conference_game": 1, "rivalry": 0, "cupcake": 0, "marquee_noncon": 0, "opponent_away_draw": 0.951, "default_opp_ranked": False, "default_tv": True,   "default_kickoff": "Evening"},
    8:  {"opponent": "Ole Miss",          "date": "Oct 24", "month": 10, "conference_game": 1, "rivalry": 0, "cupcake": 0, "marquee_noncon": 0, "opponent_away_draw": 0.926, "default_opp_ranked": False, "default_tv": True,   "default_kickoff": "Evening"},
    9:  {"opponent": "Mississippi State", "date": "Oct 31", "month": 10, "conference_game": 1, "rivalry": 0, "cupcake": 0, "marquee_noncon": 0, "opponent_away_draw": 0.916, "default_opp_ranked": False, "default_tv": True,   "default_kickoff": "Afternoon"},
    12: {"opponent": "Arkansas",          "date": "Nov 21", "month": 11, "conference_game": 1, "rivalry": 1, "cupcake": 0, "marquee_noncon": 0, "opponent_away_draw": 0.940, "default_opp_ranked": False, "default_tv": True,   "default_kickoff": "Evening"},
}

HOME_WEEKS = sorted(HOME_GAMES.keys())

# ---------------------------------------------------------------------------
# 4. Scenario presets (W/L for all 12 games, keyed by week)
# ---------------------------------------------------------------------------
PRESETS = {
    "Optimistic (11-1)": {
        1: "W", 2: "W", 3: "W", 4: "W", 5: "W", 7: "W",
        8: "W", 9: "W", 10: "W", 11: "L", 12: "W", 13: "W",
    },
    "Expected (9-3)": {
        1: "W", 2: "L", 3: "W", 4: "L", 5: "W", 7: "W",
        8: "W", 9: "W", 10: "W", 11: "L", 12: "W", 13: "W",
    },
    "Pessimistic (6-6)": {
        1: "W", 2: "L", 3: "W", 4: "L", 5: "L", 7: "W",
        8: "L", 9: "W", 10: "L", 11: "L", 12: "W", 13: "L",
    },
    "Custom": {},
}

ALL_WEEKS = [g["week"] for g in FULL_SCHEDULE]

# ---------------------------------------------------------------------------
# 5. Prediction helpers
# ---------------------------------------------------------------------------

def compute_cumulative_record(wl: dict, up_to_week: int) -> tuple[int, int]:
    """Return (wins, losses) entering the given week (exclusive of that week)."""
    wins, losses = 0, 0
    for g in FULL_SCHEDULE:
        if g["week"] >= up_to_week:
            break
        result = wl.get(g["week"], "W")
        if result == "W":
            wins += 1
        else:
            losses += 1
    return wins, losses


def build_feature_vector(week: int, meta: dict, overrides: dict,
                         wl: dict) -> dict:
    """Build the full feature dict for one home game."""
    wins, losses = compute_cumulative_record(wl, week)

    kickoff = overrides.get(f"kickoff_{week}", meta["default_kickoff"])
    national_tv = overrides.get(f"tv_{week}", meta["default_tv"])
    temp = overrides.get(f"temp_{week}", AUSTIN_TEMP_DEFAULTS[meta["month"]])
    precip = overrides.get(f"precip_{week}", 0)
    opp_ranked = overrides.get(f"ranked_{week}", meta["default_opp_ranked"])

    return {
        "season_wins": wins,
        "season_losses": losses,
        "prior_season_win_pct": PRIOR_SEASON_WIN_PCT,
        "prior_season_att_rate": PRIOR_SEASON_ATT_RATE,
        "opponent_ranked": int(opp_ranked),
        "conference_game": meta["conference_game"],
        "marquee_noncon": meta["marquee_noncon"],
        "cupcake": meta["cupcake"],
        "opponent_away_draw": meta["opponent_away_draw"],
        "non_weekend_game": 0,  # All 2026 home games are Saturday
        "new_coach": NEW_COACH,
        "interim_coach": INTERIM_COACH,
        "temperature": temp,
        "precip_severity": precip,
        "severe_weather": int(precip >= 3),
        "national_tv": int(national_tv),
        "new_conference": NEW_CONFERENCE,
        "rivalry": meta["rivalry"],
        "kickoff_time": kickoff,
    }


def predict_attendance(features: dict, coeffs: dict) -> float:
    """Manual dot-product prediction using exported coefficients."""
    rate = coeffs["Intercept"]

    # Team fixed effect
    rate += coeffs.get("C(team)[T.Texas]", 0)

    # Season fixed effect - map future seasons to last training year (2023)
    rate += coeffs.get("C(season_str)[T.2023]", 0)

    # Kickoff time (baseline = Afternoon)
    kt = features["kickoff_time"]
    if kt != "Afternoon":
        rate += coeffs.get(f"C(kickoff_time)[T.{kt}]", 0)

    # Continuous & binary features
    feature_keys = [
        "season_wins", "season_losses", "prior_season_win_pct",
        "prior_season_att_rate", "opponent_ranked", "conference_game",
        "marquee_noncon", "cupcake", "opponent_away_draw",
        "non_weekend_game", "new_coach", "interim_coach",
        "temperature", "precip_severity", "severe_weather",
        "national_tv", "new_conference", "rivalry",
    ]
    for key in feature_keys:
        rate += coeffs.get(key, 0) * features[key]

    # Interaction: season_losses * rivalry
    rate += coeffs.get("season_losses:rivalry", 0) * (
        features["season_losses"] * features["rivalry"]
    )

    return rate


def predict_all_games(wl: dict, overrides: dict) -> pd.DataFrame:
    """Run predictions for all 7 home games and return a DataFrame."""
    rows = []
    for week in HOME_WEEKS:
        meta = HOME_GAMES[week]
        fv = build_feature_vector(week, meta, overrides, wl)
        pred_rate = predict_attendance(fv, COEFFS)
        headcount = pred_rate * CAPACITY
        lo = (pred_rate - Z95 * RMSE) * CAPACITY
        hi = (pred_rate + Z95 * RMSE) * CAPACITY
        wins, losses = compute_cumulative_record(wl, week)
        rows.append({
            "Week": week,
            "Date": meta["date"],
            "Opponent": meta["opponent"],
            "Record": f"{wins}-{losses}",
            "Kickoff": overrides.get(f"kickoff_{week}", meta["default_kickoff"]),
            "TV": "Yes" if overrides.get(f"tv_{week}", meta["default_tv"]) else "No",
            "Temp (F)": overrides.get(f"temp_{week}", AUSTIN_TEMP_DEFAULTS[meta["month"]]),
            "Predicted Rate": pred_rate,
            "Headcount": int(round(headcount)),
            "Low 95%": int(round(lo)),
            "High 95%": int(round(hi)),
            "Conference": "SEC" if meta["conference_game"] else "Non-Conf",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. UI control constants
# ---------------------------------------------------------------------------
KICKOFF_OPTIONS = ["Morning", "Afternoon", "Evening", "Night"]
WEATHER_OPTIONS = ["Clear", "Light", "Moderate", "Severe"]
WEATHER_TO_SEVERITY = {"Clear": 0, "Light": 1, "Moderate": 2, "Severe": 3}

# ---------------------------------------------------------------------------
# 7. Session-state initialisation
# ---------------------------------------------------------------------------
if "preset" not in st.session_state:
    st.session_state.preset = "Expected (9-3)"

# W/L keys (all 12 games)
for g in FULL_SCHEDULE:
    wk = g["week"]
    if f"wl_{wk}" not in st.session_state:
        st.session_state[f"wl_{wk}"] = PRESETS["Expected (9-3)"].get(wk, "W")

# Home-game override keys
for wk, meta in HOME_GAMES.items():
    if f"kick_{wk}" not in st.session_state:
        st.session_state[f"kick_{wk}"] = meta["default_kickoff"]
    if f"weather_{wk}" not in st.session_state:
        st.session_state[f"weather_{wk}"] = "Clear"
    if f"temp_{wk}" not in st.session_state:
        st.session_state[f"temp_{wk}"] = AUSTIN_TEMP_DEFAULTS[meta["month"]]
    if f"ranked_{wk}" not in st.session_state:
        st.session_state[f"ranked_{wk}"] = meta["default_opp_ranked"]

# ---------------------------------------------------------------------------
# 8. Main display
# ---------------------------------------------------------------------------
HELMET_PATH = pathlib.Path(__file__).parent / "texas_helmet.png"

header_left, header_right = st.columns([0.8, 5])
with header_left:
    st.markdown("<div style='margin-top:40px'>", unsafe_allow_html=True)
    st.image(str(HELMET_PATH), width=90)
    st.markdown("</div>", unsafe_allow_html=True)
with header_right:
    st.title("Texas Longhorns -- 2026 Predicted Home Attendance")
    st.caption(
        "Predicted attendance based on historical patterns from 8 seasons of college football home games."
    )

# ---- Auto-detect Custom mode ----
current_wl = {g["week"]: st.session_state.get(f"wl_{g['week']}", "W")
              for g in FULL_SCHEDULE}
if (st.session_state.preset != "Custom"
        and current_wl != PRESETS.get(st.session_state.preset, {})):
    st.session_state.preset = "Custom"

# ---- Preset selector ----
preset_col, reset_col = st.columns([4, 1])
with preset_col:
    preset_choice = st.selectbox(
        "Season scenario",
        list(PRESETS.keys()),
        index=list(PRESETS.keys()).index(st.session_state.preset),
        help="Set win/loss outcomes for the season. Choose a preset or customize each game.",
    )
with reset_col:
    st.markdown("<br>", unsafe_allow_html=True)
    reset_clicked = st.button("Reset to defaults")

if preset_choice != st.session_state.preset and preset_choice != "Custom":
    st.session_state.preset = preset_choice
    for wk, result in PRESETS[preset_choice].items():
        st.session_state[f"wl_{wk}"] = result
    st.rerun()
elif preset_choice == "Custom" and st.session_state.preset != "Custom":
    st.session_state.preset = "Custom"

if reset_clicked:
    st.session_state.preset = "Expected (9-3)"
    for wk, result in PRESETS["Expected (9-3)"].items():
        st.session_state[f"wl_{wk}"] = result
    for wk, meta in HOME_GAMES.items():
        st.session_state[f"kick_{wk}"] = meta["default_kickoff"]
        st.session_state[f"weather_{wk}"] = "Clear"
        st.session_state[f"temp_{wk}"] = AUSTIN_TEMP_DEFAULTS[meta["month"]]
        st.session_state[f"ranked_{wk}"] = meta["default_opp_ranked"]
    st.rerun()

# ---- Scenario comparison toggle ----
cmp_toggle_col, cmp_select_col = st.columns([2, 3])
with cmp_toggle_col:
    compare_mode = st.toggle("Compare to another scenario", key="compare_mode")
if compare_mode:
    compare_presets = [k for k in PRESETS if k != "Custom"]
    with cmp_select_col:
        compare_choice = st.selectbox(
            "Compare scenario",
            compare_presets,
            key="compare_preset",
            label_visibility="collapsed",
        )
else:
    compare_choice = None

# ---- Build wl + overrides from session state ----
wl = {g["week"]: st.session_state.get(f"wl_{g['week']}", "W")
      for g in FULL_SCHEDULE}

overrides = {}
for wk in HOME_WEEKS:
    meta = HOME_GAMES[wk]
    overrides[f"kickoff_{wk}"] = st.session_state.get(f"kick_{wk}", meta["default_kickoff"])
    weather = st.session_state.get(f"weather_{wk}", "Clear")
    overrides[f"precip_{wk}"] = WEATHER_TO_SEVERITY.get(weather, 0)
    overrides[f"temp_{wk}"] = st.session_state.get(f"temp_{wk}", AUSTIN_TEMP_DEFAULTS[meta["month"]])
    overrides[f"ranked_{wk}"] = st.session_state.get(f"ranked_{wk}", meta["default_opp_ranked"])
    overrides[f"tv_{wk}"] = meta["default_tv"]

# ---- Scenario B predictions (comparison mode) ----
if compare_mode and compare_choice:
    wl_b = PRESETS[compare_choice]
    df_b = predict_all_games(wl_b, overrides)
    total_wins_b = sum(1 for v in wl_b.values() if v == "W")
    total_losses_b = len(wl_b) - total_wins_b
else:
    df_b = None

# ---- Summary metrics (top of page) ----
df = predict_all_games(wl, overrides)
total_wins = sum(1 for v in wl.values() if v == "W")
total_losses = len(wl) - total_wins

prior_total = 99_171 + 101_892 + 102_850 + 101_388 + 105_215 + 103_375 + 102_811  # 2024
yoy_pct = (df['Headcount'].sum() - prior_total) / prior_total
avg_att_rate = df['Predicted Rate'].mean()

with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)
    if df_b is not None:
        delta_att = int(df['Headcount'].sum() - df_b['Headcount'].sum())
        col1.metric("**Total Season Attendance**", f"{df['Headcount'].sum():,.0f}",
                     delta=f"{delta_att:+,} vs {compare_choice}")
        yoy_pct_b = (df_b['Headcount'].sum() - prior_total) / prior_total
        col2.metric("Y/Y Total Attendance", f"{yoy_pct:+.1%}",
                     delta=f"{yoy_pct - yoy_pct_b:+.1%} vs {compare_choice}")
        col3.metric("Avg Attendance Rate", f"{avg_att_rate:.1%}",
                     delta=f"{avg_att_rate - df_b['Predicted Rate'].mean():+.1%} vs {compare_choice}")
        col4.metric("Projected Record", f"{total_wins}-{total_losses}",
                     delta=f"vs {total_wins_b}-{total_losses_b}")
    else:
        col1.metric("**Total Season Attendance**", f"{df['Headcount'].sum():,.0f}")
        col2.metric("Y/Y Total Attendance", f"{yoy_pct:+.1%}", delta=f"{df['Headcount'].sum() - prior_total:+,.0f}")
        col3.metric("Avg Attendance Rate", f"{avg_att_rate:.1%}")
        col4.metric("Projected Record", f"{total_wins}-{total_losses}")

st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

# ---- Schedule table ----
st.markdown("### 2026 Season Schedule")
st.caption(
    f"Darrell K Royal-Texas Memorial Stadium · Capacity: {CAPACITY:,} · "
    f"Attendance Record: {STADIUM_RECORD:,} (vs Georgia, 2024)"
)

# Column widths: W/L | Opponent | Kickoff | Weather | Temp | Ranked | Pred. Att.
W = [0.5, 2.2, 1.2, 1.2, 0.8, 0.7, 1.4]

# Header
hdr = st.columns(W)
for col, label in zip(hdr, ["W/L", "Opponent", "Kickoff", "Weather",
                             "Temp (\u00b0F)", "Ranked", "Pred. Att."]):
    col.markdown(f"**{label}**")

# Rows
for game in FULL_SCHEDULE:
    wk = game["week"]
    is_home = game["location"] == "home"
    cols = st.columns(W)

    # -- W/L (all games) --
    cols[0].selectbox(
        f"wl {wk}", ["W", "L"], key=f"wl_{wk}",
        label_visibility="collapsed",
    )

    # -- Opponent --
    loc_tag = {"home": "vs", "away": "@", "neutral": "vs (N)"}[game["location"]]
    if is_home:
        cols[1].markdown(
            f"**Wk {wk}** &nbsp; vs {game['opponent']}<br>"
            f"<span style='font-size:0.85em;color:#666'>{HOME_GAMES[wk]['date']}</span>",
            unsafe_allow_html=True,
        )
    else:
        cols[1].markdown(
            f"<span style='color:#888'>Wk {wk} &nbsp; {loc_tag} {game['opponent']}</span>",
            unsafe_allow_html=True,
        )

    if is_home:
        meta = HOME_GAMES[wk]

        # -- Kickoff --
        cols[2].selectbox(
            f"kick {wk}", KICKOFF_OPTIONS, key=f"kick_{wk}",
            label_visibility="collapsed",
        )
        # -- Weather --
        cols[3].selectbox(
            f"weather {wk}", WEATHER_OPTIONS, key=f"weather_{wk}",
            label_visibility="collapsed",
        )
        # -- Temp --
        cols[4].number_input(
            f"temp {wk}", min_value=30, max_value=110, key=f"temp_{wk}",
            label_visibility="collapsed",
        )
        # -- Ranked --
        cols[5].checkbox(
            f"ranked {wk}", key=f"ranked_{wk}",
            label_visibility="collapsed",
        )
        # -- Predicted attendance --
        fv = build_feature_vector(wk, meta, overrides, wl)
        pred_rate = predict_attendance(fv, COEFFS)
        headcount = int(round(pred_rate * CAPACITY))
        if headcount > STADIUM_RECORD:
            badge = ' <span title="Projected stadium record" style="font-size:0.75em;background:#BF5700;color:#fff;padding:1px 5px;border-radius:3px;margin-left:4px">RECORD</span>'
        elif headcount > CAPACITY:
            badge = ' <span title="Above capacity" style="font-size:0.75em;background:#333F48;color:#fff;padding:1px 5px;border-radius:3px;margin-left:4px">SELLOUT+</span>'
        else:
            badge = ""
        cols[6].markdown(
            f'<span class="att-home">{headcount:,}</span>{badge}',
            unsafe_allow_html=True,
        )
    else:
        for c in (2, 3, 4, 5, 6):
            cols[c].markdown('<span class="att-away">--</span>',
                             unsafe_allow_html=True)

# ---- Cumulative attendance line chart ----
st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
st.markdown("### Cumulative Home Attendance")

# 2026 predicted cumulative
cumul_df = df[["Week", "Opponent", "Headcount", "Low 95%", "High 95%"]].copy()
cumul_df["Cumulative"] = cumul_df["Headcount"].cumsum()
cumul_df["Cumul Low 95%"] = cumul_df["Low 95%"].cumsum()
cumul_df["Cumul High 95%"] = cumul_df["High 95%"].cumsum()
cumul_df["Game"] = (cumul_df.index + 1).astype(int)
cumul_df["Series"] = "2026 Projected"

# 2024 actual cumulative
ACTUAL_2024 = [
    {"Game": 1, "Opponent": "Colorado State",    "Attendance": 99_171},
    {"Game": 2, "Opponent": "UTSA",              "Attendance": 101_892},
    {"Game": 3, "Opponent": "UL Monroe",         "Attendance": 102_850},
    {"Game": 4, "Opponent": "Mississippi State",  "Attendance": 101_388},
    {"Game": 5, "Opponent": "Georgia",            "Attendance": 105_215},
    {"Game": 6, "Opponent": "Florida",            "Attendance": 103_375},
    {"Game": 7, "Opponent": "Kentucky",           "Attendance": 102_811},
]
prior_df = pd.DataFrame(ACTUAL_2024)
prior_df["Cumulative"] = prior_df["Attendance"].cumsum()
prior_df["Series"] = "2024 Actual"
prior_df["Game Attendance"] = prior_df["Attendance"]

# Add per-game attendance to 2026 data for tooltip
cumul_df["Game Attendance"] = cumul_df["Headcount"]

# Combine for charting
chart_combined = pd.concat([
    cumul_df[["Game", "Opponent", "Cumulative", "Game Attendance", "Series"]],
    prior_df[["Game", "Opponent", "Cumulative", "Game Attendance", "Series"]],
], ignore_index=True)

# Add Scenario B cumulative if comparing
if df_b is not None:
    cumul_b = df_b[["Week", "Opponent", "Headcount"]].copy()
    cumul_b["Cumulative"] = cumul_b["Headcount"].cumsum()
    cumul_b["Game"] = (cumul_b.index + 1).astype(int)
    cumul_b["Series"] = f"2026 {compare_choice}"
    cumul_b["Game Attendance"] = cumul_b["Headcount"]
    chart_combined = pd.concat([
        chart_combined,
        cumul_b[["Game", "Opponent", "Cumulative", "Game Attendance", "Series"]],
    ], ignore_index=True)

# Also need CI data keyed by Game number
ci_df = cumul_df[["Game", "Cumul Low 95%", "Cumul High 95%"]].copy()

# Build delta-from-2024 data
actual_values = [g["Attendance"] for g in ACTUAL_2024]
delta_rows = []
for i, row in cumul_df.iterrows():
    delta_rows.append({
        "Game": int(row["Game"]),
        "Opponent": row["Opponent"],
        "Delta": int(row["Headcount"]) - actual_values[i],
        "Game Attendance": int(row["Headcount"]),
        "Series": "2026 Projected",
    })
delta_df = pd.DataFrame(delta_rows)
delta_df["Cumulative Delta"] = delta_df["Delta"].cumsum()

if df_b is not None:
    delta_b_rows = []
    for i, row in df_b.iterrows():
        delta_b_rows.append({
            "Game": i + 1,
            "Opponent": row["Opponent"],
            "Delta": int(row["Headcount"]) - actual_values[i],
            "Game Attendance": int(row["Headcount"]),
            "Series": f"2026 {compare_choice}",
        })
    delta_b_df = pd.DataFrame(delta_b_rows)
    delta_b_df["Cumulative Delta"] = delta_b_df["Delta"].cumsum()
    delta_combined = pd.concat([delta_df, delta_b_df], ignore_index=True)
else:
    delta_combined = delta_df

import altair as alt

# -- Shared encoding helpers --
color_domain = (["2026 Projected", f"2026 {compare_choice}", "2024 Actual"]
                if df_b is not None
                else ["2026 Projected", "2024 Actual"])
color_range = ([COLOR_PRIMARY, COLOR_COMPARE, COLOR_SECONDARY]
               if df_b is not None
               else [COLOR_PRIMARY, COLOR_SECONDARY])

x_axis = alt.X("Game:O", title="Home Game #",
               axis=alt.Axis(labelAngle=0))

# =============================================
# LEFT CHART: Cumulative season total (zero=False)
# =============================================
lines = (
    alt.Chart(chart_combined)
    .mark_line(point=alt.OverlayMarkDef(size=50), strokeWidth=2.5)
    .encode(
        x=x_axis,
        y=alt.Y("Cumulative:Q", title="Cumulative Attendance",
                scale=alt.Scale(zero=False),
                axis=alt.Axis(format=",.0f")),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(domain=color_domain, range=color_range),
            legend=alt.Legend(title=None, orient="top"),
        ),
        tooltip=[
            alt.Tooltip("Series:N"),
            alt.Tooltip("Opponent:N"),
            alt.Tooltip("Game Attendance:Q", format=",", title="Game Attendance"),
            alt.Tooltip("Cumulative:Q", format=",", title="Cumulative Total"),
            alt.Tooltip("Game:O", title="Home Game #"),
        ],
    )
)

# Data labels on 2026 predicted line (smaller for half-width)
label_df = cumul_df[["Game", "Cumulative"]].copy()
data_labels = (
    alt.Chart(label_df)
    .mark_text(dy=-12, fontSize=9, fontWeight="bold", color=COLOR_PRIMARY)
    .encode(
        x=alt.X("Game:O"),
        y=alt.Y("Cumulative:Q"),
        text=alt.Text("Cumulative:Q", format=",.0f"),
    )
)

ci_high = (
    alt.Chart(ci_df)
    .mark_line(strokeDash=[4, 4], color=COLOR_GREY, strokeWidth=1.2)
    .encode(x=alt.X("Game:O"), y=alt.Y("Cumul High 95%:Q"))
)
ci_low = (
    alt.Chart(ci_df)
    .mark_line(strokeDash=[4, 4], color=COLOR_GREY, strokeWidth=1.2)
    .encode(x=alt.X("Game:O"), y=alt.Y("Cumul Low 95%:Q"))
)

cumul_chart = (lines + data_labels + ci_high + ci_low).properties(
    width="container", height=370,
).configure_view(strokeWidth=0)

# =============================================
# RIGHT CHART: Ahead / behind 2024
# =============================================
# Color scale for delta chart (no 2024 Actual line — it's the zero baseline)
delta_domain = (["2026 Projected", f"2026 {compare_choice}"]
                if df_b is not None
                else ["2026 Projected"])
delta_range = ([COLOR_PRIMARY, COLOR_COMPARE]
               if df_b is not None
               else [COLOR_PRIMARY])

zero_line = (
    alt.Chart(pd.DataFrame({"y": [0]}))
    .mark_rule(strokeDash=[4, 4], color=COLOR_GREY, strokeWidth=1.2)
    .encode(y="y:Q")
)

delta_lines = (
    alt.Chart(delta_combined)
    .mark_line(point=alt.OverlayMarkDef(size=50), strokeWidth=2.5)
    .encode(
        x=x_axis,
        y=alt.Y("Cumulative Delta:Q", title="+/\u2212 Fans vs 2024",
                axis=alt.Axis(format=",.0f")),
        color=alt.Color(
            "Series:N",
            scale=alt.Scale(domain=delta_domain, range=delta_range),
            legend=alt.Legend(title=None, orient="top"),
        ),
        tooltip=[
            alt.Tooltip("Series:N"),
            alt.Tooltip("Opponent:N"),
            alt.Tooltip("Game Attendance:Q", format=",", title="Game Attendance"),
            alt.Tooltip("Cumulative Delta:Q", format="+,", title="vs 2024"),
            alt.Tooltip("Game:O", title="Home Game #"),
        ],
    )
)

# Data labels on delta chart
delta_label_df = delta_df[["Game", "Cumulative Delta"]].copy()
delta_labels = (
    alt.Chart(delta_label_df)
    .mark_text(dy=-12, fontSize=9, fontWeight="bold", color=COLOR_PRIMARY)
    .encode(
        x=alt.X("Game:O"),
        y=alt.Y("Cumulative Delta:Q"),
        text=alt.Text("Cumulative Delta:Q", format="+,"),
    )
)

delta_layers = zero_line + delta_lines + delta_labels
if df_b is not None:
    delta_b_label_df = delta_b_df[["Game", "Cumulative Delta"]].copy()
    delta_b_labels = (
        alt.Chart(delta_b_label_df)
        .mark_text(dy=16, fontSize=9, fontWeight="bold", color=COLOR_COMPARE)
        .encode(
            x=alt.X("Game:O"),
            y=alt.Y("Cumulative Delta:Q"),
            text=alt.Text("Cumulative Delta:Q", format="+,"),
        )
    )
    delta_layers = delta_layers + delta_b_labels

delta_chart = delta_layers.properties(
    width="container", height=370,
).configure_view(strokeWidth=0)

# =============================================
# Render side by side
# =============================================
chart_left, chart_right = st.columns(2)
with chart_left:
    st.markdown("**Season Total**")
    st.altair_chart(cumul_chart, use_container_width=True)
with chart_right:
    st.markdown("**Attendance vs 2024**")
    st.altair_chart(delta_chart, use_container_width=True)

st.caption(
    f"Stadium capacity: {CAPACITY:,}. "
    "Dotted lines show the range where we expect actual attendance to fall 95% of the time."
)

if df_b is not None:
    delta_total = int(df['Headcount'].sum() - df_b['Headcount'].sum())
    game_deltas = df[['Week', 'Opponent', 'Headcount']].copy()
    game_deltas['Delta'] = df['Headcount'].values - df_b['Headcount'].values
    max_row = game_deltas.loc[game_deltas['Delta'].abs().idxmax()]
    direction = "more" if delta_total > 0 else "fewer"
    st.markdown(
        f"Your scenario draws **{abs(delta_total):,} {direction}** fans "
        f"than {compare_choice} across 7 home games. "
        f"Largest swing: Week {int(max_row['Week'])} vs {max_row['Opponent']} "
        f"({int(max_row['Delta']):+,})."
    )

# ---- Model details expander ----
with st.expander("About this forecast"):
    st.markdown("""
**How it works**: This forecast uses a statistical model built on 8 seasons of
attendance data across 67 teams. It accounts for factors like win/loss record,
opponent quality, kickoff time, weather, rivalries, and TV coverage.

**Key assumptions for 2026**:
- Based on Texas's 2024 results (13-3 record)
- Steve Sarkisian entering year 6 as head coach
- Second full year in the SEC
- Opponent drawing power based on historical travel patterns

**Accuracy**: The model explains roughly 71% of the variation in attendance
and is typically within ~8,300 fans of the actual number.
""")

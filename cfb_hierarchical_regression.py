"""
Hierarchical Regression Model for CFB Home Game Attendance
Predicts attendance_rate (attendance / stadium_capacity) and measures
fan sensitivity to team performance.

Train: 2015-2023  |  Validate: 2024  |  Predict: 2025
Conferences: SEC, Big 12, ACC, Big Ten
"""

import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import seaborn as sns
import warnings
import time
import json
import os
import pickle
from datetime import datetime
from scipy import stats

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
API_KEY = os.environ.get('CFB_API_KEY', '')
BASE_URL = 'https://api.collegefootballdata.com'
HEADERS = {'Authorization': f'Bearer {API_KEY}', 'Accept': 'application/json'}
CONFERENCES = ['SEC', 'Big 12', 'ACC', 'Big Ten']
SEASONS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
TRAIN_SEASONS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
TEST_SEASON = 2024

CACHE_DIR = r'C:\Users\tkasi\Documents\cfb_attendance\cache'
OUTPUT_DIR = r'C:\Users\tkasi\Documents\cfb_attendance'
os.makedirs(CACHE_DIR, exist_ok=True)

# Economist-style formatting
CONF_COLORS = {
    'SEC': '#E3120B',
    'Big 12': '#006BA6',
    'ACC': '#6A8A3E',
    'Big Ten': '#AD8E00',
}
ECON_BG = '#F2E8D5'
ECON_TEXT = '#333333'
ECON_LIGHT = '#BBBBBB'
ECON_SERIF = 'Georgia'
ECON_SANS = 'Franklin Gothic Medium'


def economist_ax(ax):
    """Apply Economist-style formatting to an axes."""
    ax.set_facecolor(ECON_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', length=0, labelsize=9, colors=ECON_TEXT)
    ax.grid(True, axis='x', color=ECON_LIGHT, linewidth=0.5, linestyle='-')
    ax.grid(False, axis='y')
    ax.set_axisbelow(True)


# =============================================================================
# Data Fetching & Caching
# =============================================================================
def fetch_json(endpoint, params=None):
    """Fetch JSON from API with exponential backoff on rate limits."""
    for attempt in range(5):
        resp = requests.get(f'{BASE_URL}{endpoint}', headers=HEADERS, params=params or {})
        if resp.status_code == 429:
            wait = 2 ** attempt
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        ct = resp.headers.get('Content-Type', '')
        if 'json' not in ct:
            print(f"  Warning: {endpoint} returned non-JSON ({ct}), returning []")
            return []
        return resp.json()
    resp.raise_for_status()


def load_or_fetch(cache_name, fetch_fn):
    """Load from cache file if it exists, otherwise fetch and cache."""
    cache_path = os.path.join(CACHE_DIR, f'{cache_name}.json')
    if os.path.exists(cache_path):
        print(f"  Loading {cache_name} from cache...")
        with open(cache_path, 'r') as f:
            return json.load(f)
    data = fetch_fn()
    with open(cache_path, 'w') as f:
        json.dump(data, f)
    return data


# =============================================================================
# Step 0: Data Collection
# =============================================================================
print("=" * 70)
print("STEP 0: DATA COLLECTION & VARIABLE CONSTRUCTION")
print("=" * 70)

# --- Existing cached data ---
print("\n--- Loading existing cached data ---")
fbs_teams = load_or_fetch('fbs_teams_2025', lambda: fetch_json('/teams/fbs', {'year': 2025}))

all_teams = []
team_location_info = {}
for t in fbs_teams:
    loc = t.get('location') or {}
    conf = t.get('conference')
    school = t['school']
    team_location_info[school] = {
        'timezone': loc.get('timezone', 'America/New_York'),
        'latitude': loc.get('latitude'),
        'longitude': loc.get('longitude'),
        'dome': loc.get('dome', False),
        'capacity': loc.get('capacity'),
    }
    if conf not in CONFERENCES:
        continue
    all_teams.append({
        'team_id': t['id'],
        'school': school,
        'conference': conf,
        'stadium_capacity': loc.get('capacity'),
        'color': t.get('color', '#888888'),
    })

teams_df = pd.DataFrame(all_teams)
team_names = set(teams_df['school'])
team_conf_map = dict(zip(teams_df['school'], teams_df['conference']))
team_capacity_map = dict(zip(teams_df['school'], teams_df['stadium_capacity']))
print(f"  {len(teams_df)} teams across {len(CONFERENCES)} conferences")

venues_raw = load_or_fetch('venues', lambda: fetch_json('/venues'))
venue_capacity = {v['id']: v['capacity'] for v in venues_raw if v.get('capacity')}
venue_info = {}
for v in venues_raw:
    venue_info[v['id']] = {
        'capacity': v.get('capacity'),
        'dome': v.get('dome', False),
        'timezone': v.get('timezone'),
        'latitude': v.get('latitude'),
        'longitude': v.get('longitude'),
    }

all_games = load_or_fetch('all_games', lambda: None)  # Must already exist
games_df = pd.DataFrame(all_games).drop_duplicates(subset='id')
print(f"  {len(games_df)} unique games loaded")

# --- Fetch new API data (110 calls, cached after first run) ---
print("\n--- Fetching new API data ---")

# Rankings
print("Fetching AP/Coaches rankings...")
all_rankings = {}
for year in SEASONS:
    data = load_or_fetch(f'rankings_{year}',
                         lambda y=year: fetch_json('/rankings', {'year': y}))
    all_rankings[year] = data
    time.sleep(0.3)

# SP+ ratings
print("Fetching SP+ ratings...")
all_sp = {}
for year in SEASONS:
    data = load_or_fetch(f'sp_ratings_{year}',
                         lambda y=year: fetch_json('/ratings/sp', {'year': y}))
    all_sp[year] = data
    time.sleep(0.3)

# Talent composite
print("Fetching talent composites...")
all_talent = {}
for year in SEASONS:
    data = load_or_fetch(f'talent_{year}',
                         lambda y=year: fetch_json('/talent', {'year': y}))
    all_talent[year] = data
    time.sleep(0.3)

# Weather
print("Fetching weather data...")
all_weather = {}
for year in SEASONS:
    data = load_or_fetch(f'weather_{year}',
                         lambda y=year: fetch_json('/games/weather',
                                                   {'year': y, 'seasonType': 'regular'}))
    all_weather[year] = data
    time.sleep(0.3)

# Coaches
print("Fetching coaching data...")
all_coaches = {}
for year in SEASONS:
    data = load_or_fetch(f'coaches_{year}',
                         lambda y=year: fetch_json('/coaches', {'year': y}))
    all_coaches[year] = data
    time.sleep(0.3)

# Pregame win probability
print("Fetching pregame win probabilities...")
all_wp = {}
for year in SEASONS:
    data = load_or_fetch(f'pregame_wp_{year}',
                         lambda y=year: fetch_json('/metrics/wp/pregame',
                                                   {'year': y, 'seasonType': 'regular'}))
    all_wp[year] = data
    time.sleep(0.3)

# Media (TV)
print("Fetching media/broadcast data...")
all_media = {}
for year in SEASONS:
    data = load_or_fetch(f'media_{year}',
                         lambda y=year: fetch_json('/games/media',
                                                   {'year': y, 'seasonType': 'regular'}))
    all_media[year] = data
    time.sleep(0.3)

# Player stats
print("Fetching player stats (passing, rushing, receiving)...")
all_player_passing = {}
all_player_rushing = {}
all_player_receiving = {}
for year in SEASONS:
    all_player_passing[year] = load_or_fetch(
        f'player_stats_passing_{year}',
        lambda y=year: fetch_json('/stats/player/season', {'year': y, 'category': 'passing'}))
    time.sleep(0.3)
    all_player_rushing[year] = load_or_fetch(
        f'player_stats_rushing_{year}',
        lambda y=year: fetch_json('/stats/player/season', {'year': y, 'category': 'rushing'}))
    time.sleep(0.3)
    all_player_receiving[year] = load_or_fetch(
        f'player_stats_receiving_{year}',
        lambda y=year: fetch_json('/stats/player/season', {'year': y, 'category': 'receiving'}))
    time.sleep(0.3)

# Betting lines
print("Fetching betting lines...")
all_betting = {}
for year in SEASONS:
    data = load_or_fetch(f'betting_{year}',
                         lambda y=year: fetch_json('/lines',
                                                   {'year': y, 'seasonType': 'regular'}))
    all_betting[year] = data
    time.sleep(0.3)

print("  All API data loaded/cached.")

# =============================================================================
# Build Lookup Dictionaries
# =============================================================================
print("\n--- Building lookup dictionaries ---")

# Rankings: {(year, week, team): rank}
rankings_lookup = {}
for year, weeks_data in all_rankings.items():
    for week_obj in weeks_data:
        week = week_obj.get('week', 0)
        for poll in week_obj.get('polls', []):
            for entry in poll.get('ranks', []):
                school = entry.get('school')
                rank = entry.get('rank')
                if school and rank:
                    key = (year, week, school)
                    if key not in rankings_lookup or rank < rankings_lookup[key]:
                        rankings_lookup[key] = rank

# SP+ ratings: {(year, team): {'overall': x, 'offense': x, 'defense': x}}
sp_lookup = {}
for year, data in all_sp.items():
    for entry in data:
        team = entry.get('team')
        if team:
            sp_lookup[(year, team)] = {
                'overall': entry.get('rating'),
                'offense': entry.get('offense', {}).get('rating') if isinstance(entry.get('offense'), dict) else entry.get('offense'),
                'defense': entry.get('defense', {}).get('rating') if isinstance(entry.get('defense'), dict) else entry.get('defense'),
            }

# Talent: {(year, team): talent_rating}
talent_lookup = {}
for year, data in all_talent.items():
    for entry in data:
        team = entry.get('school')
        talent = entry.get('talent')
        if team and talent:
            talent_lookup[(year, team)] = float(talent)

# Weather: {game_id: {temperature, windSpeed, weatherCondition}}
weather_lookup = {}
for year, data in all_weather.items():
    for entry in data:
        gid = entry.get('id') or entry.get('gameId')
        if gid:
            weather_lookup[gid] = {
                'temperature': entry.get('temperature'),
                'windSpeed': entry.get('windSpeed'),
                'weatherCondition': entry.get('weatherCondition'),
            }

# Coaches: {(year, team): {'coach': name, 'first_year': bool}}
coaches_by_year_team = {}
for year, data in all_coaches.items():
    for entry in data:
        first_name = entry.get('firstName', '')
        last_name = entry.get('lastName', '')
        coach_name = f"{first_name} {last_name}".strip()
        for season_entry in entry.get('seasons', []):
            if season_entry.get('year') == year:
                team = season_entry.get('school')
                if team:
                    coaches_by_year_team[(year, team)] = {
                        'coach': coach_name,
                        'games': season_entry.get('games', 0),
                        'wins': season_entry.get('wins', 0),
                    }

# Pregame WP: {game_id: {homeWinProb, awayWinProb}}
wp_lookup = {}
for year, data in all_wp.items():
    for entry in data:
        gid = entry.get('gameId')
        if gid:
            wp_lookup[gid] = {
                'homeWinProb': entry.get('homeWinProb'),
                'awayWinProb': entry.get('awayWinProb'),
            }

# Media: {game_id: [outlets]}
media_lookup = {}
for year, data in all_media.items():
    for entry in data:
        gid = entry.get('id') or entry.get('gameId')
        if gid is None:
            continue
        outlets = entry.get('mediaType', '')
        outlet_name = entry.get('outlet', '')
        if gid not in media_lookup:
            media_lookup[gid] = []
        media_lookup[gid].append({
            'mediaType': outlets,
            'outlet': outlet_name,
        })

# National TV networks
NATIONAL_TV = {'ABC', 'CBS', 'NBC', 'FOX', 'ESPN', 'ESPN2', 'FS1', 'FOX Sports 1'}

# Player stats: build top-50 by yards for each category per year
def build_top_players(stats_dict, stat_type):
    """Build {(year, team): True} for teams with a top-50 player in PRIOR season."""
    top_by_year = {}
    for year, data in stats_dict.items():
        players = []
        for entry in data:
            team = entry.get('team')
            player = entry.get('player')
            stat_val = 0
            for s in entry.get('statType', ''):
                pass
            # Stats are in a list of stat entries
            if isinstance(entry.get('stat'), (int, float)):
                stat_val = float(entry.get('stat', 0))
            elif isinstance(entry.get('stat'), str):
                try:
                    stat_val = float(entry['stat'])
                except (ValueError, TypeError):
                    stat_val = 0
            # Try to get yards specifically
            stat_name = entry.get('statType', '')
            if stat_type == 'passing' and stat_name == 'YDS':
                players.append({'team': team, 'player': player, 'yards': stat_val})
            elif stat_type == 'rushing' and stat_name == 'YDS':
                players.append({'team': team, 'player': player, 'yards': stat_val})
            elif stat_type == 'receiving' and stat_name == 'YDS':
                players.append({'team': team, 'player': player, 'yards': stat_val})
            elif stat_name in ('YDS', 'YARDS', 'yards'):
                players.append({'team': team, 'player': player, 'yards': stat_val})
            elif not stat_name:
                # Some API responses have flat structure
                players.append({'team': team, 'player': player, 'yards': stat_val})

        if players:
            pdf = pd.DataFrame(players)
            if 'yards' in pdf.columns and len(pdf) > 0:
                pdf = pdf.sort_values('yards', ascending=False)
                top50_teams = set(pdf.head(50)['team'].dropna().unique())
                top_by_year[year] = top50_teams
    return top_by_year

top_passers_by_year = build_top_players(all_player_passing, 'passing')
top_rushers_by_year = build_top_players(all_player_rushing, 'rushing')
top_receivers_by_year = build_top_players(all_player_receiving, 'receiving')

# Betting: {game_id: spread}
betting_lookup = {}
for year, data in all_betting.items():
    for entry in data:
        gid = entry.get('id') or entry.get('gameId')
        lines = entry.get('lines', [])
        if gid and lines:
            for line in lines:
                spread = line.get('spread')
                if spread is not None:
                    try:
                        betting_lookup[gid] = float(spread)
                    except (ValueError, TypeError):
                        pass
                    break

print("  Lookups built.")

# =============================================================================
# Construct Analysis Dataset
# =============================================================================
print("\n--- Constructing analysis dataset ---")

# Filter to regular season, non-neutral-site, home team in target conferences
reg_games = games_df[
    (games_df['seasonType'] == 'regular') &
    (~games_df['neutralSite'].astype(bool)) &
    (games_df['homeTeam'].isin(team_names)) &
    (games_df['attendance'].notna()) &
    (games_df['attendance'] > 0) &
    (games_df['season'] != 2020)
].copy()

print(f"  {len(reg_games)} regular-season home games for target conferences")

# --- Win/Loss records for all teams ---
completed_games = games_df[games_df['completed'] == True].copy()
win_records = []
for _, g in completed_games.iterrows():
    home, away = g['homeTeam'], g['awayTeam']
    hp, ap = g.get('homePoints'), g.get('awayPoints')
    season, week = g['season'], g.get('week', 0)
    st = g.get('seasonType', 'regular')
    if hp is None or ap is None:
        continue
    if home:
        win_records.append({
            'school': home, 'season': season, 'week': week,
            'seasonType': st, 'win': 1 if hp > ap else 0,
            'is_home': True
        })
    if away:
        win_records.append({
            'school': away, 'season': season, 'week': week,
            'seasonType': st, 'win': 1 if ap > hp else 0,
            'is_home': False
        })

win_df = pd.DataFrame(win_records)

# Season win% per team
season_winpct = win_df.groupby(['school', 'season']).agg(
    wins=('win', 'sum'), games=('win', 'count')
).reset_index()
season_winpct['win_pct'] = season_winpct['wins'] / season_winpct['games']
season_wpct_lookup = dict(zip(
    zip(season_winpct['school'], season_winpct['season']),
    season_winpct['win_pct']
))

# Prior season win%
prior_season_wpct = {}
for school in team_names:
    for i, year in enumerate(SEASONS):
        prev_year = SEASONS[i - 1] if i > 0 else None
        if prev_year:
            prior_season_wpct[(school, year)] = season_wpct_lookup.get(
                (school, prev_year), np.nan)
        else:
            # 2015: use conference mean
            prior_season_wpct[(school, year)] = np.nan

# Fill 2015 with conference mean
conf_means_2014 = {}
for conf in CONFERENCES:
    conf_teams = teams_df[teams_df['conference'] == conf]['school'].tolist()
    vals = [season_wpct_lookup.get((t, 2014), np.nan) for t in conf_teams]
    vals = [v for v in vals if not np.isnan(v)]
    conf_means_2014[conf] = np.mean(vals) if vals else 0.5

for school in team_names:
    if np.isnan(prior_season_wpct.get((school, 2015), np.nan)):
        conf = team_conf_map.get(school, 'SEC')
        prior_season_wpct[(school, 2015)] = conf_means_2014.get(conf, 0.5)

# Cumulative season win% AND separate wins/losses before each game (lagged)
regular_wins = win_df[win_df['seasonType'] == 'regular'].sort_values(['school', 'season', 'week'])

cum_season_wpct = {}
cum_season_wins_losses = {}  # {(school, season, week): (wins, losses)}
for (school, season), grp in regular_wins.groupby(['school', 'season']):
    cum_wins = 0
    cum_losses = 0
    for _, row in grp.iterrows():
        week = row['week']
        cum_season_wins_losses[(school, season, week)] = (cum_wins, cum_losses)
        cum_games = cum_wins + cum_losses
        if cum_games > 0:
            cum_season_wpct[(school, season, week)] = cum_wins / cum_games
        else:
            cum_season_wpct[(school, season, week)] = prior_season_wpct.get(
                (school, season), 0.5)
        if row['win']:
            cum_wins += 1
        else:
            cum_losses += 1

# Rolling 3-game win% across ALL games (home + away), lagged 1 game
rolling_all_wpct = {}
all_regular = regular_wins.sort_values(['school', 'season', 'week'])
for school, grp in all_regular.groupby('school'):
    results = grp['win'].tolist()
    seasons = grp['season'].tolist()
    weeks = grp['week'].tolist()
    for i in range(len(results)):
        if i == 0:
            val = prior_season_wpct.get((school, seasons[i]), 0.5)
        elif i < 3:
            val = np.mean(results[:i])
        else:
            val = np.mean(results[i-3:i])
        rolling_all_wpct[(school, seasons[i], weeks[i])] = val

# Prior season home attendance rate per team
print("  Computing prior season attendance rates...")
prior_season_att = {}  # {(school, season): mean home attendance rate}
# First compute home attendance rate per team per season from all regular home games
home_att_by_season = {}
for _, g in games_df.iterrows():
    if g.get('seasonType') != 'regular' or g.get('neutralSite'):
        continue
    home = g['homeTeam']
    att = g.get('attendance')
    season = g['season']
    if not att or att <= 0 or season == 2020:
        continue
    cap = team_capacity_map.get(home)
    if not cap or cap <= 0:
        vid = g.get('venueId')
        if vid and vid in venue_capacity:
            cap = venue_capacity[vid]
    if not cap or cap <= 0:
        continue
    key = (home, season)
    if key not in home_att_by_season:
        home_att_by_season[key] = []
    home_att_by_season[key].append(att / cap)

home_att_rate_by_season = {k: np.mean(v) for k, v in home_att_by_season.items()}

# Now build prior season attendance rate lookup
for school in team_names:
    for i, year in enumerate(SEASONS):
        prev_year = SEASONS[i - 1] if i > 0 else None
        if prev_year:
            prior_season_att[(school, year)] = home_att_rate_by_season.get(
                (school, prev_year), np.nan)
        else:
            prior_season_att[(school, year)] = np.nan

# Fill 2015 with conference mean prior att rate
conf_mean_att_2014 = {}
for conf in CONFERENCES:
    conf_teams = teams_df[teams_df['conference'] == conf]['school'].tolist()
    vals = [home_att_rate_by_season.get((t, 2014), np.nan) for t in conf_teams]
    vals = [v for v in vals if not np.isnan(v)]
    conf_mean_att_2014[conf] = np.mean(vals) if vals else 0.85

for school in team_names:
    if np.isnan(prior_season_att.get((school, 2015), np.nan)):
        conf = team_conf_map.get(school, 'SEC')
        prior_season_att[(school, 2015)] = conf_mean_att_2014.get(conf, 0.85)

print(f"  Prior season att rates: {len([v for v in prior_season_att.values() if not np.isnan(v)])} entries")

# --- New coach detection ---
new_coach_lookup = {}
for year in SEASONS:
    for school in team_names:
        curr = coaches_by_year_team.get((year, school))
        prev_year = year - 1 if year != 2021 else 2019
        prev = coaches_by_year_team.get((prev_year, school))
        if curr and prev:
            new_coach_lookup[(school, year)] = curr['coach'] != prev['coach']
        elif curr and not prev:
            new_coach_lookup[(school, year)] = False  # Unknown, assume not new
        else:
            new_coach_lookup[(school, year)] = False

# --- Rivalry detection (curated + data-driven) ---
print("  Building rivalry set...")
# Curated major rivalries (bidirectional -- stored as frozensets)
CURATED_RIVALRIES = {
    # SEC
    frozenset(('Alabama', 'Auburn')),            # Iron Bowl
    frozenset(('Alabama', 'Tennessee')),          # Third Saturday in October
    frozenset(('Alabama', 'LSU')),
    frozenset(('Georgia', 'Florida')),            # World's Largest Cocktail Party
    frozenset(('Georgia', 'Georgia Tech')),       # Clean Old-Fashioned Hate
    frozenset(('Georgia', 'Auburn')),             # Deep South's Oldest Rivalry
    frozenset(('Florida', 'Florida State')),
    frozenset(('Florida', 'Tennessee')),
    frozenset(('Florida', 'Miami')),
    frozenset(('Ole Miss', 'Mississippi State')),  # Egg Bowl
    frozenset(('South Carolina', 'Clemson')),     # Palmetto Bowl
    frozenset(('Kentucky', 'Louisville')),        # Governor's Cup
    frozenset(('Tennessee', 'Vanderbilt')),
    frozenset(('Arkansas', 'LSU')),
    frozenset(('Arkansas', 'Texas A&M')),
    frozenset(('Texas A&M', 'LSU')),
    frozenset(('Missouri', 'Kansas')),            # Border War
    frozenset(('Texas', 'Oklahoma')),             # Red River Rivalry
    frozenset(('Texas', 'Texas A&M')),
    # Big Ten
    frozenset(('Ohio State', 'Michigan')),        # The Game
    frozenset(('Michigan', 'Michigan State')),
    frozenset(('Ohio State', 'Penn State')),
    frozenset(('Minnesota', 'Wisconsin')),        # Paul Bunyan's Axe
    frozenset(('Iowa', 'Iowa State')),            # Cy-Hawk
    frozenset(('Iowa', 'Minnesota')),             # Floyd of Rosedale
    frozenset(('Iowa', 'Nebraska')),              # Heroes Trophy
    frozenset(('Indiana', 'Purdue')),             # Old Oaken Bucket
    frozenset(('Illinois', 'Northwestern')),
    frozenset(('Penn State', 'Pittsburgh')),
    frozenset(('Nebraska', 'Colorado')),
    frozenset(('Rutgers', 'Penn State')),
    frozenset(('Maryland', 'Penn State')),
    frozenset(('USC', 'UCLA')),                   # Victory Bell
    frozenset(('USC', 'Notre Dame')),
    frozenset(('Oregon', 'Oregon State')),        # Civil War
    frozenset(('Oregon', 'Washington')),
    frozenset(('Washington', 'Washington State')),  # Apple Cup
    # ACC
    frozenset(('Duke', 'North Carolina')),        # Blue-White Rivalry
    frozenset(('NC State', 'North Carolina')),    # Tobacco Road
    frozenset(('Clemson', 'South Carolina')),
    frozenset(('Florida State', 'Miami')),        # War on I-4 (old ACC)
    frozenset(('Virginia', 'Virginia Tech')),     # Commonwealth Cup
    frozenset(('Louisville', 'Kentucky')),
    frozenset(('Georgia Tech', 'Georgia')),
    frozenset(('Pittsburgh', 'West Virginia')),   # Backyard Brawl
    frozenset(('Stanford', 'California')),        # Big Game
    frozenset(('Boston College', 'Notre Dame')),
    frozenset(('Syracuse', 'Pittsburgh')),
    frozenset(('Wake Forest', 'Duke')),
    frozenset(('SMU', 'TCU')),                    # Iron Skillet
    # Big 12
    frozenset(('Kansas', 'Kansas State')),        # Sunflower Showdown
    frozenset(('Oklahoma State', 'Oklahoma')),    # Bedlam
    frozenset(('Iowa State', 'Iowa')),
    frozenset(('West Virginia', 'Pittsburgh')),
    frozenset(('Baylor', 'TCU')),                 # Revivalry
    frozenset(('Texas Tech', 'Texas')),
    frozenset(('BYU', 'Utah')),                   # Holy War
    frozenset(('Colorado', 'Colorado State')),
    frozenset(('Arizona', 'Arizona State')),      # Territorial Cup
    frozenset(('Cincinnati', 'Louisville')),      # Keg of Nails
    frozenset(('UCF', 'South Florida')),          # War on I-4
    # Cross-conference notable
    frozenset(('Notre Dame', 'Stanford')),
    frozenset(('Notre Dame', 'Navy')),
    frozenset(('Notre Dame', 'Purdue')),
    frozenset(('UCLA', 'USC')),
}

# Data-driven augmentation: find home-away pairs with 2+ games and >10pp excess
# above the home team's average attendance rate
_team_avg_att = df_placeholder = None  # Will compute after dataset is built
# We'll add data-driven rivalries after dataset construction

# Precipitation severity mapping
PRECIP_SEVERITY = {
    'Clear': 0, 'Fair': 0, 'Cloudy': 0, 'Overcast': 0, 'Fog': 0,
    'Light Rain': 1, 'Light Snowfall': 1,
    'Rain': 2, 'Rain Shower': 2, 'Sleet': 2, 'Snowfall': 2,
    'Heavy Rain': 3, 'Heavy Rain Shower': 3, 'Thunderstorm': 3,
    'Storm': 3, 'Heavy Snowfall': 3, 'Heavy Sleet': 3, 'Heavy Sleet Shower': 3,
}

def is_rivalry(home, away):
    """Check if a game is a rivalry matchup."""
    return frozenset((home, away)) in CURATED_RIVALRIES

# --- Opponent away draw rate (P4 teams only) ---
print("  Computing opponent away draw rates...")
# For each P4 team, compute their average away attendance rate across road games
# (attendance / host stadium capacity when this team is the visitor)
all_p4_schools = set(teams_df['school'])
opp_away_draw = {}  # {team: average away attendance rate}

away_draw_records = []
for _, g in games_df.iterrows():
    if g.get('seasonType') != 'regular':
        continue
    if g.get('neutralSite'):
        continue
    away = g['awayTeam']
    if away not in all_p4_schools:
        continue
    att = g.get('attendance')
    if not att or att <= 0:
        continue
    season = g['season']
    if season == 2020:
        continue
    # Use home team's stadium capacity (the venue the away team is visiting)
    home = g['homeTeam']
    cap = team_capacity_map.get(home)
    if not cap or cap <= 0:
        vid = g.get('venueId')
        if vid and vid in venue_capacity:
            cap = venue_capacity[vid]
    if not cap or cap <= 0:
        continue
    away_draw_records.append({
        'school': away,
        'season': season,
        'rate': att / cap,
    })

away_draw_df = pd.DataFrame(away_draw_records)
if len(away_draw_df) > 0:
    opp_away_draw = away_draw_df.groupby('school')['rate'].mean().to_dict()
    p4_mean_away_draw = np.mean(list(opp_away_draw.values()))
    print(f"  Computed away draw for {len(opp_away_draw)} P4 teams "
          f"(mean={p4_mean_away_draw:.3f})")
else:
    p4_mean_away_draw = 0.85

# --- Build the observation-level dataset ---
print("  Building observation-level dataset...")
rows = []
for _, g in reg_games.iterrows():
    gid = g['id']
    home = g['homeTeam']
    away = g['awayTeam']
    season = g['season']
    week = g.get('week', 1)
    attendance = g['attendance']

    # Stadium capacity
    cap = team_capacity_map.get(home)
    if not cap or cap <= 0:
        vid = g.get('venueId')
        if vid and vid in venue_capacity:
            cap = venue_capacity[vid]
    if not cap or cap <= 0:
        continue

    attendance_rate = attendance / cap

    # Conference
    conf = team_conf_map.get(home, '')

    # Elo
    home_elo = g.get('homePregameElo')
    away_elo = g.get('awayPregameElo')

    # Win percentages
    recent_all_win_pct = rolling_all_wpct.get((home, season, week), 0.5)
    season_win_pct = cum_season_wpct.get((home, season, week), 0.5)
    sw, sl = cum_season_wins_losses.get((home, season, week), (0, 0))
    season_wins = sw
    season_losses = sl
    prior_season_win_pct_val = prior_season_wpct.get((home, season), 0.5)
    if np.isnan(prior_season_win_pct_val):
        prior_season_win_pct_val = 0.5
    prior_season_att_rate = prior_season_att.get((home, season), np.nan)
    if np.isnan(prior_season_att_rate):
        prior_season_att_rate = 0.85  # fallback

    # Conference game
    conference_game = 1 if g.get('conferenceGame') else 0

    # Opponent rank (use AP poll week or closest)
    opp_rank = None
    for w in range(week, 0, -1):
        r = rankings_lookup.get((season, w, away))
        if r:
            opp_rank = r
            break
    opponent_ranked = 1 if opp_rank and opp_rank <= 25 else 0

    # FCS opponent
    away_class = g.get('awayClassification', 'fbs')
    fcs_opponent = 1 if away_class != 'fbs' else 0

    # SP+ ratings
    home_sp = sp_lookup.get((season, home), {})
    away_sp = sp_lookup.get((season, away), {})
    home_sp_overall = home_sp.get('overall')
    away_sp_overall = away_sp.get('overall')
    sp_diff = None
    if home_sp_overall is not None and away_sp_overall is not None:
        try:
            sp_diff = float(home_sp_overall) - float(away_sp_overall)
        except (TypeError, ValueError):
            sp_diff = None

    # FCS imputation: use 5th percentile FBS values
    if fcs_opponent and away_elo is None:
        fbs_elos = [g2.get('awayPregameElo') for _, g2 in reg_games.iterrows()
                    if g2.get('awayPregameElo') and g2.get('awayClassification') == 'fbs']
        if fbs_elos:
            away_elo = np.percentile([e for e in fbs_elos if e], 5)

    # Expected win probability
    wp_data = wp_lookup.get(gid, {})
    expected_win_prob = wp_data.get('homeWinProb')
    if expected_win_prob is None and sp_diff is not None:
        expected_win_prob = 1 / (1 + np.exp(-0.03 * sp_diff))
    if expected_win_prob is None:
        expected_win_prob = 0.5

    # Kickoff time & day of week
    start_date_str = g.get('startDate', '')
    kickoff_hour_local = None
    day_of_week = 'Saturday'
    if start_date_str:
        try:
            dt = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
            # Approximate local time using venue timezone offset
            tz_name = team_location_info.get(home, {}).get('timezone', 'America/New_York')
            # Simple offset map (avoid pytz dependency)
            tz_offsets = {
                'America/New_York': -5, 'America/Chicago': -6,
                'America/Denver': -7, 'America/Los_Angeles': -8,
                'America/Indiana/Indianapolis': -5, 'America/Kentucky/Louisville': -5,
                'America/Phoenix': -7,
            }
            offset = tz_offsets.get(tz_name, -5)
            from datetime import timedelta
            local_dt = dt + timedelta(hours=offset)
            kickoff_hour_local = local_dt.hour
            day_of_week = local_dt.strftime('%A')
        except Exception:
            pass

    # Categorize kickoff time
    if kickoff_hour_local is not None:
        if kickoff_hour_local < 12:
            kickoff_time = 'Morning'
        elif kickoff_hour_local < 17:
            kickoff_time = 'Afternoon'
        elif kickoff_hour_local < 20:
            kickoff_time = 'Evening'
        else:
            kickoff_time = 'Night'
    else:
        kickoff_time = 'Afternoon'  # default

    non_weekend_game = 0 if day_of_week in ('Saturday', 'Friday') else 1

    # Bye week before (check if team had a game the prior week)
    bye_week_before = 0
    if week > 1:
        prior_games = reg_games[
            (reg_games['homeTeam'] == home) & (reg_games['season'] == season) &
            (reg_games['week'] == week - 1)
        ]
        away_prior = games_df[
            (games_df['awayTeam'] == home) & (games_df['season'] == season) &
            (games_df['week'] == week - 1) & (games_df['seasonType'] == 'regular')
        ]
        if len(prior_games) == 0 and len(away_prior) == 0:
            bye_week_before = 1

    # Opponent quality categories
    away_conf = g.get('awayConference', '')
    g5_conferences = {'Mountain West', 'Sun Belt', 'Conference USA', 'Mid-American',
                      'American Athletic', 'MAC', 'C-USA', 'AAC', 'MWC'}
    away_season_wpct = season_wpct_lookup.get((away, season), 0.5)
    cupcake = 1 if (fcs_opponent or (away_conf in g5_conferences and away_season_wpct < 0.4)) else 0

    marquee_noncon = 0
    if not conference_game and not fcs_opponent:
        if opponent_ranked or (away_elo and home_elo and away_elo > 1500):
            marquee_noncon = 1

    # Opponent away draw (P4 only; non-P4 gets P4 mean, handled by fcs/cupcake dummies)
    if away in all_p4_schools:
        opp_draw = opp_away_draw.get(away, p4_mean_away_draw)
    else:
        opp_draw = p4_mean_away_draw  # neutral fill for non-P4

    # Weather
    w_data = weather_lookup.get(gid, {})
    temperature = w_data.get('temperature')
    wind_speed = w_data.get('windSpeed')
    weather_cond = w_data.get('weatherCondition', '')

    # Precipitation severity (0=none, 1=light, 2=moderate, 3=severe)
    precip_severity = PRECIP_SEVERITY.get(weather_cond, 0) if weather_cond else 0

    # Dome check - no bad weather in domes
    vid = g.get('venueId')
    is_dome = False
    if vid and vid in venue_info:
        is_dome = venue_info[vid].get('dome', False)
    if not is_dome:
        is_dome = team_location_info.get(home, {}).get('dome', False)

    if is_dome:
        temperature = 72  # Controlled environment
        wind_speed = 0
        precip_severity = 0

    bad_weather = 0
    severe_weather = 0
    if temperature is not None:
        try:
            temp_f = float(temperature)
            wind_f = float(wind_speed) if wind_speed else 0
            if temp_f < 40 or wind_f > 20 or precip_severity > 0:
                bad_weather = 1
            if temp_f < 32 or temp_f > 95:
                severe_weather = 1
        except (TypeError, ValueError):
            pass

    # Rivalry
    rivalry = 1 if is_rivalry(home, away) else 0

    # New coach
    new_coach = 1 if new_coach_lookup.get((home, season), False) else 0

    # Interim coach (games < 6 in a season where coach changed)
    interim_coach = 0
    coach_data = coaches_by_year_team.get((season, home))
    if coach_data and coach_data.get('games', 12) < 6:
        interim_coach = 1

    # Top-50 players (from PRIOR season)
    prior_yr = season - 1 if season != 2021 else 2019
    top50_passer = 1 if home in top_passers_by_year.get(prior_yr, set()) else 0
    top50_rusher = 1 if home in top_rushers_by_year.get(prior_yr, set()) else 0
    top50_receiver = 1 if home in top_receivers_by_year.get(prior_yr, set()) else 0

    # National TV
    game_media = media_lookup.get(gid, [])
    national_tv = 0
    for m in game_media:
        outlet = m.get('outlet', '')
        if outlet in NATIONAL_TV:
            national_tv = 1
            break

    # New conference (first year in new conf)
    new_conference = 0
    # Check if team's conference changed from prior year
    # Use home conference from game data vs prior year
    if season > 2015:
        prior_yr_games = games_df[
            (games_df['homeTeam'] == home) & (games_df['season'] == prior_yr) &
            (games_df['seasonType'] == 'regular')
        ]
        if len(prior_yr_games) > 0:
            prev_conf = prior_yr_games.iloc[0].get('homeConference', '')
            curr_conf = g.get('homeConference', '')
            if prev_conf and curr_conf and prev_conf != curr_conf:
                new_conference = 1

    # Spread
    spread = betting_lookup.get(gid)

    rows.append({
        'game_id': gid,
        'season': season,
        'week': week,
        'home_team': home,
        'away_team': away,
        'conference': conf,
        'attendance': attendance,
        'stadium_capacity': cap,
        'attendance_rate': attendance_rate,
        'home_elo': home_elo,
        'away_elo': away_elo,
        'recent_all_win_pct': recent_all_win_pct,
        'season_wins': season_wins,
        'season_losses': season_losses,
        'season_win_pct': season_win_pct,
        'prior_season_win_pct': prior_season_win_pct_val,
        'prior_season_att_rate': prior_season_att_rate,
        'conference_game': conference_game,
        'opponent_ranked': opponent_ranked,
        'opponent_rank': opp_rank if opp_rank else 0,
        'fcs_opponent': fcs_opponent,
        'cupcake': cupcake,
        'marquee_noncon': marquee_noncon,
        'opponent_away_draw': opp_draw,
        'expected_win_prob': expected_win_prob,
        'kickoff_time': kickoff_time,
        'non_weekend_game': non_weekend_game,
        'bye_week_before': bye_week_before,
        'new_coach': new_coach,
        'interim_coach': interim_coach,
        'top50_passer': top50_passer,
        'top50_rusher': top50_rusher,
        'top50_receiver': top50_receiver,
        'temperature': temperature,
        'precip_severity': precip_severity,
        'bad_weather': bad_weather,
        'severe_weather': severe_weather,
        'rivalry': rivalry,
        'national_tv': national_tv,
        'new_conference': new_conference,
        'spread': spread,
        'sp_diff': sp_diff,
        'day_of_week': day_of_week,
    })

df = pd.DataFrame(rows)
print(f"  Built dataset: {len(df)} observations, {len(df.columns)} columns")

# Data-driven rivalry augmentation: flag pairs with 2+ games and >10pp excess
print("  Augmenting rivalry set with data-driven pairs...")
_team_avg = df.groupby('home_team')['attendance_rate'].mean()
_pair_stats = df.groupby(['home_team', 'away_team']).agg(
    _mean_rate=('attendance_rate', 'mean'),
    _games=('attendance_rate', 'count')
).reset_index()
_pair_stats['_team_avg'] = _pair_stats['home_team'].map(_team_avg)
_pair_stats['_excess'] = _pair_stats['_mean_rate'] - _pair_stats['_team_avg']
_dd_rivals = _pair_stats[(_pair_stats['_games'] >= 2) & (_pair_stats['_excess'] > 0.10)]
data_driven_rivalries = set()
for _, pr in _dd_rivals.iterrows():
    pair = frozenset((pr['home_team'], pr['away_team']))
    if pair not in CURATED_RIVALRIES:
        data_driven_rivalries.add(pair)
        CURATED_RIVALRIES.add(pair)

# Re-apply rivalry flag with augmented set
df['rivalry'] = df.apply(lambda r: 1 if frozenset((r['home_team'], r['away_team']))
                         in CURATED_RIVALRIES else 0, axis=1)
n_curated = len(CURATED_RIVALRIES) - len(data_driven_rivalries)
print(f"  Rivalries: {n_curated} curated + {len(data_driven_rivalries)} data-driven = "
      f"{len(CURATED_RIVALRIES)} total pairs")
print(f"  Data-driven additions: {', '.join(sorted(str(s) for s in data_driven_rivalries))}")
print(f"  Games flagged as rivalry: {df['rivalry'].sum()} ({df['rivalry'].mean()*100:.1f}%)")

# Fill missing numeric values
for col in ['temperature', 'spread', 'sp_diff', 'home_elo', 'away_elo', 'prior_season_att_rate']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

# Create C() compatible team and season columns
df['team'] = df['home_team'].astype(str)
df['season_str'] = df['season'].astype(str)

# Train/test split
train = df[df['season'].isin(TRAIN_SEASONS)].copy()
test = df[df['season'] == TEST_SEASON].copy()
print(f"  Train: {len(train)} obs ({min(TRAIN_SEASONS)}-{max(TRAIN_SEASONS)})")
print(f"  Test:  {len(test)} obs ({TEST_SEASON})")

# Save dataset
csv_path = os.path.join(OUTPUT_DIR, 'model_data.csv')
df.to_csv(csv_path, index=False)
print(f"  Saved model_data.csv ({len(df)} rows)")

# =============================================================================
# Helper: Run OLS with HC3 robust standard errors
# =============================================================================
def run_ols(formula, data, label="Model"):
    """Fit OLS model and return results with HC3 robust standard errors."""
    model = smf.ols(formula, data=data).fit(cov_type='HC3')
    return model


def print_model_summary(model, label=""):
    """Print compact model summary."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  R-squared:     {model.rsquared:.4f}")
    print(f"  Adj R-squared: {model.rsquared_adj:.4f}")
    print(f"  Observations:  {int(model.nobs)}")
    print(f"  F-statistic:   {model.fvalue:.2f} (p={model.f_pvalue:.2e})")

    # In-sample metrics
    resid = model.resid
    fitted = model.fittedvalues
    actual = fitted + resid
    mae = mean_absolute_error(actual, fitted)
    mape = np.mean(np.abs(resid / actual)) * 100
    rmse = np.sqrt(mean_squared_error(actual, fitted))
    print(f"  MAE:           {mae:.4f}")
    print(f"  MAPE:          {mape:.2f}%")
    print(f"  RMSE:          {rmse:.4f}")


def get_holdout_metrics(model, test_data):
    """Compute metrics on holdout set.

    Maps unseen categorical levels to the most recent training level so that
    patsy doesn't raise on out-of-sample seasons.
    """
    try:
        td = test_data.copy()
        # Map 2024 season_str to the last training season so the FE is defined
        known_seasons = [str(s) for s in TRAIN_SEASONS]
        if 'season_str' in td.columns:
            td['season_str'] = td['season_str'].apply(
                lambda x: x if x in known_seasons else known_seasons[-1])
        # Map unseen teams to most common training team (fallback)
        known_teams = set(model.model.data.frame['team'].unique())
        if 'team' in td.columns:
            td['team'] = td['team'].apply(
                lambda x: x if x in known_teams else list(known_teams)[0])
        pred = model.predict(td)
        actual = test_data['attendance_rate']
        valid = actual.notna() & pred.notna()
        actual = actual[valid]
        pred = pred[valid]
        mae = mean_absolute_error(actual, pred)
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        rmse = np.sqrt(mean_squared_error(actual, pred))
        return {'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'n': len(actual),
                'predictions': pred, 'actuals': actual}
    except Exception as e:
        print(f"  Holdout prediction error: {e}")
        return None


# =============================================================================
# Open results file
# =============================================================================
results_path = os.path.join(OUTPUT_DIR, 'hierarchical_results.txt')
results_lines = []


def log(text=""):
    """Print and log to results file."""
    print(text)
    results_lines.append(text)


# =============================================================================
# Step 1: Baseline Model
# =============================================================================
log("\n" + "=" * 70)
log("STEP 1: BASELINE MODEL")
log("=" * 70)

baseline_formula = (
    'attendance_rate ~ C(team) + C(season_str) + '
    'season_wins + season_losses + prior_season_win_pct + '
    'prior_season_att_rate + opponent_ranked + conference_game'
)

baseline_model = run_ols(baseline_formula, train, "Baseline")
print_model_summary(baseline_model, "Baseline Model")

# Key coefficients
key_vars_baseline = ['season_wins', 'season_losses',
                     'prior_season_win_pct', 'prior_season_att_rate',
                     'opponent_ranked', 'conference_game']
log("\n  Key Coefficients:")
log(f"  {'Variable':<25} {'Coeff':>10} {'Std Err':>10} {'p-value':>10}")
log(f"  {'-'*55}")
for var in key_vars_baseline:
    if var in baseline_model.params:
        coef = baseline_model.params[var]
        se = baseline_model.bse[var]
        pval = baseline_model.pvalues[var]
        sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
        log(f"  {var:<25} {coef:>10.4f} {se:>10.4f} {pval:>10.4f} {sig}")

baseline_r2 = baseline_model.rsquared
log(f"\n  Baseline R-squared: {baseline_r2:.4f}")

holdout_baseline = get_holdout_metrics(baseline_model, test)
if holdout_baseline:
    log(f"  Holdout MAE:  {holdout_baseline['MAE']:.4f}")
    log(f"  Holdout MAPE: {holdout_baseline['MAPE']:.2f}%")
    log(f"  Holdout RMSE: {holdout_baseline['RMSE']:.4f}")

# =============================================================================
# Step 2: Sequential Block Addition
# =============================================================================
log("\n" + "=" * 70)
log("STEP 2: SEQUENTIAL BLOCK ADDITION")
log("=" * 70)

current_formula = baseline_formula
current_r2 = baseline_r2
block_results = {}
retained_vars = list(key_vars_baseline)

blocks = {
    'A: Opponent Quality': ['marquee_noncon', 'cupcake', 'expected_win_prob', 'opponent_away_draw'],
    'B: Schedule Factors': ['non_weekend_game', 'C(kickoff_time)', 'bye_week_before'],
    'C: Coaching': ['new_coach', 'interim_coach'],
    'D: Weather': ['temperature', 'precip_severity', 'severe_weather'],
    'E: Special Events': ['national_tv', 'new_conference'],
    'F: Rivalry': ['rivalry'],
}

for block_name, block_vars in blocks.items():
    log(f"\n--- Block {block_name} ---")

    test_formula = current_formula + ' + ' + ' + '.join(block_vars)
    try:
        block_model = run_ols(test_formula, train, block_name)
    except Exception as e:
        log(f"  ERROR fitting block: {e}")
        continue

    new_r2 = block_model.rsquared
    delta_r2 = new_r2 - current_r2
    log(f"  R-squared: {new_r2:.4f}  (delta: {delta_r2:+.4f})")

    # Report coefficients for block variables
    keep_vars = []
    for var in block_vars:
        clean_var = var.replace('C(', '').replace(')', '')
        matching = [p for p in block_model.params.index if clean_var in p]
        if matching:
            for m in matching:
                coef = block_model.params[m]
                pval = block_model.pvalues[m]
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                log(f"    {m:<35} coeff={coef:>8.4f}  p={pval:.4f} {sig}")
                if pval < 0.10:
                    if var not in keep_vars:
                        keep_vars.append(var)
        else:
            log(f"    {var:<35} (not in model output)")

    # Recommendation
    if delta_r2 > 0.001 or keep_vars:
        vars_to_add = keep_vars if keep_vars else block_vars
        log(f"  RECOMMEND: Retain {', '.join(vars_to_add)} (delta R2={delta_r2:+.4f})")
        current_formula += ' + ' + ' + '.join(vars_to_add)
        current_r2 = run_ols(current_formula, train).rsquared
        retained_vars.extend([v.replace('C(', '').replace(')', '') for v in vars_to_add])
    else:
        log(f"  RECOMMEND: Drop block (delta R2={delta_r2:+.4f}, no significant variables)")

    block_results[block_name] = {
        'delta_r2': delta_r2,
        'new_r2': new_r2,
        'kept': len(keep_vars) > 0 or delta_r2 > 0.001,
    }

log(f"\n  Current formula after block addition:")
log(f"  R-squared: {current_r2:.4f}")

# =============================================================================
# Step 3: Interaction Terms
# =============================================================================
log("\n" + "=" * 70)
log("STEP 3: INTERACTION TERMS")
log("=" * 70)

interactions = [
    ('season_losses:new_coach', 'Season losses x New coach'),
    ('season_losses:conference_game', 'Season losses x Conference game'),
    ('season_losses:rivalry', 'Season losses x Rivalry'),
]

interaction_base_r2 = current_r2
for interaction_term, label in interactions:
    log(f"\n--- {label} ---")
    test_formula_int = current_formula + f' + {interaction_term}'
    try:
        int_model = run_ols(test_formula_int, train)
        new_r2 = int_model.rsquared
        delta = new_r2 - interaction_base_r2

        # Find the interaction coefficient
        matching = [p for p in int_model.params.index if ':' in p and
                    all(t.strip() in p for t in interaction_term.split(':'))]
        if matching:
            for m in matching:
                coef = int_model.params[m]
                pval = int_model.pvalues[m]
                se = int_model.bse[m]
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                log(f"  {m}: coeff={coef:.4f}, SE={se:.4f}, p={pval:.4f} {sig}")

        log(f"  Delta R2: {delta:+.5f}")

        if delta > 0.0005 and any(int_model.pvalues[m] < 0.10 for m in matching):
            log(f"  RECOMMEND: Retain (significant interaction)")
            current_formula = test_formula_int
            interaction_base_r2 = new_r2
        else:
            log(f"  RECOMMEND: Drop (negligible improvement)")
    except Exception as e:
        log(f"  ERROR: {e}")

# =============================================================================
# Step 4: Final Model & Diagnostics
# =============================================================================
log("\n" + "=" * 70)
log("STEP 4: FINAL MODEL & DIAGNOSTICS")
log("=" * 70)

final_formula = current_formula
log(f"\nFinal formula: {final_formula[:100]}...")

final_model = run_ols(final_formula, train, "Final Model")
print_model_summary(final_model, "Final Model (In-Sample)")

# Full coefficient table
log("\n  Full Coefficient Table (non-FE terms):")
log(f"  {'Variable':<40} {'Coeff':>10} {'SE':>10} {'p-value':>10} {'Sig':>5}")
log(f"  {'-'*75}")
for var in final_model.params.index:
    if 'C(team)' in var or 'C(season_str)' in var:
        continue
    coef = final_model.params[var]
    se = final_model.bse[var]
    pval = final_model.pvalues[var]
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
    log(f"  {var:<40} {coef:>10.4f} {se:>10.4f} {pval:>10.4f} {sig:>5}")

# Holdout metrics
holdout = get_holdout_metrics(final_model, test)
if holdout:
    log(f"\n  Holdout (2024) Metrics:")
    log(f"    MAE:  {holdout['MAE']:.4f}")
    log(f"    MAPE: {holdout['MAPE']:.2f}%")
    log(f"    RMSE: {holdout['RMSE']:.4f}")
    log(f"    N:    {holdout['n']}")

# Breusch-Pagan test
try:
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(final_model.resid, final_model.model.exog)
    log(f"\n  Breusch-Pagan test: stat={bp_stat:.2f}, p={bp_pvalue:.4f}")
    if bp_pvalue < 0.05:
        log("  -> Heteroskedasticity detected (using HC3 robust SEs)")
    else:
        log("  -> No significant heteroskedasticity")
except Exception as e:
    log(f"  Breusch-Pagan test failed: {e}")

# VIF for non-categorical variables
log("\n  Variance Inflation Factors (non-FE variables):")
numeric_vars = [v for v in retained_vars if not v.startswith('C(') and v in train.columns
                and train[v].dtype in ('float64', 'int64', 'float32', 'int32')]
if numeric_vars:
    vif_data = train[numeric_vars].dropna()
    vif_data = sm.add_constant(vif_data)
    try:
        for i, col in enumerate(vif_data.columns):
            if col == 'const':
                continue
            vif = variance_inflation_factor(vif_data.values, i)
            log(f"    {col:<30} VIF = {vif:.2f}")
    except Exception as e:
        log(f"    VIF calculation error: {e}")

# Robustness check: capped attendance rate
log("\n  Robustness Check: Capped attendance_rate at 1.0")
train_capped = train.copy()
train_capped['attendance_rate'] = train_capped['attendance_rate'].clip(upper=1.0)
capped_model = run_ols(final_formula, train_capped, "Capped")
log(f"    Uncapped R2: {final_model.rsquared:.4f}")
log(f"    Capped R2:   {capped_model.rsquared:.4f}")
log(f"    Difference:  {final_model.rsquared - capped_model.rsquared:+.4f}")

# =============================================================================
# Predictive Examples
# =============================================================================
log("\n" + "=" * 70)
log("PREDICTIVE EXAMPLES")
log("=" * 70)

if holdout:
    pred_df = test.copy()
    pred_df['predicted'] = holdout['predictions']
    pred_df['actual'] = pred_df['attendance_rate']
    pred_df['error'] = pred_df['predicted'] - pred_df['actual']
    pred_df['abs_error'] = pred_df['error'].abs()

    # Best predictions
    log("\n  Closest Predictions (2024 holdout):")
    log(f"  {'Home':<20} {'Away':<20} {'Wk':>3} {'Actual':>8} {'Pred':>8} {'Error':>8}")
    log(f"  {'-'*67}")
    best = pred_df.nsmallest(10, 'abs_error')
    for _, r in best.iterrows():
        log(f"  {r['home_team']:<20} {r['away_team']:<20} {r['week']:>3} "
            f"{r['actual']:>8.3f} {r['predicted']:>8.3f} {r['error']:>+8.3f}")

    # Worst predictions
    log("\n  Worst Predictions (2024 holdout):")
    log(f"  {'Home':<20} {'Away':<20} {'Wk':>3} {'Actual':>8} {'Pred':>8} {'Error':>8}")
    log(f"  {'-'*67}")
    worst = pred_df.nlargest(10, 'abs_error')
    for _, r in worst.iterrows():
        log(f"  {r['home_team']:<20} {r['away_team']:<20} {r['week']:>3} "
            f"{r['actual']:>8.3f} {r['predicted']:>8.3f} {r['error']:>+8.3f}")

# Scenario analysis: record-based predictions
log("\n  Scenario Analysis: Effect of Season Record")
coef_sw = final_model.params.get('season_wins', 0)
coef_sl = final_model.params.get('season_losses', 0)
coef_psa = final_model.params.get('prior_season_att_rate', 0)
log(f"  Coefficient on season_wins:         {coef_sw:+.4f}")
log(f"  Coefficient on season_losses:       {coef_sl:+.4f}")
log(f"  Coefficient on prior_season_att_rate: {coef_psa:+.4f}")
log(f"\n  Marginal effect of one additional win:  {coef_sw:+.4f} ({coef_sw*100:+.1f} pp)")
log(f"  Marginal effect of one additional loss: {coef_sl:+.4f} ({coef_sl*100:+.1f} pp)")
log(f"  Asymmetry (|loss| - |win|):             {abs(coef_sl) - abs(coef_sw):+.4f} pp")

log(f"\n  Record scenarios at Week 11 (holding other vars at mean):")
log(f"  {'Record':<12} {'Wins':>5} {'Losses':>7} {'Win Effect':>12} {'Loss Effect':>13} {'Net vs 5-5':>12}")
log(f"  {'-'*62}")
for wins, losses in [(9, 0), (8, 1), (7, 2), (6, 3), (5, 4), (5, 5), (4, 5), (3, 6), (2, 7), (1, 8)]:
    # Net effect relative to a 5-5 baseline
    win_eff = coef_sw * (wins - 5)
    loss_eff = coef_sl * (losses - 5)
    net = win_eff + loss_eff
    log(f"  {wins}-{losses:<10} {wins:>5} {losses:>7} {win_eff:>+12.4f} {loss_eff:>+13.4f} {net:>+12.4f}")

log(f"\n  Prior season attendance rate effect:")
log(f"  If prior season att rate 0.98 -> 0.65 (FSU-type collapse):")
log(f"    Change: {coef_psa * (0.65 - 0.98):+.4f} ({coef_psa * (0.65 - 0.98) * 100:+.1f} pp)")
log(f"  If prior season att rate 0.70 -> 0.95 (program revival):")
log(f"    Change: {coef_psa * (0.95 - 0.70):+.4f} ({coef_psa * (0.95 - 0.70) * 100:+.1f} pp)")

# =============================================================================
# 2025 Forward Predictions
# =============================================================================
log("\n" + "=" * 70)
log("2025 FORWARD PREDICTIONS")
log("=" * 70)

games_2025 = df[df['season'] == 2025].copy()
if len(games_2025) > 0:
    try:
        # Map unseen levels for prediction
        known_seasons = [str(s) for s in TRAIN_SEASONS]
        known_teams = set(train['team'].unique())
        g25 = games_2025.copy()
        g25['season_str'] = g25['season_str'].apply(
            lambda x: x if x in known_seasons else known_seasons[-1])
        g25['team'] = g25['team'].apply(
            lambda x: x if x in known_teams else list(known_teams)[0])
        games_2025['predicted_rate'] = final_model.predict(g25)
        games_2025['predicted_attendance'] = (games_2025['predicted_rate'] *
                                              games_2025['stadium_capacity']).round(0)

        log(f"\n  2025 Predictions ({len(games_2025)} games):")
        log(f"  {'Home':<20} {'Away':<20} {'Wk':>3} {'Pred Rate':>10} {'Pred Att':>10}")
        log(f"  {'-'*67}")
        for _, r in games_2025.head(20).iterrows():
            log(f"  {r['home_team']:<20} {r['away_team']:<20} {r['week']:>3} "
                f"{r['predicted_rate']:>10.3f} {r['predicted_attendance']:>10.0f}")

        if len(games_2025) > 20:
            log(f"  ... ({len(games_2025) - 20} more games)")
    except Exception as e:
        log(f"  2025 prediction error: {e}")
else:
    log("  No 2025 games in dataset (season may not have started).")

# =============================================================================
# Save Model
# =============================================================================
model_path = os.path.join(OUTPUT_DIR, 'regression_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
log(f"\n  Saved model to {model_path}")

# =============================================================================
# Visualization 1: Model Diagnostics (multi-panel)
# =============================================================================
print("\n--- Generating diagnostic plots ---")

fig_diag, axes = plt.subplots(2, 2, figsize=(16, 14), facecolor=ECON_BG, dpi=150)
fig_diag.suptitle('Model Diagnostics', fontsize=18, fontweight='bold',
                  fontfamily=ECON_SERIF, color=ECON_TEXT, y=0.98)

# Panel 1: Residual vs Fitted
ax = axes[0, 0]
economist_ax(ax)
ax.grid(True, axis='y', color=ECON_LIGHT, linewidth=0.5)
ax.scatter(final_model.fittedvalues, final_model.resid, alpha=0.3, s=15,
           c=CONF_COLORS['SEC'], edgecolors='none')
ax.axhline(0, color=ECON_TEXT, linewidth=1, linestyle='--')
ax.set_xlabel('Fitted values', fontfamily=ECON_SANS, fontsize=10, color=ECON_TEXT)
ax.set_ylabel('Residuals', fontfamily=ECON_SANS, fontsize=10, color=ECON_TEXT)
ax.text(0, 1.05, 'Residuals vs Fitted', transform=ax.transAxes,
        fontsize=13, fontweight='bold', fontfamily=ECON_SERIF, color=ECON_TEXT)

# Panel 2: Q-Q Plot
ax = axes[0, 1]
economist_ax(ax)
ax.grid(True, axis='y', color=ECON_LIGHT, linewidth=0.5)
resid_standardized = (final_model.resid - final_model.resid.mean()) / final_model.resid.std()
theoretical_q = np.sort(stats.norm.ppf(np.linspace(0.001, 0.999, len(resid_standardized))))
sample_q = np.sort(resid_standardized.values)
ax.scatter(theoretical_q, sample_q, alpha=0.3, s=15, c=CONF_COLORS['Big 12'],
           edgecolors='none')
lims = [min(theoretical_q.min(), sample_q.min()), max(theoretical_q.max(), sample_q.max())]
ax.plot(lims, lims, '--', color=ECON_TEXT, linewidth=1)
ax.set_xlabel('Theoretical quantiles', fontfamily=ECON_SANS, fontsize=10, color=ECON_TEXT)
ax.set_ylabel('Sample quantiles', fontfamily=ECON_SANS, fontsize=10, color=ECON_TEXT)
ax.text(0, 1.05, 'Normal Q-Q Plot', transform=ax.transAxes,
        fontsize=13, fontweight='bold', fontfamily=ECON_SERIF, color=ECON_TEXT)

# Panel 3: Actual vs Predicted (holdout)
ax = axes[1, 0]
economist_ax(ax)
ax.grid(True, axis='y', color=ECON_LIGHT, linewidth=0.5)
if holdout:
    for conf in CONFERENCES:
        mask = test['conference'] == conf
        valid_mask = mask & test.index.isin(holdout['actuals'].index)
        if valid_mask.sum() > 0:
            ax.scatter(holdout['actuals'][valid_mask], holdout['predictions'][valid_mask],
                       c=CONF_COLORS[conf], label=conf, alpha=0.6, s=30, edgecolors='white',
                       linewidth=0.3)
    lims = [0.3, 1.3]
    ax.plot(lims, lims, '--', color=ECON_TEXT, linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(fontsize=8, frameon=False, prop={'family': ECON_SANS})
ax.set_xlabel('Actual attendance rate', fontfamily=ECON_SANS, fontsize=10, color=ECON_TEXT)
ax.set_ylabel('Predicted attendance rate', fontfamily=ECON_SANS, fontsize=10, color=ECON_TEXT)
ax.text(0, 1.05, 'Actual vs Predicted (2024)', transform=ax.transAxes,
        fontsize=13, fontweight='bold', fontfamily=ECON_SERIF, color=ECON_TEXT)

# Panel 4: Residuals by Team (box plot - top 15 by abs median residual)
ax = axes[1, 1]
economist_ax(ax)
if holdout:
    pred_df_plot = test.copy()
    pred_df_plot['resid'] = holdout['actuals'] - holdout['predictions']
    team_med = pred_df_plot.groupby('home_team')['resid'].median().abs().nlargest(15)
    top_teams = team_med.index.tolist()
    plot_data = pred_df_plot[pred_df_plot['home_team'].isin(top_teams)]

    team_order = (plot_data.groupby('home_team')['resid'].median()
                  .reindex(top_teams).sort_values().index.tolist())

    bp = ax.boxplot([plot_data[plot_data['home_team'] == t]['resid'].values
                     for t in team_order],
                    vert=False, patch_artist=True, widths=0.6)
    for patch in bp['boxes']:
        patch.set_facecolor(CONF_COLORS['ACC'])
        patch.set_alpha(0.5)
    for element in ['whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_color(ECON_TEXT)
    ax.set_yticklabels(team_order, fontsize=7, fontfamily=ECON_SANS)
    ax.axvline(0, color=ECON_TEXT, linewidth=1, linestyle='--')

ax.set_xlabel('Prediction residual', fontfamily=ECON_SANS, fontsize=10, color=ECON_TEXT)
ax.text(0, 1.05, 'Residuals by Team (2024)', transform=ax.transAxes,
        fontsize=13, fontweight='bold', fontfamily=ECON_SERIF, color=ECON_TEXT)

fig_diag.tight_layout(rect=[0, 0, 1, 0.95])
diag_path = os.path.join(OUTPUT_DIR, 'model_diagnostics.png')
fig_diag.savefig(diag_path, dpi=150, bbox_inches='tight', facecolor=ECON_BG)
plt.close(fig_diag)
log(f"  Saved {diag_path}")

# =============================================================================
# Visualization 2: Coefficient Plot
# =============================================================================
print("--- Generating coefficient plot ---")

# Get non-FE coefficients
coef_data = []
for var in final_model.params.index:
    if 'C(team)' in var or 'C(season_str)' in var or var == 'Intercept':
        continue
    coef_data.append({
        'variable': var,
        'coeff': final_model.params[var],
        'ci_low': final_model.conf_int().loc[var, 0],
        'ci_high': final_model.conf_int().loc[var, 1],
        'pvalue': final_model.pvalues[var],
    })

coef_df = pd.DataFrame(coef_data).sort_values('coeff')

fig_coef, ax = plt.subplots(figsize=(12, max(8, len(coef_df) * 0.4)),
                            facecolor=ECON_BG, dpi=150)
economist_ax(ax)

y_pos = range(len(coef_df))
colors = [CONF_COLORS['SEC'] if p < 0.05 else ECON_LIGHT
          for p in coef_df['pvalue']]

ax.barh(list(y_pos), coef_df['coeff'].values, color=colors, height=0.6, alpha=0.8)
ax.errorbar(coef_df['coeff'].values, list(y_pos),
            xerr=[coef_df['coeff'].values - coef_df['ci_low'].values,
                  coef_df['ci_high'].values - coef_df['coeff'].values],
            fmt='none', ecolor=ECON_TEXT, elinewidth=1, capsize=3)
ax.axvline(0, color=ECON_TEXT, linewidth=1, linestyle='-')
ax.set_yticks(list(y_pos))
ax.set_yticklabels(coef_df['variable'].values, fontsize=8, fontfamily=ECON_SANS)
ax.set_xlabel('Coefficient (95% CI)', fontfamily=ECON_SANS, fontsize=11, color=ECON_TEXT)
ax.text(0, 1.03, 'What drives attendance?', transform=ax.transAxes,
        fontsize=16, fontweight='bold', fontfamily=ECON_SERIF, color=ECON_TEXT)
ax.text(0, 1.01, 'Coefficient estimates with 95% confidence intervals (red = p < 0.05)',
        transform=ax.transAxes, fontsize=10, fontfamily=ECON_SANS, color='#666666')

fig_coef.tight_layout()
coef_path = os.path.join(OUTPUT_DIR, 'coefficient_plots.png')
fig_coef.savefig(coef_path, dpi=150, bbox_inches='tight', facecolor=ECON_BG)
plt.close(fig_coef)
log(f"  Saved {coef_path}")

# =============================================================================
# Visualization 3: Variable Importance
# =============================================================================
print("--- Generating variable importance chart ---")

fig_imp, axes_imp = plt.subplots(2, 2, figsize=(18, 16), facecolor=ECON_BG, dpi=150)
fig_imp.suptitle('Model Analysis', fontsize=18, fontweight='bold',
                 fontfamily=ECON_SERIF, color=ECON_TEXT, y=0.99)

# Panel 1: Variable importance (absolute t-stats)
ax = axes_imp[0, 0]
economist_ax(ax)

importance = []
for var in final_model.params.index:
    if 'C(team)' in var or 'C(season_str)' in var or var == 'Intercept':
        continue
    importance.append({
        'variable': var,
        'abs_t': abs(final_model.tvalues[var]),
    })

imp_df = pd.DataFrame(importance).sort_values('abs_t', ascending=True).tail(15)
bar_colors = [CONF_COLORS['Big 12'] if t > 1.96 else ECON_LIGHT for t in imp_df['abs_t']]
ax.barh(range(len(imp_df)), imp_df['abs_t'].values, color=bar_colors, height=0.6)
ax.set_yticks(range(len(imp_df)))
ax.set_yticklabels(imp_df['variable'].values, fontsize=8, fontfamily=ECON_SANS)
ax.axvline(1.96, color=ECON_TEXT, linewidth=1, linestyle='--', alpha=0.5)
ax.text(1.96, len(imp_df) - 0.5, 'p=0.05', fontsize=8, fontfamily=ECON_SANS, color='#666666')
ax.set_xlabel('|t-statistic|', fontfamily=ECON_SANS, fontsize=10, color=ECON_TEXT)
ax.text(0, 1.05, 'Variable Importance', transform=ax.transAxes,
        fontsize=13, fontweight='bold', fontfamily=ECON_SERIF, color=ECON_TEXT)
ax.text(0, 1.01, 'Top 15 by absolute t-statistic (blue = significant at 5%)',
        transform=ax.transAxes, fontsize=9, fontfamily=ECON_SANS, color='#666666')

# Panel 2: R-squared waterfall
ax = axes_imp[0, 1]
economist_ax(ax)

waterfall_data = [('Baseline', baseline_r2)]
cum_r2 = baseline_r2
for block_name, result in block_results.items():
    if result['kept']:
        delta = result['delta_r2']
        cum_r2 += delta
        waterfall_data.append((block_name.split(':')[0].strip(), cum_r2))

labels = [w[0] for w in waterfall_data]
values = [w[1] for w in waterfall_data]

bar_colors_wf = [CONF_COLORS['SEC']] + [CONF_COLORS['Big 12']] * (len(values) - 1)
ax.bar(range(len(values)), values, color=bar_colors_wf, alpha=0.8, width=0.6)
for i, (l, v) in enumerate(zip(labels, values)):
    ax.text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=9,
            fontfamily=ECON_SANS, color=ECON_TEXT, fontweight='bold')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9, fontfamily=ECON_SANS, rotation=30, ha='right')
ax.set_ylabel('R-squared', fontfamily=ECON_SANS, fontsize=10, color=ECON_TEXT)
ax.text(0, 1.05, 'Cumulative R-squared', transform=ax.transAxes,
        fontsize=13, fontweight='bold', fontfamily=ECON_SERIF, color=ECON_TEXT)
ax.text(0, 1.01, 'Block-by-block model improvement',
        transform=ax.transAxes, fontsize=9, fontfamily=ECON_SANS, color='#666666')

# Panel 3: Record scenario chart -- attendance effect by season record
ax = axes_imp[1, 0]
economist_ax(ax)
ax.grid(True, axis='y', color=ECON_LIGHT, linewidth=0.5)

coef_sw_plot = final_model.params.get('season_wins', 0)
coef_sl_plot = final_model.params.get('season_losses', 0)
base_att = train['attendance_rate'].mean()

records = [(w, 10 - w) for w in range(10, 0, -1)]  # 10-0 down to 1-9
labels = [f'{w}-{l}' for w, l in records]
effects = [coef_sw_plot * (w - 5) + coef_sl_plot * (l - 5) for w, l in records]
pred_rates = [base_att + e for e in effects]

bar_colors_rec = [CONF_COLORS['SEC'] if e >= 0 else CONF_COLORS['Big 12'] for e in effects]
ax.barh(range(len(records)), effects, color=bar_colors_rec, height=0.6, alpha=0.8)
ax.set_yticks(range(len(records)))
ax.set_yticklabels(labels, fontsize=9, fontfamily=ECON_SANS)
ax.axvline(0, color=ECON_TEXT, linewidth=1, linestyle='-')
ax.set_xlabel('Effect on attendance rate vs 5-5 baseline', fontfamily=ECON_SANS,
              fontsize=10, color=ECON_TEXT)
ax.text(0, 1.05, 'Season Record Effect', transform=ax.transAxes,
        fontsize=13, fontweight='bold', fontfamily=ECON_SERIF, color=ECON_TEXT)
ax.text(0, 1.01, 'Predicted attendance change by record (red=winning, blue=losing)',
        transform=ax.transAxes, fontsize=9, fontfamily=ECON_SANS, color='#666666')

# Panel 4: Conference comparison (prediction error box plots)
ax = axes_imp[1, 1]
economist_ax(ax)

if holdout:
    pred_df_conf = test.copy()
    pred_df_conf['error'] = holdout['actuals'] - holdout['predictions']

    conf_data = [pred_df_conf[pred_df_conf['conference'] == c]['error'].dropna().values
                 for c in CONFERENCES]

    bp = ax.boxplot(conf_data, labels=CONFERENCES, vert=True, patch_artist=True, widths=0.5)
    for patch, conf in zip(bp['boxes'], CONFERENCES):
        patch.set_facecolor(CONF_COLORS[conf])
        patch.set_alpha(0.5)
    for element in ['whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_color(ECON_TEXT)
    ax.axhline(0, color=ECON_TEXT, linewidth=1, linestyle='--')

ax.set_ylabel('Prediction error', fontfamily=ECON_SANS, fontsize=10, color=ECON_TEXT)
ax.text(0, 1.05, 'Prediction Error by Conference', transform=ax.transAxes,
        fontsize=13, fontweight='bold', fontfamily=ECON_SERIF, color=ECON_TEXT)
ax.text(0, 1.01, '2024 holdout: actual minus predicted',
        transform=ax.transAxes, fontsize=9, fontfamily=ECON_SANS, color='#666666')

fig_imp.tight_layout(rect=[0, 0, 1, 0.96])
imp_path = os.path.join(OUTPUT_DIR, 'variable_importance.png')
fig_imp.savefig(imp_path, dpi=150, bbox_inches='tight', facecolor=ECON_BG)
plt.close(fig_imp)
log(f"  Saved {imp_path}")

# =============================================================================
# Visualization 4: Actual vs Predicted Scatter (standalone)
# =============================================================================
print("--- Generating actual vs predicted scatter ---")

if holdout:
    fig_avp, ax = plt.subplots(figsize=(10, 10), facecolor=ECON_BG, dpi=150)
    economist_ax(ax)
    ax.grid(True, axis='y', color=ECON_LIGHT, linewidth=0.5)

    for conf in CONFERENCES:
        mask = test['conference'] == conf
        valid_mask = mask & test.index.isin(holdout['actuals'].index)
        if valid_mask.sum() > 0:
            ax.scatter(holdout['actuals'][valid_mask], holdout['predictions'][valid_mask],
                       c=CONF_COLORS[conf], label=conf, alpha=0.6, s=50, edgecolors='white',
                       linewidth=0.5)

    lims = [0.3, max(1.3, holdout['actuals'].max() * 1.05)]
    ax.plot(lims, lims, '--', color=ECON_TEXT, linewidth=1.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Actual attendance rate', fontfamily=ECON_SANS, fontsize=12, color=ECON_TEXT)
    ax.set_ylabel('Predicted attendance rate', fontfamily=ECON_SANS, fontsize=12, color=ECON_TEXT)
    ax.text(0, 1.06, 'How well does the model predict?', transform=ax.transAxes,
            fontsize=16, fontweight='bold', fontfamily=ECON_SERIF, color=ECON_TEXT)
    ax.text(0, 1.02, f'2024 holdout set  |  MAE={holdout["MAE"]:.3f}, RMSE={holdout["RMSE"]:.3f}',
            transform=ax.transAxes, fontsize=11, fontfamily=ECON_SANS, color='#666666')
    ax.legend(fontsize=10, frameon=False, prop={'family': ECON_SANS},
              loc='lower right')

    fig_avp.text(0.05, -0.01,
                 'Source: CollegeFootballData.com  |  HC3 robust standard errors',
                 fontsize=8, fontfamily=ECON_SANS, color='#999999')

    fig_avp.tight_layout()
    avp_path = os.path.join(OUTPUT_DIR, 'actual_vs_predicted.png')
    fig_avp.savefig(avp_path, dpi=150, bbox_inches='tight', facecolor=ECON_BG)
    plt.close(fig_avp)
    log(f"  Saved {avp_path}")

# =============================================================================
# Save Results File
# =============================================================================
with open(results_path, 'w') as f:
    f.write('\n'.join(results_lines))
log(f"\n  Saved {results_path}")

# =============================================================================
# Summary
# =============================================================================
log("\n" + "=" * 70)
log("SUMMARY")
log("=" * 70)
log(f"  Dataset: {len(df)} observations, {len(df['home_team'].unique())} teams, "
    f"{len(df['season'].unique())} seasons")
log(f"  Baseline R2:    {baseline_r2:.4f}")
log(f"  Final R2:       {final_model.rsquared:.4f}")
log(f"  Improvement:    {final_model.rsquared - baseline_r2:+.4f}")
if holdout:
    log(f"  Holdout MAE:    {holdout['MAE']:.4f}")
    log(f"  Holdout RMSE:   {holdout['RMSE']:.4f}")

log(f"\n  Artifacts saved:")
log(f"    - model_data.csv")
log(f"    - regression_model.pkl")
log(f"    - model_diagnostics.png")
log(f"    - coefficient_plots.png")
log(f"    - variable_importance.png")
log(f"    - actual_vs_predicted.png")
log(f"    - hierarchical_results.txt")

# Save final results
with open(results_path, 'w') as f:
    f.write('\n'.join(results_lines))

print("\nDone!")

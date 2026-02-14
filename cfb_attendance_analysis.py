"""
College Football Attendance Analysis - One-Sheeter
Analyzes attendance patterns for SEC, Big 12, ACC, Big Ten from 2022-present.
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from adjustText import adjust_text
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
import os
API_KEY = os.environ.get('CFB_API_KEY', '')
BASE_URL = 'https://api.collegefootballdata.com'
HEADERS = {'Authorization': f'Bearer {API_KEY}', 'Accept': 'application/json'}
CONFERENCES = ['SEC', 'Big 12', 'ACC', 'Big Ten']
# Last 10 seasons excluding COVID (2020)
SEASONS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
OUTPUT_PATH = r'C:\Users\tkasi\Documents\cfb_attendance\cfb_attendance_one_sheeter.png'

# Conference colors — Economist-style muted palette
CONF_COLORS = {
    'SEC': '#E3120B',     # Economist red
    'Big 12': '#006BA6',  # Economist blue
    'ACC': '#6A8A3E',     # olive/sage green
    'Big Ten': '#AD8E00',  # dark gold/ochre
}

# =============================================================================
# Data Extraction (with local cache to avoid repeated API calls)
# =============================================================================
import time
import json
import os

CACHE_DIR = r'C:\Users\tkasi\Documents\cfb_attendance\cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def fetch_json(endpoint, params=None):
    for attempt in range(5):
        resp = requests.get(f'{BASE_URL}{endpoint}', headers=HEADERS, params=params or {})
        if resp.status_code == 429:
            wait = 2 ** attempt
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
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

# 1. Get current teams and their conferences + stadium capacities
print("Fetching team data...")
fbs_teams = load_or_fetch('fbs_teams_2025', lambda: fetch_json('/teams/fbs', {'year': 2025}))
all_teams = []
for t in fbs_teams:
    conf = t.get('conference')
    if conf not in CONFERENCES:
        continue
    loc = t.get('location') or {}
    cap = loc.get('capacity')
    all_teams.append({
        'team_id': t['id'],
        'school': t['school'],
        'conference': conf,
        'stadium_capacity': cap,
        'color': t.get('color', '#888888'),
    })

teams_df = pd.DataFrame(all_teams)
team_names = set(teams_df['school'])
team_conf_map = dict(zip(teams_df['school'], teams_df['conference']))
team_capacity_map = dict(zip(teams_df['school'], teams_df['stadium_capacity']))
print(f"  Found {len(teams_df)} teams across {len(CONFERENCES)} conferences")

# 2. Get venue data for capacity lookups (some games are at non-home venues)
print("Fetching venue data...")
venues_raw = load_or_fetch('venues', lambda: fetch_json('/venues'))
venue_capacity = {v['id']: v['capacity'] for v in venues_raw if v.get('capacity')}
print(f"  Found {len(venue_capacity)} venues with capacity data")

# 3. Fetch all FBS games by YEAR (not by team/conference) to minimize API calls.
#    ~8 calls total instead of 500+. Filter to our teams client-side.
print("Fetching game data...")
def fetch_all_games():
    games = []
    for year in SEASONS:
        for season_type in ['regular', 'postseason']:
            try:
                result = fetch_json('/games', {
                    'year': year,
                    'seasonType': season_type,
                    'classification': 'fbs',
                })
                games.extend(result)
                print(f"  Fetched {len(result)} {season_type} games for {year}")
                time.sleep(0.5)
            except Exception as e:
                print(f"  Warning: Failed to fetch {year} {season_type}: {e}")
    return games

all_games = load_or_fetch('all_games', fetch_all_games)

# Deduplicate games by ID (a game can appear under both teams)
games_df = pd.DataFrame(all_games).drop_duplicates(subset='id')
print(f"  Found {len(games_df)} unique games")

# =============================================================================
# Data Processing
# =============================================================================
print("Processing data...")

# Filter to only games involving at least one of our target teams
games_df = games_df[
    (games_df['homeTeam'].isin(team_names)) | (games_df['awayTeam'].isin(team_names))
].copy()

# Drop games with no attendance data
games_df = games_df[games_df['attendance'].notna() & (games_df['attendance'] > 0)].copy()
print(f"  {len(games_df)} games with attendance data")

# Classify game type
def classify_game(row):
    if row['seasonType'] == 'postseason':
        return 'Bowl'
    elif row['neutralSite']:
        return 'Neutral'
    else:
        return 'Regular'

games_df['game_type'] = games_df.apply(classify_game, axis=1)

# Get venue capacity for each game (use venueId lookup)
def get_game_venue_capacity(row):
    vid = row.get('venueId')
    if vid and vid in venue_capacity:
        return venue_capacity[vid]
    return None

games_df['venue_capacity'] = games_df.apply(get_game_venue_capacity, axis=1)

# =============================================================================
# Chart 1: Scatter Plot - Home vs Away Attendance Rate
# =============================================================================
print("Building scatter plot data...")

# For scatter plot: exclude bowl games
scatter_games = games_df[games_df['game_type'] != 'Bowl'].copy()

# Home attendance rate per team
# When a team is the HOME team: attendance / home team's stadium capacity
home_records = []
for _, g in scatter_games.iterrows():
    home = g['homeTeam']
    if home in team_names:
        # Use the venue capacity for the game (covers neutral site cases too)
        # For home games, the home team's stadium capacity is most relevant
        if g['game_type'] == 'Regular':
            cap = team_capacity_map.get(home)
        else:
            # Neutral site: use venue capacity or home team capacity
            cap = g['venue_capacity'] or team_capacity_map.get(home)
        if cap and cap > 0:
            home_records.append({
                'school': home,
                'attendance': g['attendance'],
                'capacity': cap,
                'rate': g['attendance'] / cap,
            })

home_df = pd.DataFrame(home_records)
home_rates = home_df.groupby('school')['rate'].mean()

# Away attendance rate per team
# When a team is the AWAY team: attendance / host stadium capacity
away_records = []
for _, g in scatter_games.iterrows():
    away = g['awayTeam']
    if away in team_names:
        # Use venue capacity (the host's stadium)
        cap = g['venue_capacity']
        if not cap or cap <= 0:
            # Fall back to home team's stadium capacity
            cap = team_capacity_map.get(g['homeTeam'])
        if cap and cap > 0:
            away_records.append({
                'school': away,
                'attendance': g['attendance'],
                'capacity': cap,
                'rate': g['attendance'] / cap,
            })

away_df = pd.DataFrame(away_records)
away_rates = away_df.groupby('school')['rate'].mean()

# Combine
scatter_data = pd.DataFrame({
    'home_rate': home_rates,
    'away_rate': away_rates,
}).dropna()
scatter_data['conference'] = scatter_data.index.map(team_conf_map)
scatter_data = scatter_data.dropna(subset=['conference'])
print(f"  {len(scatter_data)} teams with both home and away data")

# =============================================================================
# Chart 2: Dumbbell Plot - Attendance Elasticity (Winning vs Losing Seasons)
# =============================================================================
print("Building dumbbell plot data...")

# Step A: Compute win% per team per season from all games (including bowls)
# Use the full games_df (which already has attendance > 0 filter, but we need
# all completed games for W/L). Re-fetch completed games from the deduplicated set.
all_games_deduped = pd.DataFrame(all_games).drop_duplicates(subset='id')
completed = all_games_deduped[all_games_deduped['completed'] == True].copy()

win_records = []
for _, g in completed.iterrows():
    home = g['homeTeam']
    away = g['awayTeam']
    hp = g.get('homePoints')
    ap = g.get('awayPoints')
    if hp is None or ap is None:
        continue
    season = g['season']
    if home in team_names:
        win_records.append({'school': home, 'season': season, 'win': 1 if hp > ap else 0})
    if away in team_names:
        win_records.append({'school': away, 'season': season, 'win': 1 if ap > hp else 0})

win_df = pd.DataFrame(win_records)
season_records = win_df.groupby(['school', 'season']).agg(
    wins=('win', 'sum'),
    games=('win', 'count'),
).reset_index()
season_records['win_pct'] = season_records['wins'] / season_records['games']

# Step B: Classify each season as winning (>0.600) or losing (<0.500)
# Seasons between 0.500 and 0.600 are excluded from the elasticity analysis
season_records['tier'] = season_records['win_pct'].apply(
    lambda x: 'winning' if x > 0.600 else ('losing' if x < 0.500 else 'middle')
)

# Step C: Compute HOME attendance rate per team per season
# Only home games (team is listed as homeTeam, exclude neutral site and bowls)
home_only_games = scatter_games[scatter_games['game_type'] == 'Regular'].copy()
att_records = []
for _, g in home_only_games.iterrows():
    home = g['homeTeam']
    att = g['attendance']
    season = g['season']

    if home in team_names:
        cap = team_capacity_map.get(home)
        if cap and cap > 0:
            att_records.append({'school': home, 'season': season, 'rate': att / cap})

att_df = pd.DataFrame(att_records)
season_att = att_df.groupby(['school', 'season'])['rate'].mean().reset_index()

# Step D: Normalize attendance rate by removing each team's linear trend.
# This prevents secular changes (stadium renovations, conference realignment,
# ticket policy shifts) from masquerading as a fair-weather signal.
merged = season_att.merge(season_records[['school', 'season', 'win_pct', 'tier']], on=['school', 'season'])

from numpy.polynomial import polynomial as P

# For each team, fit a linear trend (rate ~ year) and compute residuals.
# This removes secular attendance changes (renovations, realignment, etc.)
residuals = []
for school, grp in merged.groupby('school'):
    x = grp['season'].values.astype(float)
    y = grp['rate'].values
    if len(x) < 2:
        grp = grp.copy()
        grp['residual'] = 0.0
    else:
        coeffs = P.polyfit(x, y, 1)
        predicted = P.polyval(x, coeffs)
        grp = grp.copy()
        grp['residual'] = y - predicted
    residuals.append(grp)
merged = pd.concat(residuals)

# Average RESIDUAL during losing vs winning seasons
# Positive residual = attendance above that team's trend; negative = below trend
losing_resid = merged[merged['tier'] == 'losing'].groupby('school')['residual'].mean()
winning_resid = merged[merged['tier'] == 'winning'].groupby('school')['residual'].mean()

# Build dumbbell dataset — only keep teams with BOTH winning and losing seasons
dumbbell_data = pd.DataFrame(index=sorted(team_names))
dumbbell_data['losing_resid'] = dumbbell_data.index.map(losing_resid)
dumbbell_data['winning_resid'] = dumbbell_data.index.map(winning_resid)
dumbbell_data['conference'] = dumbbell_data.index.map(team_conf_map)
dumbbell_data = dumbbell_data.dropna(subset=['conference'])
dumbbell_data['diff'] = dumbbell_data['winning_resid'] - dumbbell_data['losing_resid']

dumbbell_qualified = dumbbell_data.dropna(subset=['losing_resid', 'winning_resid']).copy()
print(f"  {len(dumbbell_qualified)} teams with both winning (>60%) and losing (<50%) seasons")

# Top 10 most fair-weathered (largest spread)
top10_fairweather = dumbbell_qualified.nlargest(10, 'diff').sort_values('diff', ascending=True)

# Top 10 least fair-weathered (smallest POSITIVE spread — exclude negative deviations)
positive_only = dumbbell_qualified[dumbbell_qualified['diff'] > 0]
top10_loyal = positive_only.nsmallest(10, 'diff').sort_values('diff', ascending=False)

# =============================================================================
# Create One-Sheeter Figure
# =============================================================================
print("Generating one-sheeter...")

from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# =========================================================================
# Economist-style formatting
# =========================================================================
ECON_BG = '#F2E8D5'       # warm ivory background
ECON_TEXT = '#333333'      # near-black text
ECON_LIGHT = '#BBBBBB'    # light gray for gridlines/medians
ECON_SERIF = 'Georgia'    # Economist-like serif font
ECON_SANS = 'Franklin Gothic Medium'  # clean sans fallback

def economist_ax(ax):
    """Apply Economist-style formatting to an axes."""
    ax.set_facecolor(ECON_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', length=0, labelsize=9, colors=ECON_TEXT)
    ax.grid(True, axis='x', color=ECON_LIGHT, linewidth=0.5, linestyle='-')
    ax.grid(False, axis='y')
    ax.set_axisbelow(True)

fig = plt.figure(figsize=(22, 20), facecolor=ECON_BG, dpi=150)

# --- Economist-style subtitle only ---
fig.text(0.06, 0.97, 'SEC, Big 12, ACC and Big Ten conferences  |  2015-2024 seasons, excluding COVID 2020',
         fontsize=13, fontfamily=ECON_SANS, color='#666666',
         ha='left', va='top')

# Layout
gs = gridspec.GridSpec(2, 2, height_ratios=[1.1, 1], hspace=0.30, wspace=0.35,
                       left=0.06, right=0.96, top=0.89, bottom=0.05)

# =====================================================================
# Chart 1: Scatter Plot
# =====================================================================
ax1 = fig.add_subplot(gs[0, :])
economist_ax(ax1)
ax1.grid(True, axis='y', color=ECON_LIGHT, linewidth=0.5, linestyle='-')  # both axes for scatter

for conf in CONFERENCES:
    mask = scatter_data['conference'] == conf
    subset = scatter_data[mask]
    ax1.scatter(subset['home_rate'] * 100, subset['away_rate'] * 100,
                c=CONF_COLORS[conf], label=conf, s=110, alpha=0.9,
                edgecolors='white', linewidth=0.6, zorder=3)

texts = []
for school, row in scatter_data.iterrows():
    label = school
    if len(label) > 15:
        parts = label.split()
        if len(parts) > 1:
            label = parts[-1]
    texts.append(ax1.text(row['home_rate'] * 100, row['away_rate'] * 100, label,
                          fontsize=6, ha='center', va='center', fontfamily=ECON_SANS,
                          color=ECON_TEXT))

adjust_text(texts, ax=ax1, arrowprops=dict(arrowstyle='-', color='#aaaaaa', lw=0.4),
            force_text=(0.5, 0.5), max_move=50)

median_home = scatter_data['home_rate'].median() * 100
median_away = scatter_data['away_rate'].median() * 100
ax1.axvline(median_home, color='#333333', linestyle='--', linewidth=1.5, zorder=1)
ax1.axhline(median_away, color='#333333', linestyle='--', linewidth=1.5, zorder=1)

x_min, x_max = ax1.get_xlim()
y_min, y_max = ax1.get_ylim()
ax1.text(x_max - 1, y_max - 1, 'Elite Draw\n(High Home & Away)', ha='right', va='top',
         fontsize=10, fontweight='bold', color='#333333', fontstyle='italic', fontfamily=ECON_SANS)
ax1.text(x_min + 1, y_max - 1, 'Road Warriors\n(High Away, Low Home)', ha='left', va='top',
         fontsize=10, fontweight='bold', color='#333333', fontstyle='italic', fontfamily=ECON_SANS)
ax1.text(x_max - 1, y_min + 1, 'Home Strong\n(High Home, Low Away)', ha='right', va='bottom',
         fontsize=10, fontweight='bold', color='#333333', fontstyle='italic', fontfamily=ECON_SANS)
ax1.text(x_min + 1, y_min + 1, 'Low Draw\n(Low Home & Away)', ha='left', va='bottom',
         fontsize=10, fontweight='bold', color='#333333', fontstyle='italic', fontfamily=ECON_SANS)

ax1.set_xlabel('Home attendance rate, %', fontsize=11, fontfamily=ECON_SANS, color=ECON_TEXT, labelpad=8)
ax1.set_ylabel('Away attendance rate, %', fontsize=11, fontfamily=ECON_SANS, color=ECON_TEXT, labelpad=8)

# Left-aligned chart title
ax1.text(0, 1.06, 'Who fills the stadium?', transform=ax1.transAxes,
         fontsize=16, fontweight='bold', fontfamily=ECON_SERIF, color=ECON_TEXT, ha='left')
ax1.text(0, 1.02, 'Average attendance as a share of stadium capacity, home and away',
         transform=ax1.transAxes, fontsize=10, fontfamily=ECON_SANS, color='#666666', ha='left')

# Legend — no frame, Economist style
ax1.legend(loc='right', fontsize=9, frameon=False, bbox_to_anchor=(1.0, 0.35),
           labelcolor=ECON_TEXT, prop={'family': ECON_SANS})

# =====================================================================
# Dumbbell helpers — Economist style
# =====================================================================
def draw_dumbbell_right_aligned(ax, data, title, subtitle):
    economist_ax(ax)
    y_pos = np.arange(len(data))
    for i, (school, row) in enumerate(data.iterrows()):
        conf = row['conference']
        color = CONF_COLORS.get(conf, '#888888')
        diff_pp = row['diff'] * 100
        winning_x = 0
        losing_x = -diff_pp

        ax.plot([losing_x, winning_x], [i, i], color=color, linewidth=2.5, alpha=0.7, zorder=2)
        ax.scatter(losing_x, i, color=ECON_BG, edgecolors=color, linewidth=1.8, s=80, zorder=3)
        ax.scatter(winning_x, i, color=color, edgecolors='white', linewidth=0.8, s=80, zorder=3)
        ax.text(losing_x - 0.3, i, f'+{diff_pp:.1f}pp', va='center', ha='right',
                fontsize=8.5, color=ECON_TEXT, fontfamily=ECON_SANS)

    ax.axvline(0, color='#999999', linestyle='-', linewidth=0.8, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data.index, fontsize=9.5, fontfamily=ECON_SANS, color=ECON_TEXT)
    ax.set_xlabel('')  # self-evident from title
    ax.text(0, 1.08, title, transform=ax.transAxes,
            fontsize=13, fontweight='bold', fontfamily=ECON_SERIF, color=ECON_TEXT, ha='left')
    ax.text(0, 1.03, subtitle, transform=ax.transAxes,
            fontsize=9, fontfamily=ECON_SANS, color='#666666', ha='left')
    max_spread = max(abs(row['diff'] * 100) for _, row in data.iterrows())
    ax.set_xlim(-max_spread * 1.4, max_spread * 1.4)

def draw_dumbbell_left_aligned(ax, data, title, subtitle):
    economist_ax(ax)
    y_pos = np.arange(len(data))
    for i, (school, row) in enumerate(data.iterrows()):
        conf = row['conference']
        color = CONF_COLORS.get(conf, '#888888')
        diff_pp = row['diff'] * 100
        losing_x = 0
        winning_x = diff_pp

        ax.plot([losing_x, winning_x], [i, i], color=color, linewidth=2.5, alpha=0.7, zorder=2)
        ax.scatter(losing_x, i, color=ECON_BG, edgecolors=color, linewidth=1.8, s=80, zorder=3)
        ax.scatter(winning_x, i, color=color, edgecolors='white', linewidth=0.8, s=80, zorder=3)
        ax.text(winning_x + 0.3, i, f'+{diff_pp:.1f}pp', va='center', ha='left',
                fontsize=8.5, color=ECON_TEXT, fontfamily=ECON_SANS)

    ax.axvline(0, color='#999999', linestyle='-', linewidth=0.8, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data.index, fontsize=9.5, fontfamily=ECON_SANS, color=ECON_TEXT)
    ax.set_xlabel('')
    ax.text(0, 1.08, title, transform=ax.transAxes,
            fontsize=13, fontweight='bold', fontfamily=ECON_SERIF, color=ECON_TEXT, ha='left')
    ax.text(0, 1.03, subtitle, transform=ax.transAxes,
            fontsize=9, fontfamily=ECON_SANS, color='#666666', ha='left')
    max_spread = max(abs(row['diff'] * 100) for _, row in data.iterrows())
    ax.set_xlim(-max_spread * 1.4, max_spread * 1.4)

# --- Chart 2 (left): Most Fair-Weathered ---
ax2 = fig.add_subplot(gs[1, 0])
draw_dumbbell_right_aligned(ax2, top10_fairweather,
    'The bandwagon riders',
    'Largest home attendance swing between winning and losing seasons, pp')

# --- Chart 3 (right): Least Fair-Weathered ---
ax3 = fig.add_subplot(gs[1, 1])
draw_dumbbell_left_aligned(ax3, top10_loyal,
    'The diehards',
    'Smallest home attendance swing between winning and losing seasons, pp')

# --- Shared legend (no frame, Economist style) ---
legend_elements = [
    Line2D([0], [0], marker='o', color='white', markerfacecolor=ECON_BG,
           markeredgecolor='#555555', markeredgewidth=1.8, markersize=8,
           label='Losing season (<50% W)', linestyle='None'),
    Line2D([0], [0], marker='o', color='white', markerfacecolor='#555555',
           markeredgecolor='white', markeredgewidth=0.8, markersize=8,
           label='Winning season (>60% W)', linestyle='None'),
    Line2D([0], [0], color='#555555', linewidth=2.5, alpha=0.6,
           label='Fair-weather spread'),
]
for conf in CONFERENCES:
    legend_elements.append(
        Line2D([0], [0], marker='o', color='white', markerfacecolor=CONF_COLORS[conf],
               markersize=8, label=conf, linestyle='None')
    )
fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements),
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.005),
           prop={'family': ECON_SANS})

# --- Source line (Economist convention) ---
fig.text(0.06, -0.02, 'Source: CollegeFootballData.com  |  Trend-normalised to remove secular attendance changes',
         fontsize=8, fontfamily=ECON_SANS, color='#999999', ha='left', va='top')

plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor=ECON_BG)
print(f"\nSaved to {OUTPUT_PATH}")

# =============================================================================
# Key Insights
# =============================================================================
print("\n=== KEY INSIGHTS ===")

top_home = scatter_data['home_rate'].nlargest(5)
print("\nTop 5 Home Attendance Rates:")
for school, rate in top_home.items():
    print(f"  {school}: {rate*100:.1f}%")

top_away = scatter_data['away_rate'].nlargest(5)
print("\nTop 5 Away Attendance Rates:")
for school, rate in top_away.items():
    print(f"  {school}: {rate*100:.1f}%")

print(f"\nTop 10 Most Fair-Weathered:")
for school, row in top10_fairweather.iloc[::-1].iterrows():
    print(f"  {school}: {row['diff']*100:+.1f}pp spread")

print(f"\nTop 10 Least Fair-Weathered:")
for school, row in top10_loyal.iloc[::-1].iterrows():
    print(f"  {school}: {row['diff']*100:+.1f}pp spread")

print("\nDone!")

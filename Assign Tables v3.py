#!/usr/bin/env python3
"""
Fast community-based reassignment:
- Groups tables by overlapping clubs using community detection.
- For each community, assigns tables to the nearest available allowed grid positions via BFS.
- Output: table_allocations_grid_adj_gap_bfs.png
"""

from pathlib import Path
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque, defaultdict

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
EXCEL_PATH   = Path("Table_Allocations.xlsx")
SHEET_NAME   = "Sheet1"
WRAP_AT_CH   = 18
FONT_SIZE_PT = 6
CELL_EDGE_IN = 1.2
GAP_FRAC     = 0.15
OUTFILE      = Path("table_allocations_grid_adj_gap_bfs.png")

# ------------------------------------------------------------------
# 1. Load the spreadsheet
# ------------------------------------------------------------------
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=0)

# All allowed positions
allowed_positions = set(zip(df["Row"], df["Col"]))

# ------------------------------------------------------------------
# 2. Build Table-Club Graph for Community Detection
# ------------------------------------------------------------------
grouped = df.groupby(["Row", "Col"])
positions = list(grouped.groups.keys())
G = nx.Graph()
G.add_nodes_from(positions)
for i, a in enumerate(positions):
    clubs_a = set(grouped.get_group(a)["Club"])
    for j in range(i+1, len(positions)):
        b = positions[j]
        clubs_b = set(grouped.get_group(b)["Club"])
        if clubs_a & clubs_b:
            G.add_edge(a, b)

from networkx.algorithms.community import greedy_modularity_communities
communities = list(greedy_modularity_communities(G))
# Sort communities largest first
communities.sort(key=len, reverse=True)

# ------------------------------------------------------------------
# 3. Largest Contiguous Block Placement for Each Community
# ------------------------------------------------------------------

def bfs_pick_block(start, needed, unused):
    queue = deque([start])
    seen = {start}
    result = [start]
    while queue and len(result) < needed:
        node = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = node[0]+dr, node[1]+dc
            npos = (nr, nc)
            if npos in unused and npos not in seen:
                seen.add(npos)
                result.append(npos)
                queue.append(npos)
                if len(result) == needed:
                    break
    return result if len(result) == needed else None


def find_all_contiguous_blocks(unused, size):
    # Return all strictly N/S/E/W-connected blocks of `size` from unused positions
    blocks = []
    for start in unused:
        block = bfs_pick_block(start, size, unused)
        if block:
            blocks.append(block)
    return blocks

import random

MAX_ATTEMPTS = 500
best_assignment = None
fewest_split_clubs = None
best_split_clubs = []

def is_contiguous(poses):
    if not poses: return True
    seen = {poses[0]}
    queue = deque([poses[0]])
    pose_set = set(poses)
    while queue:
        node = queue.popleft()
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (node[0]+dr, node[1]+dc)
            if neighbor in pose_set and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == len(poses)

for attempt in range(MAX_ATTEMPTS):
    unused_positions = set(allowed_positions)
    community_assignment = dict()
    # Shuffle community order for new attempt (all communities every time)
    attempt_communities = communities.copy()
    random.shuffle(attempt_communities)
    for comm in attempt_communities:
        comm = list(comm)
        assigned = None
        all_blocks = find_all_contiguous_blocks(unused_positions, len(comm))
        if all_blocks:
            assigned = min(all_blocks, key=lambda block: sorted(block)[0])
        else:
            assigned = sorted(list(unused_positions))[:len(comm)]
        for orig_pos, tgt_pos in zip(comm, assigned):
            community_assignment[tgt_pos] = grouped.get_group(orig_pos)
        unused_positions -= set(assigned)
    # Build club_to_positions for this attempt
    club_to_positions = defaultdict(list)
    for (r, c), group in community_assignment.items():
        for club in set(group["Club"]):
            club_to_positions[club].append((r, c))
    split_clubs = [club for club, poses in club_to_positions.items() if len(poses) > 1 and not is_contiguous(poses)]
    # If no splits, keep this assignment and break
    if not split_clubs:
        best_assignment = dict(community_assignment)
        fewest_split_clubs = 0
        break
    # Otherwise, keep the best so far
    if fewest_split_clubs is None or len(split_clubs) < fewest_split_clubs:
        best_assignment = dict(community_assignment)
        best_split_clubs = list(split_clubs)
        fewest_split_clubs = len(split_clubs)

# Use the best_assignment found
community_assignment = best_assignment

if fewest_split_clubs:
    print(f"Could not find a fully contiguous solution after {MAX_ATTEMPTS} attempts. Closest has {fewest_split_clubs} split clubs: {best_split_clubs}")


# ------------------------------------------------------------------
# 4. Build output grid for plotting
# ------------------------------------------------------------------
n_rows, n_cols = df["Row"].max() + 1, df["Col"].max() + 1
grid_rows, grid_cols = 2 * n_rows - 1, 2 * n_cols - 1
grid = pd.DataFrame("", index=range(grid_rows), columns=range(grid_cols))

# Club label
WRAP_LABEL = lambda group: textwrap.fill(", ".join(f"{c} ({int(m)})" for c, m in zip(group["Club"], group["Members"])), width=WRAP_AT_CH)

for (r, c), group in community_assignment.items():
    grid.iat[r * 2, c * 2] = WRAP_LABEL(group)

# ------------------------------------------------------------------
# 5. Compute cell & gap sizes so data cells stay square
# ------------------------------------------------------------------
denom_rows = n_rows + (n_rows - 1) * GAP_FRAC
denom_cols = n_cols + (n_cols - 1) * GAP_FRAC
N          = max(denom_rows, denom_cols)
unit       = 1.0 / N
cell_size  = unit
gap_size   = unit * GAP_FRAC

# ------------------------------------------------------------------
# 6. Draw the table with gaps
# ------------------------------------------------------------------
fig_side = CELL_EDGE_IN * N
fig, ax  = plt.subplots(figsize=(fig_side, fig_side), dpi=300)
ax.set_aspect("equal"); ax.axis("off")

tbl = ax.table(cellText=grid.values,
               cellLoc="center",
               loc="center")

tbl.auto_set_font_size(False)
tbl.set_fontsize(FONT_SIZE_PT)

# Adjust each cell’s width/height; hide spacer-cell borders & text
for (r, c), cell in tbl.get_celld().items():
    is_data_row = (r % 2 == 0)
    is_data_col = (c % 2 == 0)

    cell.set_height(cell_size if is_data_row else gap_size)
    cell.set_width(cell_size  if is_data_col  else gap_size)

    if not (is_data_row and is_data_col):       # spacer cell
        cell.set_linewidth(0)
        cell.set_facecolor("white")
        cell.get_text().set_text("")

# ------------------------------------------------------------------
# 7. Save PNG
# ------------------------------------------------------------------
# --------------------------------------------------------------
# 7. Draw lines: for each club, connect each adjacent pair of club tables with one colored line
# --------------------------------------------------------------
import matplotlib.cm as cm

club_to_positions = defaultdict(list)
for (r, c), group in community_assignment.items():
    for club in set(group["Club"]):
        club_to_positions[club].append((r, c))

# ------------------------------------------------------------------
# Check for split clubs and print warnings
# ------------------------------------------------------------------
def is_contiguous(poses):
    if not poses: return True
    seen = {poses[0]}
    queue = deque([poses[0]])
    pose_set = set(poses)
    while queue:
        node = queue.popleft()
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (node[0]+dr, node[1]+dc)
            if neighbor in pose_set and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == len(poses)

def cell_center(row, col, cell_size, gap_size):
    # Flip Y so row 0 is at the TOP, matching matplotlib's table layout
    y = 1.0 - ((cell_size + gap_size) * row / 2 + cell_size / 2)
    x = (cell_size + gap_size) * col / 2 + cell_size / 2
    return x, y

for club, poses in club_to_positions.items():
    if len(poses) > 1 and not is_contiguous(poses):
        print(f"WARNING: Club '{club}' assigned to non-adjacent tables: {poses}")

import matplotlib.cm as cm

cmap = cm.get_cmap('tab20')
club_names = sorted(club_to_positions)
club_colors = {club: cmap(i % 20) for i, club in enumerate(club_names)}

for club, poses in club_to_positions.items():
    pose_set = set(poses)
    for (r, c) in poses:
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (r + dr, c + dc)
            if neighbor in pose_set and (r, c) < neighbor:  # only once per pair
                p0 = cell_center(r * 2, c * 2, cell_size, gap_size)
                p1 = cell_center(neighbor[0] * 2, neighbor[1] * 2, cell_size, gap_size)
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=2.8, color=club_colors[club], alpha=0.8, zorder=1)

# Uncomment for a legend:
# handles = [plt.Line2D([0], [0], color=club_colors[c], lw=2, label=c) for c in club_names if len(club_to_positions[c]) > 1]
# if handles:
#     ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=5, title='Clubs')

plt.savefig(OUTFILE, bbox_inches="tight")
plt.close(fig)





print(f"\u2713 Community BFS chart saved to: {OUTFILE.resolve()}")

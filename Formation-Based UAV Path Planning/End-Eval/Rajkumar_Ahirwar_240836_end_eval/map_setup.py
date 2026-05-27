"""
map_setup.py
------------
Defines the 2-D grid environment: obstacle, start, and goal.
All other scripts import constants from here.
"""

import numpy as np

# ── Grid ──────────────────────────────────────────────────────────────────────
GRID_SIZE = 100          # 100 × 100 unit grid

# ── Key coordinates ───────────────────────────────────────────────────────────
START = (5, 50)
GOAL  = (95, 50)

# ── Circular obstacle ─────────────────────────────────────────────────────────
OBSTACLE_CENTER = (50, 50)
OBSTACLE_RADIUS = 10
SAFETY_MARGIN   = 4      # extra inflation radius used by the planner


def get_obstacle_mask(grid_size=GRID_SIZE,
                      center=OBSTACLE_CENTER,
                      radius=OBSTACLE_RADIUS,
                      margin=SAFETY_MARGIN):
    """
    Return a boolean array (shape: grid_size × grid_size) where
    True  = cell is blocked (obstacle body + safety inflation).
    Indexed as mask[row=y, col=x].
    """
    xs = np.arange(grid_size, dtype=float)
    ys = np.arange(grid_size, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    cx, cy  = center
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    return dist_sq <= (radius + margin) ** 2


def get_true_obstacle_mask(grid_size=GRID_SIZE,
                            center=OBSTACLE_CENTER,
                            radius=OBSTACLE_RADIUS):
    """Obstacle body only (no safety margin) — used for visualisation."""
    xs = np.arange(grid_size, dtype=float)
    ys = np.arange(grid_size, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    cx, cy  = center
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    return dist_sq <= radius ** 2
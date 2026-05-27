"""
formation.py
------------
Defines the letter 'R' formation using 12 UAVs.

How formation flying works
--------------------------
  1. The planner computes a single path for the formation centroid.
  2. Each UAV's world position = centroid position + its fixed offset.
  3. Offsets are constant throughout the flight  →  shape is preserved.
"""

import numpy as np

# ── Raw letter-R shape (before centring) ─────────────────────────────────────
#
#   Y-axis ↑
#   4 ●──●──●
#     |     |
#   2 ●     ●
#     |     |
#   0 ●─────●          ← middle crossbar (spine ↔ bump base)
#     |  ↗
#  -2 ●
#     |     ↘
#  -4 ●        ●
#      ↑              ↑
#   x=-3            x=+2
#
#   Drone indices:
#    0: [-3,-4]   spine bottom
#    1: [-3,-2]   spine lower
#    2: [-3, 0]   spine middle
#    3: [-3, 2]   spine upper
#    4: [-3, 4]   spine top
#    5: [-1, 4]   top-bar left
#    6: [ 1, 4]   top-bar right
#    7: [ 2, 3]   bump arc upper
#    8: [ 2, 1]   bump arc lower
#    9: [ 0, 0]   bump base / R junction
#   10: [ 0,-2]   leg upper
#   11: [ 2,-4]   leg lower tip

_raw_offsets = np.array([
    [-3, -4],   # 0  — spine bottom
    [-3, -2],   # 1  — spine lower
    [-3,  0],   # 2  — spine middle
    [-3,  2],   # 3  — spine upper
    [-3,  4],   # 4  — spine top
    [-1,  4],   # 5  — top crossbar left
    [ 1,  4],   # 6  — top crossbar right
    [ 2,  3],   # 7  — right bump upper arc
    [ 2,  1],   # 8  — right bump lower arc
    [ 0,  0],   # 9  — bump base (R junction)
    [ 0, -2],   # 10 — diagonal leg upper
    [ 2, -4],   # 11 — diagonal leg tip
], dtype=float)

# Centre so centroid lies at (0, 0)
FORMATION_OFFSETS = _raw_offsets - _raw_offsets.mean(axis=0)

N_UAVS = len(FORMATION_OFFSETS)   # 12

# Edges used to draw the letter R in the animation
FORMATION_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),   # left vertical spine
    (4, 5), (5, 6),                     # top horizontal crossbar
    (6, 7), (7, 8), (8, 9),             # right bump arc
    (2, 9),                             # middle horizontal bar (spine→bump)
    (9, 10), (10, 11),                  # diagonal leg
]


# ── Public API ────────────────────────────────────────────────────────────────

def get_drone_positions(centroid_x: float, centroid_y: float,
                         scale: float = 1.0) -> np.ndarray:
    """
    World positions for all N_UAVS drones given the centroid location.

    Parameters
    ----------
    centroid_x, centroid_y : centroid world coordinates
    scale                  : stretch / shrink the formation (default 1.0)

    Returns
    -------
    positions : ndarray, shape (N_UAVS, 2)  — [x, y] per drone
    """
    offsets   = FORMATION_OFFSETS * scale
    positions = offsets + np.array([centroid_x, centroid_y])
    return positions


# ── Quick visualisation ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    scale = 3.0
    pos   = get_drone_positions(0.0, 0.0, scale=scale)

    fig, ax = plt.subplots(figsize=(5, 8))
    for i, j in FORMATION_EDGES:
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                "royalblue", lw=2, alpha=0.7)
    ax.scatter(pos[:, 0], pos[:, 1], c="royalblue", s=140, zorder=5)
    for k, (px, py) in enumerate(pos):
        ax.annotate(str(k), (px, py), textcoords="offset points",
                    xytext=(6, 5), fontsize=9, color="navy")

    ax.set_aspect("equal")
    ax.set_title(f"Letter 'R' Formation — {N_UAVS} UAVs", fontsize=13)
    ax.set_xlabel("X offset");  ax.set_ylabel("Y offset")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"Formation offsets (centred):\n{FORMATION_OFFSETS}")
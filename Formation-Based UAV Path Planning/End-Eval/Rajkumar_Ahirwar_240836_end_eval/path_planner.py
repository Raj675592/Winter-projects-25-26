"""
path_planner.py
---------------
Implements A* on a 2-D grid with 8-connectivity, then applies a
greedy line-of-sight shortcutting pass to reduce waypoint count while
keeping the path obstacle-free.
"""

import heapq
import numpy as np

from map_setup import START, GOAL, GRID_SIZE, get_obstacle_mask


# ── Heuristic ─────────────────────────────────────────────────────────────────

def _heuristic(node, goal):
    """Euclidean distance — admissible and consistent on a 2-D grid."""
    return np.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)


# ── A* ────────────────────────────────────────────────────────────────────────

def astar(start=START, goal=GOAL, obstacle_mask=None):
    """
    A* pathfinding on a grid.

    Parameters
    ----------
    start, goal    : (x, y) integer tuples
    obstacle_mask  : 2-D bool array (mask[y, x] == True → blocked)

    Returns
    -------
    list of (x, y) tuples from start to goal (inclusive)
    """
    if obstacle_mask is None:
        obstacle_mask = get_obstacle_mask()

    grid_h, grid_w = obstacle_mask.shape

    open_set = []
    heapq.heappush(open_set, (0.0, start))

    came_from = {}
    g_score   = {start: 0.0}
    closed    = set()

    # 8-connected movement: cardinal + diagonal
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),           ( 0, 1),
        ( 1, -1), ( 1, 0),  ( 1, 1),
    ]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            return _reconstruct(came_from, goal)

        cx, cy = current
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < grid_w and 0 <= ny < grid_h):
                continue
            if obstacle_mask[ny, nx]:
                continue
            neighbor = (nx, ny)
            if neighbor in closed:
                continue

            move_cost      = np.sqrt(dx ** 2 + dy ** 2)
            tentative_g    = g_score[current] + move_cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor]   = tentative_g
                f                   = tentative_g + _heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))

    raise RuntimeError(
        "A* could not find a path. "
        "Check that start/goal are not inside the obstacle."
    )


def _reconstruct(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    return path[::-1]


# ── Path simplification ───────────────────────────────────────────────────────

def _line_of_sight(p1, p2, obstacle_mask):
    """
    True if the straight line from p1 → p2 passes only through free cells.
    Uses Bresenham-style rasterisation via np.linspace.
    """
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    x2, y2 = int(round(p2[0])), int(round(p2[1]))
    n  = max(abs(x2 - x1), abs(y2 - y1)) + 1
    xs = np.round(np.linspace(x1, x2, n)).astype(int)
    ys = np.round(np.linspace(y1, y2, n)).astype(int)
    h, w = obstacle_mask.shape
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    return not np.any(obstacle_mask[ys, xs])


def simplify_path(path, obstacle_mask):
    """
    Greedy line-of-sight shortcutting.
    From each waypoint, jump as far forward as line-of-sight allows.
    """
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if _line_of_sight(path[i], path[j], obstacle_mask):
                break
            j -= 1
        simplified.append(path[j])
        i = j
    return simplified


# ── Public API ────────────────────────────────────────────────────────────────

def get_path():
    """Run A* then simplify. Returns (N, 2) float array of waypoints."""
    mask      = get_obstacle_mask()
    raw_path  = astar(obstacle_mask=mask)
    simplified = simplify_path(raw_path, mask)
    return np.array(simplified, dtype=float)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from map_setup import OBSTACLE_CENTER, OBSTACLE_RADIUS

    waypoints = get_path()
    print(f"Simplified path: {len(waypoints)} waypoints")

    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    circle = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS,
                         color='tomato', alpha=0.45, label='Obstacle')
    ax.add_patch(circle)
    ax.plot(waypoints[:, 0], waypoints[:, 1],
            'b-o', markersize=6, linewidth=2, label='A* Path')
    ax.plot(*START, 'go', markersize=14, label='Start', zorder=5)
    ax.plot(*GOAL,  'r*', markersize=14, label='Goal',  zorder=5)
    ax.set_xlim(0, GRID_SIZE); ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect('equal')
    ax.set_title("A* Planned Path", fontsize=14)
    ax.set_xlabel("X (units)"); ax.set_ylabel("Y (units)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "path_plot.png"), dpi=150)
    print("Saved results/path_plot.png")
    plt.show()
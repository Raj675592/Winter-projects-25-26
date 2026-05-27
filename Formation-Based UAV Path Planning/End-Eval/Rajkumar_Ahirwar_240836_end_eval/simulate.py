"""
simulate.py
-----------
Entry point for the Formation-Based UAV Path Planning simulation.

Run with:
    python simulate.py

Outputs saved to results/
    path_plot.png              — A* path over the obstacle map
    trajectory_comparison.png  — speed & acceleration profiles
    formation_animation.gif    — 12 UAVs flying in letter-'R' formation
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                    # save files without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

from map_setup   import (START, GOAL, GRID_SIZE,
                          OBSTACLE_CENTER, OBSTACLE_RADIUS,
                          get_true_obstacle_mask)
from path_planner import get_path
from trajectory   import get_trajectories
from formation    import (get_drone_positions, N_UAVS,
                           FORMATION_EDGES, FORMATION_OFFSETS)

# ── Configuration ─────────────────────────────────────────────────────────────
RESULTS_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FORMATION_SCALE   = 2.5    # visual scale of the R formation
N_FRAMES          = 100    # animation frames per trajectory
FPS               = 18     # GIF frame rate


# ── Helpers ───────────────────────────────────────────────────────────────────

def _draw_obstacle(ax, alpha=0.45):
    circle = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS,
                        color="tomato", alpha=alpha, zorder=2, label="Obstacle")
    ax.add_patch(circle)
    ax.annotate("Obstacle", xy=OBSTACLE_CENTER,
                ha="center", va="center", fontsize=8,
                color="darkred", fontweight="bold", zorder=4)


def _frame_indices(traj, n_frames):
    """Evenly spaced frame indices into a trajectory dict."""
    return np.linspace(0, len(traj["t"]) - 1, n_frames, dtype=int)


# ── Plot 1 — path_plot.png ────────────────────────────────────────────────────

def plot_path(waypoints):
    fig, ax = plt.subplots(figsize=(8, 8))

    _draw_obstacle(ax)

    ax.plot(waypoints[:, 0], waypoints[:, 1],
            "b-o", markersize=6, linewidth=2.5, label="A* Waypoints", zorder=5)
    ax.plot(*START, "go", markersize=14, label="Start",  zorder=6)
    ax.plot(*GOAL,  "r*", markersize=16, label="Goal",   zorder=6)

    ax.set_xlim(0, GRID_SIZE);  ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect("equal")
    ax.set_title("A* Planned Path — Formation Centroid", fontsize=14)
    ax.set_xlabel("X (units)");  ax.set_ylabel("Y (units)")
    ax.legend(fontsize=11);  ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "path_plot.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓  Saved {out}")


# ── Plot 2 — trajectory_comparison.png ───────────────────────────────────────

def plot_trajectory_comparison(tmt, tme):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Speed
    axes[0].plot(tmt["t"], tmt["speed_profile"],
                 color="crimson",   lw=2,
                 label=f"Min-Time   (T={tmt['total_time']:.1f}s)")
    axes[0].plot(tme["t"], tme["speed_profile"],
                 color="steelblue", lw=2,
                 label=f"Min-Energy (T={tme['total_time']:.1f}s)")
    axes[0].set_title("Speed vs Time", fontsize=13)
    axes[0].set_xlabel("Time (s)");  axes[0].set_ylabel("Speed (units/s)")
    axes[0].legend();  axes[0].grid(True, alpha=0.3)

    # Acceleration
    axes[1].plot(tmt["t"], tmt["accel_profile"],
                 color="crimson",   lw=2,
                 label=f"Min-Time   (E={tmt['energy']:.1f})")
    axes[1].plot(tme["t"], tme["accel_profile"],
                 color="steelblue", lw=2,
                 label=f"Min-Energy (E={tme['energy']:.1f})")
    axes[1].set_title("Acceleration vs Time", fontsize=13)
    axes[1].set_xlabel("Time (s)");  axes[1].set_ylabel("Acceleration (units/s²)")
    axes[1].legend();  axes[1].grid(True, alpha=0.3)

    plt.suptitle("Trajectory Comparison: Min-Time vs Min-Energy", fontsize=15)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "trajectory_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓  Saved {out}")


# ── Animation — formation_animation.gif ──────────────────────────────────────

def animate_formation(tmt, tme, waypoints):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor("#f4f4f4")

    panel_cfg = [
        ("Min-Time Trajectory",   tmt,  "crimson",   _frame_indices(tmt, N_FRAMES)),
        ("Min-Energy Trajectory", tme, "steelblue", _frame_indices(tme, N_FRAMES)),
    ]

    # ── per-panel setup ──
    all_artists = []
    for ax, (title, traj, color, fidx) in zip(axes, panel_cfg):
        ax.set_facecolor("#f9f9f9")
        ax.set_xlim(0, GRID_SIZE);  ax.set_ylim(0, GRID_SIZE)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("X (units)");  ax.set_ylabel("Y (units)")
        ax.grid(True, alpha=0.25)

        _draw_obstacle(ax)

        ax.plot(*START, "go", markersize=10, zorder=5)
        ax.plot(*GOAL,  "r*", markersize=12, zorder=5)
        ax.annotate("S", START, fontsize=8, color="green",
                    ha="center", va="bottom", zorder=6)
        ax.annotate("G", GOAL,  fontsize=8, color="red",
                    ha="center", va="bottom", zorder=6)

        # Faint centroid path
        ax.plot(traj["x"], traj["y"], "-", color=color,
                alpha=0.15, lw=1.5, zorder=2)

        # Centroid trail (grows each frame)
        trail_line, = ax.plot([], [], "-", color=color,
                              alpha=0.45, lw=1.2, zorder=3)

        # Drone scatter
        scat = ax.scatter([], [], c=color, s=55, zorder=7,
                          edgecolors="white", linewidths=0.6)

        # Formation edge lines
        edge_lines = []
        for _ in FORMATION_EDGES:
            ln, = ax.plot([], [], "-", color=color, lw=1.3,
                          alpha=0.75, zorder=6)
            edge_lines.append(ln)

        # Centroid marker
        c_dot, = ax.plot([], [], "k+", ms=9, zorder=8)

        # Time label
        time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                            fontsize=9, va="top",
                            bbox=dict(boxstyle="round,pad=0.3",
                                      fc="white", alpha=0.7))

        all_artists.append(dict(
            scat=scat, edge_lines=edge_lines, c_dot=c_dot,
            trail=trail_line, time_text=time_text,
            cx=traj["x"][fidx], cy=traj["y"][fidx],
            times=traj["t"][fidx],
        ))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle("Formation UAV Simulation — Letter 'R'  (12 Drones)",
                 fontsize=15, fontweight="bold")

    def update(frame):
        updated = []
        for p in all_artists:
            cx_f = p["cx"][frame]
            cy_f = p["cy"][frame]

            # Drone positions
            pos = get_drone_positions(cx_f, cy_f, scale=FORMATION_SCALE)
            p["scat"].set_offsets(pos)

            # Formation edges
            for k, (i, j) in enumerate(FORMATION_EDGES):
                p["edge_lines"][k].set_data(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                )

            # Centroid dot
            p["c_dot"].set_data([cx_f], [cy_f])

            # Centroid trail (up to current frame)
            p["trail"].set_data(p["cx"][:frame + 1], p["cy"][:frame + 1])

            # Time annotation
            p["time_text"].set_text(f"t = {p['times'][frame]:.1f} s")

            updated.extend(
                [p["scat"], p["c_dot"], p["trail"], p["time_text"]]
                + p["edge_lines"]
            )
        return updated

    anim = FuncAnimation(fig, update, frames=N_FRAMES,
                          interval=1000 // FPS, blit=True)

    out = os.path.join(RESULTS_DIR, "formation_animation.gif")
    print("  ⏳ Saving animation … (may take ~30 s)")
    anim.save(out, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"  ✓  Saved {out}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(tmt, tme):
    print()
    print("=" * 58)
    print(f"{'Metric':<30} {'Min-Time':>12} {'Min-Energy':>12}")
    print("-" * 58)
    print(f"{'Total distance (units)':<30} "
          f"{tmt['total_dist']:>12.1f} {tme['total_dist']:>12.1f}")
    print(f"{'Total flight time (s)':<30} "
          f"{tmt['total_time']:>12.1f} {tme['total_time']:>12.1f}")
    print(f"{'Energy (∫|a|² dt)':<30} "
          f"{tmt['energy']:>12.2f} {tme['energy']:>12.2f}")
    print("=" * 58)
    speedup      = tme["total_time"] / tmt["total_time"]
    energy_ratio = tmt["energy"]     / tme["energy"]
    print(f"  Min-Time is {speedup:.1f}× faster than Min-Energy.")
    print(f"  Min-Time uses {energy_ratio:.1f}× more energy than Min-Energy.")
    print("=" * 58)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n[1/5]  Running A* path planner …")
    waypoints = get_path()
    print(f"       Found path: {len(waypoints)} waypoints after simplification")

    print("[2/5]  Generating trajectories …")
    tmt, tme = get_trajectories(waypoints)
    print(f"       Min-Time  : {tmt['total_time']:.1f} s")
    print(f"       Min-Energy: {tme['total_time']:.1f} s")

    print("[3/5]  Saving path plot …")
    plot_path(waypoints)

    print("[4/5]  Saving trajectory comparison …")
    plot_trajectory_comparison(tmt, tme)

    print("[5/5]  Creating formation animation …")
    animate_formation(tmt, tme, waypoints)

    print_summary(tmt, tme)
    print("All outputs saved to:  results/")
    print("  • path_plot.png")
    print("  • trajectory_comparison.png")
    print("  • formation_animation.gif\n")


if __name__ == "__main__":
    main()
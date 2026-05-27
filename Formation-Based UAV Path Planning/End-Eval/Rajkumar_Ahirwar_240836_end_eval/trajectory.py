"""
trajectory.py
-------------
Converts an ordered list of 2-D waypoints into two smooth trajectories:

  • Min-time   — high constant speed; shorter duration, sharper cornering,
                 higher acceleration demands.
  • Min-energy — low constant speed; longer duration, gradual acceleration,
                 significantly lower energy expenditure.

Both use scipy CubicSpline for C2-continuous interpolation.
Energy is estimated as the integral of |acceleration|² over time.
"""

import numpy as np
from scipy.interpolate import CubicSpline

# ── Speed parameters ──────────────────────────────────────────────────────────
MIN_TIME_SPEED   = 20.0   # units / second  — fast, prioritises arrival time
MIN_ENERGY_SPEED =  5.0   # units / second  — slow, minimises acceleration load

N_SAMPLES = 600           # points in the output trajectory arrays


# ── Core function ─────────────────────────────────────────────────────────────

def waypoints_to_trajectory(waypoints, speed, n_samples=N_SAMPLES):
    """
    Build a smooth, time-stamped trajectory from waypoints at constant speed.

    Parameters
    ----------
    waypoints : array-like, shape (N, 2)
    speed     : float — constant arc-length speed (units / s)
    n_samples : int   — resolution of the output arrays

    Returns
    -------
    dict with keys
        t             — 1-D array of timestamps  (s)
        x, y          — position arrays          (units)
        speed_profile — scalar speed vs time     (units/s)
        accel_profile — scalar |acceleration|    (units/s²)
        total_time    — flight duration          (s)
        total_dist    — path arc length          (units)
        energy        — ∫|a|² dt                (energy proxy)
    """
    wp = np.asarray(waypoints, dtype=float)

    # Arc-length parameterisation
    deltas      = np.diff(wp, axis=0)
    seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))
    cum_dist    = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_dist  = cum_dist[-1]

    # Time stamps at each waypoint (constant speed → t = dist / v)
    t_wp       = cum_dist / speed
    total_time = t_wp[-1]

    # Fit cubic splines (C2-continuous)
    cs_x = CubicSpline(t_wp, wp[:, 0])
    cs_y = CubicSpline(t_wp, wp[:, 1])

    # Sample uniformly in time
    t  = np.linspace(0.0, total_time, n_samples)
    x  = cs_x(t)
    y  = cs_y(t)

    vx = cs_x(t, 1);  vy = cs_y(t, 1)   # velocity components
    ax = cs_x(t, 2);  ay = cs_y(t, 2)   # acceleration components

    speed_profile = np.sqrt(vx ** 2 + vy ** 2)
    accel_profile = np.sqrt(ax ** 2 + ay ** 2)

    # Energy proxy: integral of squared acceleration magnitude
    energy = float(np.trapezoid(accel_profile ** 2, t))

    return {
        "t":             t,
        "x":             x,
        "y":             y,
        "speed_profile": speed_profile,
        "accel_profile": accel_profile,
        "total_time":    total_time,
        "total_dist":    total_dist,
        "energy":        energy,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def get_trajectories(waypoints):
    """
    Return (traj_min_time, traj_min_energy) for the given waypoint sequence.
    """
    traj_mt = waypoints_to_trajectory(waypoints, speed=MIN_TIME_SPEED)
    traj_me = waypoints_to_trajectory(waypoints, speed=MIN_ENERGY_SPEED)
    return traj_mt, traj_me


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from path_planner import get_path

    os.makedirs("results", exist_ok=True)
    wp = get_path()
    tmt, tme = get_trajectories(wp)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(tmt["t"], tmt["speed_profile"],
                 color="crimson",   lw=2, label=f"Min-Time  ({tmt['total_time']:.1f} s)")
    axes[0].plot(tme["t"], tme["speed_profile"],
                 color="steelblue", lw=2, label=f"Min-Energy ({tme['total_time']:.1f} s)")
    axes[0].set_title("Speed vs Time"); axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed (units/s)"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(tmt["t"], tmt["accel_profile"],
                 color="crimson",   lw=2, label=f"Min-Time  (E={tmt['energy']:.1f})")
    axes[1].plot(tme["t"], tme["accel_profile"],
                 color="steelblue", lw=2, label=f"Min-Energy (E={tme['energy']:.1f})")
    axes[1].set_title("Acceleration vs Time"); axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Acceleration (units/s²)"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("Trajectory Comparison: Min-Time vs Min-Energy", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "trajectory_comparison.png"), dpi=150)
    print("Saved results/trajectory_comparison.png")
    plt.show()

    for label, t in [("Min-Time", tmt), ("Min-Energy", tme)]:
        print(f"{label:12} | time={t['total_time']:6.1f}s | "
              f"dist={t['total_dist']:6.1f} | energy={t['energy']:8.2f}")
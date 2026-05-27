# Formation-Based UAV Path Planning

**End-Term Project — UAV Simulation**

---

## Part 1 — What did you build?

This project simulates **12 UAVs** flying in formation across a 100 × 100 unit 2-D grid, from a start point to a goal point while maintaining the shape of the **letter 'R'** . The formation centroid's collision-free route is computed using the **A\*** algorithm\* (with Euclidean heuristic and 8-connectivity). The raw waypoints are then compressed via greedy line-of-sight shortcutting, smoothed with cubic splines, and converted into two distinct trajectories — **minimum-time** and **minimum-energy** — whose profiles are animated and compared.

---

## Part 2 — Setup

```bash
git clone https://github.com/Raj675592/Winter-projects-25-26.git
cd "Winter-projects-25-26\Formation-Based UAV Path Planning\End-Eval\Rajkumar_Ahirwar_240836_end_eval"
pip install -r requirements.txt
```

**Dependencies** (auto-installed by the command above):

| Package        | Purpose                      |
| -------------- | ---------------------------- |
| `numpy`      | Array maths, grid operations |
| `scipy`      | Cubic spline interpolation   |
| `matplotlib` | Plotting and animation       |
| `pillow`     | GIF encoding                 |

---

## Part 3 — How to run

```bash
python simulate.py
```

The script runs in five sequential steps and prints progress to the terminal:

```
[1/5]  Running A* path planner …
[2/5]  Generating trajectories …
[3/5]  Saving path plot …
[4/5]  Saving trajectory comparison …
[5/5]  Creating formation animation …
```

No interactive window opens. All outputs are saved directly to the `results/` folder. A summary table is printed at the end showing flight time, distance, and energy for both trajectories.

---

## Part 4 — What each script does

| Script              | Role                                                                                                                                                                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `map_setup.py`    | Defines the 100 × 100 grid, places the circular obstacle at (50, 50) with radius 10, and exposes `START`,`GOAL`, and the inflated obstacle mask used by the planner                                           |
| `path_planner.py` | Implements A\* with 8-connected movement and Euclidean heuristic; follows up with greedy line-of-sight shortcutting to minimise waypoint count while preserving obstacle avoidance                                 |
| `trajectory.py`   | Converts the simplified waypoint list into smooth cubic-spline trajectories; generates min-time (20 u/s) and min-energy (5 u/s) versions with time, position, speed, acceleration, and energy arrays               |
| `formation.py`    | Defines the 12-drone letter-'R' offsets (centred at the origin) and the edge connectivity used to draw the formation; provides `get_drone_positions()`to map a centroid position to all 12 drone world positions |
| `simulate.py`     | Orchestrates the full pipeline — calls each module, saves `path_plot.png`,`trajectory_comparison.png`, and `formation_animation.gif`, and prints the performance summary                                    |

---

## Part 5 — Results

### Path Plot

![Path Plot](https://claude.ai/chat/results/path_plot.png)

### Trajectory Comparison

![Trajectory Comparison](https://claude.ai/chat/results/trajectory_comparison.png)

### Observations

| Metric                 | Min-Time | Min-Energy |
| ---------------------- | -------- | ---------- |
| Total distance (units) | 95.0     | 95.0       |
| Total flight time (s)  | 4.7      | 19.0       |
| Energy — ∫           | a        | ² dt      |

- **Min-Time** is **4.0 × faster** (4.7 s vs 19.0 s) because it uses a constant speed of 20 u/s.
- **Min-Energy** uses **64 × less energy** (3.26 vs 208.42) because the lower speed of 5 u/s requires far smaller accelerations when cornering. The acceleration profile in the right subplot visually confirms this: min-time shows sharp spikes at the path bends while min-energy remains nearly flat.

---

## Part 6 — Formation details

| Property        | Value                                   |
| --------------- | --------------------------------------- |
| Formation shape | Letter**'R'**                           |
| Number of UAVs  | **12**                            |
| Formation scale | 2.5 × (offsets scaled before plotting) |

**How drones are assigned to positions:**

The 12 drones are assigned fixed index positions that together trace the letter R:

- Drones **0–4** : left vertical spine
- Drones **5–6** : top horizontal crossbar
- Drones **7–9** : right bump arc
- Drones **10–11** : diagonal leg

Each drone's world position at time _t_ is computed as:

```
position_i(t) = centroid(t) + offset_i × scale
```

The offsets are fixed constants (centred so their mean is zero), so the shape is perfectly maintained throughout the entire flight regardless of the centroid trajectory.

**Edge connections used for visualisation:**

```python
FORMATION_EDGES = [
    (0,1),(1,2),(2,3),(3,4),   # spine
    (4,5),(5,6),               # top crossbar
    (6,7),(7,8),(8,9),         # right bump arc
    (2,9),                     # middle horizontal bar
    (9,10),(10,11),            # diagonal leg
]
```

---

## Project Structure

```
end_term/
├── README.md
├── requirements.txt
├── map_setup.py
├── path_planner.py
├── trajectory.py
├── formation.py
├── simulate.py
└── results/
    ├── path_plot.png
    ├── trajectory_comparison.png
    └── formation_animation.gif
```

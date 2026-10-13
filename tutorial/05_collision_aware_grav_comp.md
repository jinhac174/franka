# Collision-aware gravity compensation

Gravity compensation ([03](03_grav_comp.md)) lets the arm float — including into a wall. Here we add a **slow planner** that, when links get close to obstacles, gently pushes them away, while the **fast loop** keeps the arm floating.

---

## 1. Run & LCM communication

```bash
python run_env.py --config configs/franka_fr3/env/collision.yaml

python run_controller.py --config configs/franka_fr3/controllers/col_aware_grav_comp.yaml

python run_visualizer.py --config configs/visualizer/franka_fr3/collision.yaml
```

| Key | Action |
|-----|--------|
| `p` / `q` | Pause / quit |

### What you should observe

Far from walls, behavior feels like plain grav-comp. Drag the hand toward the floor or a wall in the MuJoCo viewer: once a link gets within about 10 cm, the controller prints `State changed from grav_comp to repulsive` and the arm **backs off** instead of sinking through the planning geometry. The push fades out smoothly as the link leaves the warning band, so the arm settles near its edge. In Viser, tick **Show collision geometry** and **Show collision distances** to see the shapes and the nearest-point lines.

### Two rates (important mental model)

| Loop | Typical rate | Responsibility |
|------|--------------|----------------|
| **Plan** | 100 Hz | Measure distances with FCL; solve a small QP for a desired joint velocity $\dot q_{\mathrm{des}}$ |
| **Control** | 1 kHz | Track that $\dot q_{\mathrm{des}}$ with PD + damping |

Planning is heavier (geometry + OSQP). Control must stay fast and boring. On the real FR3 this split matters even more: the robot stops itself if commands arrive late.

### LCM

| Channel | Direction | Role |
|---------|-----------|------|
| `franka_arm_joint_meas`, `franka_gripper_joint_meas` | env → ctrl | State |
| `franka_robot_col_info` | env → ctrl / viz | Robot shapes ([04](04_publish_collision_geom.md)) |
| `franka_static_col_info` | env → ctrl / viz | Walls / floor |
| `franka_arm_joint_ctrl` | ctrl → env | Command |

---

## 2. Ideas

### Step A — Where am I relative to the wall?

For each monitored link–obstacle pair, FCL returns a distance $d$ and nearest points. Define a **margin residual**

$$
m = d - d_{\mathrm{safe}}.
$$

- $m \gg 0$: comfortably far.
- $0 \le m < d_{\mathrm{margin}}$: inside a soft “warning band” → start repelling.
- $m < 0$: penetrating the safety margin → treat as collision / stop.

The outward unit direction $\hat n$ points from the obstacle toward the robot point.

### Step B — Decide a mode

| Situation | Mode | Intuition |
|-----------|------|-----------|
| Any $m<0$ | Stop | Do not command motion that digs deeper |
| All $m \ge d_{\mathrm{margin}}$ | Pure grav-comp | Nothing to avoid |
| Otherwise | Repulsive QP | Ask joints to move the threatened points along $\hat n$ |

### Step C — A soft repulsive “velocity field”

Inside the warning band, scale a desired Cartesian escape speed by how deep you are:

$$
w = \alpha\Bigl(\frac{m}{d_{\mathrm{margin}}}-1\Bigr)^{2},
\qquad
v^\star = w\,\hat n.
$$

Far edge of the band → $w\approx 0$. Near violation → larger $w$.

### Step D — Turn Cartesian wishes into joint rates (QP)

Each threatened point has a Jacobian $J$ mapping $\dot q$ to point velocity. We ask:

$$
\min_{\dot q}
  \sum \|J\dot q - v^\star\|^2
  + \varepsilon\|\dot q\|^2
$$

with joint speed and position limits and linearized “do not decrease distance too fast” constraints. The solution is the plan’s $\dot q_{\mathrm{des}}$.

The model has 9 joints (7 arm + 2 fingers), but the fingers belong to the gripper, not to this controller. Their velocities are fixed to $0$ in the QP, and only the 7 arm joints are commanded.

### Step E — Fast tracking

Every millisecond:

$$
q_{\mathrm{des}} = q + \dot q_{\mathrm{des}}\,\Delta t,
\qquad
\tau_{\mathrm{ff}} = - b\,\dot q_{\mathrm{arm}},
$$

and the PD gains track the latest $\dot q_{\mathrm{des}}$ from the plan mailbox. In the grav-comp state the gains are zero, so the arm floats. As in [03](03_grav_comp.md), gravity itself is cancelled by the FR3 (`plant_gravity_comp: true`), not by $\tau_{\mathrm{ff}}$.

---

## 3. How it is implemented

| Piece | Role |
|-------|------|
| `controller/CollisionAwareGravCompControl.py` | Plan thread (FCL + OSQP) + 1 kHz update |
| `env/MujocoColEnv.py` | Publishes ColInfo |
| `configs/franka_fr3/controllers/col_aware_grav_comp.yaml` | `safty_dist_thr`, `magin_thr`, `repulsive_coeff`, `plan_freq`, `col_link_names`, `col_pairs`, `num_ctrl_joints`, `plant_gravity_comp` |

**Note:** some YAML keys are intentionally spelled `safty` / `magin` (historical). Match the file, do not “fix” them silently.

**Tuning intuition:** larger `repulsive_coeff` → stronger escape; larger `safty_dist_thr` / `magin_thr` → earlier reaction. Keep in mind that the FR3 joints have real friction (up to about 1 Nm): a very weak push may not move the arm at all.

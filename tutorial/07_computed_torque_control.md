# Computed torque–style point-to-point motion

So far we either floated ([03](03_grav_comp.md)) or chased a live Cartesian target ([06](06_inverse_kinematics_quadratic_program.md)). Here we do classical **go-to-a-joint-goal** motion: sample a safe goal, **plan a path**, then **track** it with PD + friction feedforward (plus gravity $g(q)$ from the controller's URDF, [03](03_grav_comp.md)).

“Computed torque” in the soft sense: we cancel known dynamics terms in $\tau_{\mathrm{ff}}$ and let PD clean up the rest.

---

## 1. Run & LCM communication

```bash
python run_env.py --config configs/franka_fr3/env/collision.yaml

python run_controller.py --config configs/franka_fr3/controllers/min_jerk_p2p.yaml

python run_visualizer.py --config configs/visualizer/franka_fr3/min_jerk_p2p.yaml
```

| Key | Meaning (suggested workflow) |
|-----|------------------------------|
| `1` | Sample a new **collision-free** joint goal |
| `2` | Build a smooth plan to that goal |
| `3` | **Track** the plan |
| `g` | Fall back to gravity compensation |
| `h` | Hold |
| `p`/`q` | Pause / quit |

### What you should observe

Type the keys in the **controller** terminal. After `1`, a goal appears in Viser. After `2`, the TCP path is drawn. After `3`, the arm follows it smoothly toward the goal: moves of 2 rad take a little over 2 s, the tracking error stays below about 0.01 rad, and the arm ends within a fraction of a milliradian of the goal. If sampling or planning fails, try `1` again (the goal might be hard to reach).

### LCM

| Channel | Direction | Role |
|---------|-----------|------|
| Joint meas + ColInfo | env → ctrl | State & geometry |
| `franka_arm_joint_ctrl` | ctrl → env | Tracking torques / setpoints (7 arm joints) |
| `franka_p2p_target_joint` | ctrl → viz | Sampled goal |
| `franka_p2p_joint_traj` | ctrl → viz | Planned trajectory |
| `franka_p2p_status` | ctrl → viz | Mode, errors, gains |
| `franka_p2p_gains_cmd` | viz → ctrl | Live gain tweaks from the GUI |

Control runs at 1 kHz; status overlays update more slowly.

---

## 2. Ideas

### Tracking law

Along a reference $(q_d(t),\dot q_d(t))$:

$$
\tau
  = K_p(q_d-q)
  + K_d(\dot q_d-\dot q)
  + \underbrace{F_{\mathrm{jc}}\,\mathrm{sat}(\dot q_d/\phi)}_{\tau_{\mathrm{ff}}}
  \;+\;g(q).
$$

- PD pulls the error to zero.
- $g(q)$ comes from the controller's URDF; the plant adds none (`plant_gravity_comp: false`, [03](03_grav_comp.md)).
- Coulomb compensation overcomes the FR3’s joint friction (about 1.1 Nm on joints 1–4) when starting and stopping. Toggle it in Viser and compare the tracking error.

The gains are libfranka’s joint-impedance example values,

$$
K_p = \mathrm{diag}(600,600,600,600,250,150,50),\qquad
K_d = \mathrm{diag}(50,50,50,20,20,20,10),
$$

the usual safe starting point on the real FR3. As a sanity check against the inertia $m_i$ each joint sees (diagonal of $M(q)$ at the ready pose), critical damping would be $k_{d,i} = 2\sqrt{k_{p,i} m_i}$. Relative to that, these gains are damped at about $1.2,\ 0.75,\ 0.9,\ 0.37,\ 1.9,\ 2.3,\ 2.6$ times critical: the light wrist joints 5–7 are overdamped, and joint 4 (the elbow) is the most underdamped. All of them are comfortably stable at 1 kHz.

### Planning pipeline (story form)

1. **Sample** a joint vector that is far enough from walls and self-collisions (FCL).
2. **Connect** start to goal with RRT-Connect in joint space.
3. **Shortcut** useless zig-zags.
4. **Smooth** with a B-spline (or a densified polyline fallback).
5. **Time** the path with a rest-to-rest **minimum-jerk** profile so $\dot q_d$ starts and ends at zero.

The controller then only has to track; it does not rethink the whole path every millisecond.

---

## 3. How it is implemented

| Piece | Role |
|-------|------|
| `controller/MinimumJerkP2PControl.py` | Keys `1`/`2`/`3`, planner calls, PD tracking |
| `utils/planning/` (RRT, B-spline, min-jerk helpers) | Path construction |
| `configs/franka_fr3/controllers/min_jerk_p2p.yaml` | Gains, friction, RRT parameters, clearance, collision and self-collision pairs, timing |

**Self-collision pairs** come from Franka’s SRDF: it lists every link pair that is adjacent or can never touch; the remaining pairs (forearm/wrist/hand against the base and shoulder) are the ones worth checking. Franka’s capsules deliberately overlap at the joints, so checking adjacent pairs would report “collision” everywhere.

**Exercise:** raise `kd` with Viser while tracking and feel the motion get heavier; lower clearance and see planning fail more often.

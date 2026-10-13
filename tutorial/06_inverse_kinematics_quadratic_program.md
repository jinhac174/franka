# Inverse kinematics as a quadratic program

Goal: move a **point on the robot** (here the gripper’s tool centre point, `fr3_hand_tcp`, between the fingertips) toward a target you drive with the keyboard — while **not** driving through walls.

This is the first tutorial where “desired Cartesian velocity” becomes a **joint velocity** through an optimizer (OSQP).

---

## 1. Run & LCM communication

```bash
python run_env.py --config configs/franka_fr3/env/collision.yaml

python run_controller.py --config configs/franka_fr3/controllers/ik_qp.yaml

python run_visualizer.py --config configs/visualizer/franka_fr3/ik_qp.yaml
```

| Key | Action |
|-----|--------|
| `w`/`s` | Move target along $\pm x$ |
| `a`/`d` | Move target along $\pm y$ |
| `r`/`f` | Move target along $\pm z$ |
| `[`/`]` | Smaller / larger step |
| `h` | Go back to the joint-space home (Franka ready pose) |
| `p`/`q` | Pause / quit |

### What you should observe

A target sphere moves when you press keys (type them in the **controller** terminal). The gripper’s TCP should chase it, settling within about 5 mm (`pos_tol`). Near walls, motion should bend away rather than ignore geometry. If the QP cannot find a safe velocity, the tip may briefly freeze ($\dot q=0$ that plan) instead of slamming through.

### LCM

| Channel | Direction | Role |
|---------|-----------|------|
| Joint measurements | env → ctrl | State |
| ColInfo (robot + static) | env → ctrl | FCL shapes |
| `franka_arm_joint_ctrl` | ctrl → env | Tracking command (7 arm joints) |
| `franka_ik_target_pose` | ctrl → viz | Target position, current tip, tip velocity (for drawing) |

Planning runs slower than control; control tracks the latest $\dot q_{\mathrm{des}}$.

---

## 2. Ideas

### From joints to a point in space

Pick a link and a fixed offset $\Delta$ in that link’s frame (the “task point”). Forward kinematics gives

$$
p(q) = p_{\mathrm{link}}(q) + R_{\mathrm{link}}(q)\,\Delta.
$$

Differentiating yields a **point Jacobian**:

$$
\dot p = J_p(q)\,\dot q.
$$

If you know a desired $\dot p$, any $\dot q$ that approximately satisfies this equation is a candidate inverse-kinematics solution **at the velocity level**.

### A simple “go toward the target” field

Let $p_{\mathrm{des}}$ be the keyboard target. A contracting field is

$$
v^\star
  = \mathrm{saturate}\bigl(
      K_v(p_{\mathrm{des}}-p) - D_v\dot p,\; v_{\max}
    \bigr).
$$

Far away → move at about $v_{\max}$. Nearby → slow down. Damping on $\dot p$ reduces overshoot.

### The QP (heart of the tutorial)

We do **not** invert $J_p$ alone (that ignores limits and walls). We solve

$$
\min_{\dot q}
  \underbrace{\|J_p\dot q - v^\star\|^2}_{\text{track the field}}
  + \underbrace{\tfrac{\varepsilon}{2}\|\dot q\|^2}_{\text{prefer small joint rates}}
$$

subject to:

1. **Joint speed limits** and a short-horizon box so joints cannot race through their range.
2. **Collision inequalities** from FCL: for each watched pair with distance $d$ and normal $\hat n$,

$$
\hat n^\top J_{\mathrm{pt}}\,\dot q
  \ge
  \text{(clipped escape demand)}.
$$

Intuition: the component of motion that **closes** the gap must not be too large; if already too close, demand a bit of **opening** motion.

### Failure without drama

If OSQP reports infeasible / unsolved, command $\dot q=0$ for that plan and try again next cycle. **Never** send OSQP’s huge “certificate” vector to the motors — that is a common source of sudden violent motion.

### Tracking $\dot q_{\mathrm{des}}$ on a real arm: friction

The 1 kHz loop turns the plan into torque with a damper on the velocity error plus feedforward:

$$
\tau = K_d(\dot q_{\mathrm{des}}-\dot q)
      + C(q,\dot q)\dot q
      + F_c\,\mathrm{sat}\!\left(\dot q_{\mathrm{des}}/\phi\right).
$$

- No $g(q)$: the FR3 cancels gravity itself ([03](03_grav_comp.md)).
- $C(q,\dot q)\dot q$: Coriolis/centrifugal torques, which the FR3 does **not** cancel.
- $F_c\,\mathrm{sat}(\cdot)$: Coulomb friction compensation. The FR3 joints have about 1.1 Nm of friction on joints 1–4. Near a goal the planner asks for tiny velocities; a small damper torque cannot overcome that friction and the arm **stalls** just short of the target. The feedforward pushes with the full friction torque in the commanded direction as soon as $|\dot q_{\mathrm{des}}| > \phi$.

**Try:** set `friction_phi: 0.03` or halve `Fjc`, press `h`, and watch joints 2 and 6 stall a few hundredths of a radian from home. On the real robot the friction values must be identified again — the MuJoCo numbers are only a starting point.

The model has 9 joints (arm + fingers). The finger velocities are fixed to 0 in the QP; only the 7 arm joints are commanded.

---

## 3. How it is implemented

| Piece | Role |
|-------|------|
| `controller/IKQPControl.py` | Keyboard target, field, FCL constraints, OSQP, tracking |
| `utils/lie/kinematics.py` | `get_point_Jacobian` |
| `configs/franka_fr3/controllers/ik_qp.yaml` | `task_link`, `task_point_local`, gains, `Fjc` / `friction_phi`, margins, `col_pairs` |

**Learning path:** first chase the target in free space (`h` home if lost), then intentionally drive toward a wall and watch the path bend. That bend is the constraint set speaking.

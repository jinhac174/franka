# Joint-space vs SE(3) impedance

Impedance control means: behave like a **spring–damper** relative to a nominal pose — compliant when pushed, restoring when released. This tutorial compares two ways to define that spring:

1. **Per joint** (easy to implement, configuration-dependent in Cartesian space),
2. **On the gripper’s SE(3) pose** at the tool centre point `fr3_hand_tcp` (geometric, natural for “hold this hand frame”).

---

## 1. Run & LCM communication

```bash
python run_env.py --config configs/franka_fr3/env/default.yaml

python run_controller.py --config configs/franka_fr3/controllers/impedance.yaml

python run_visualizer.py --config configs/visualizer/franka_fr3/impedance.yaml
```

| Key | Mode |
|-----|------|
| `j` | Joint-space impedance |
| `t` | Task-space (TCP SE(3)) impedance |
| `p`/`q` | Pause / quit |

### What you should observe

Disturb the arm in MuJoCo (drag a link). In both modes it should push back toward the nominal posture / TCP pose. Switch `j` ↔ `t` (keys in the controller terminal; also `--mode task` on the command line) and compare:

- **Joint mode** brings every joint back: from a 0.3 rad offset the arm is home within a fraction of a milliradian in about 2 s.
- **Task mode** brings the **gripper frame** back (from 17 cm to under 1 cm in a few seconds, then slowly — friction against a soft 150 N/m spring), but the joints do **not** all return. The FR3 has 7 joints for a 6-D pose, so one “elbow” motion leaves the TCP fixed; the SE(3) spring cannot see it. Drag the elbow in task mode and it stays where you leave it.

Viser shows the nominal and current frames and lets you scale gains live.

### LCM

| Channel | Direction | Role |
|---------|-----------|------|
| Joint measurements | env → ctrl | $q,\dot q$ |
| `franka_arm_joint_ctrl` | ctrl → env | Pure $\tau_{\mathrm{ff}}$ (PD gains zero at the plant interface), 7 arm joints |
| `franka_imp_status` | ctrl → viz | Errors / mode |
| `franka_imp_gains_cmd` | viz → ctrl | Gain scales from the GUI |

---

## 2. Ideas

### Shared setup

Let $q_{\mathrm{nom}}$ be a comfortable default posture and

$$
T_{\mathrm{nom}} = \mathrm{FK}_{\mathrm{tcp}}(q_{\mathrm{nom}})
$$

the corresponding TCP pose in $\mathrm{SE}(3)$ (position + orientation). Here $q_{\mathrm{nom}}$ is Franka’s ready pose.

### Joint impedance

Treat each joint as its own spring–damper:

$$
\tau
  = K_q(q_{\mathrm{nom}}-q)
  - D_q\dot q
  \;\;(+\;g(q)\text{ added by the FR3}).
$$

The gains are libfranka’s joint-impedance example values, $K_q=\mathrm{diag}(600,600,600,600,250,150,50)$, $D_q=\mathrm{diag}(50,50,50,20,20,20,10)$.

**Pros:** simple, stable if gains are modest.  
**Cons:** the Cartesian stiffness at the gripper **changes with configuration** — the same $K_q$ does not mean “100 N/m in $x$.”

### SE(3) impedance (body frame)

Work with the TCP pose $T(q)$ and the **body** (LOCAL) twist $V_b = J_b(q)\dot q$.

Measure pose error on the group with the matrix logarithm:

$$
\xi = \log\bigl(T(q)^{-1} T_{\mathrm{nom}}\bigr)^{\vee}
   = \begin{pmatrix} \nu \\ \omega \end{pmatrix}
\quad
(\nu\text{: translation-like},\;\omega\text{: rotation-like}).
$$

Choose diagonal stiffness / damping in that chart:

$$
K = \mathrm{diag}(k_t I_3,\, k_r I_3),
\qquad
D = \mathrm{diag}(d_t I_3,\, d_r I_3).
$$

Command a body wrench and map it to joints:

$$
F = K\xi - D V_b,
\qquad
\tau = J_b(q)^\top F \;\;(+\;g(q)\text{ added by the FR3}).
$$

The defaults are libfranka’s `cartesian_impedance_control` example values: $k_t = 150$ N/m, $k_r = 10$ Nm/rad, $d = 2\sqrt{k}$.

The fingers are not part of this controller; the gripper holds its own width.

### Redundancy and the null space

$J_b$ is $6\times7$, so it has (at least) a one-dimensional **null space**: joint velocities with $J_b\dot q = 0$ move the elbow without moving the gripper. $\tau = J_b^\top F$ produces no torque along that direction, so nothing holds the elbow. Franka’s own example adds a small **null-space spring** toward $q_{\mathrm{nom}}$, projected so it does not disturb the task:

$$
\tau_{\mathrm{null}} = \bigl(I - J_b^\top J_b^{\top+}\bigr)\bigl(K_n(q_{\mathrm{nom}}-q) - D_n\dot q\bigr).
$$

This controller leaves it out on purpose, so you can see the effect.

### A gentle warning about “just add damping”

In a perfect single-axis toy model, more $D$ only overdamps. On a real (or simulated) arm with **delay**, **noisy $\dot q$**, and **light wrists**, large $D$ can chatter. Increase $K$ and $D$ together, and prefer modest steps when tuning in Viser.

---

## 3. How it is implemented

| Piece | Role |
|-------|------|
| `controller/ImpedanceControl.py` | Modes `j` / `t`; Pinocchio `log6` + LOCAL Jacobian |
| `configs/franka_fr3/controllers/impedance.yaml` | `kq`/`dq`, `kt_*`/`dt_*`, `task_frame`, `plant_gravity_comp` |

**Try:** set very soft $k_t$ in task mode and push the gripper — it should yield in translation while still roughly holding orientation if $k_r$ stays larger. Then drag the elbow and watch it stay put (null space).

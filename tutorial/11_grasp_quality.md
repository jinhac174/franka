# Grasp quality: how hard must two fingers squeeze?

[10](10_reactive_grasping.md) picked grasps and squeezed with a fixed 30 N. That works, but it hides the real question of grasping with a parallel gripper:

> For **this** grasp on **this** object, what is the smallest squeeze force that holds it?

A parallel gripper has exactly two decisions: **where** to grasp (the contact pair and the hand orientation) and **how hard** (one force). This tutorial builds a model of the second one, then checks it against MuJoCo.

---

## 1. Run

```bash
python run_grasp_quality.py --config configs/grasp_quality/default.yaml

python run_grasp_quality.py --config configs/grasp_quality/default.yaml --view   # watch it
```

No LCM here: a single script with a **floating Robotiq 2F-85** (no arm) and the mustard bottle. For 10 antipodal grasps it

1. predicts the minimum grip force with the model below, and
2. finds the force MuJoCo actually needs, by bisection: close with gravity off, switch gravity on, lift 10 cm, and check that the bottle moved less than 5 mm / 5° in the hand.

It takes about 25 s. Expected output:

```
 #  width  tilt  r_com tau_axis | F_square F_pred  F_sim  sim/pred
 0  0.057   2.3    58mm   0.013Nm |      3.8    3.8    3.7   0.96
 1  0.055  22.2    75mm   0.008Nm |      4.0    7.6    9.1   1.19
 6  0.054  20.7    75mm   0.004Nm |      4.0    7.2    5.9   0.82
 8  0.057  15.8    39mm   0.050Nm |      5.0    5.6    3.9   0.70
 ...
median F_sim / F_pred = 0.96, correlation 0.91
```

These forces are far below the 2F-85's 20 N minimum: the real gripper always squeezes harder than this bottle needs. The study is about the model, so the floating gripper may go lower. Its MuJoCo linkage has springs that already squeeze ~2 N with the motor off, so the sim cannot test less than that: such grasps print `<=2.2 (held at the sim's minimum squeeze)`.

| Column | Meaning |
|--------|---------|
| `tilt` | Worst angle (deg) between a surface normal and the squeeze direction |
| `r_com` | Distance from the grasp centre to the bottle’s centre of mass |
| `tau_axis` | Torque the weight makes about the finger axis |
| `F_square` | Model that assumes both surfaces face the fingers squarely |
| `F_pred` | Model with each contact’s friction cone around its real normal |
| `F_sim` | What MuJoCo needed |

---

## 2. Ideas

### Contact model

Each fingertip pad touches the object at (roughly) one point. With Coulomb friction coefficient $\mu$ the contact force $f$ must stay inside a **friction cone** around the inward surface normal $m$:

$$
\|f_t\| \le \mu\, f_n,\qquad f_n = f\cdot m .
$$

A rubber pad also resists **twisting** about the normal, up to a torque $\gamma f_n$ (torsional friction; $\gamma$ has units of length — think “effective contact radius × $\mu$”). Combined, the soft-finger limit is an ellipse:

$$
\Bigl(\frac{\|f_t\|}{\mu}\Bigr)^2 + \Bigl(\frac{\tau_n}{\gamma}\Bigr)^2 \le f_n^2 .
$$

The finger pads in `fr3_robotiq_robot.xml` use $\mu = 0.8$, $\gamma = 8$ mm, and so does the model.

### What the contacts must carry

Gravity pulls the object with $f_g = m g$ at its centre of mass. About the TCP (midway between the fingers) the two contacts together must supply

$$
w = \begin{pmatrix} -f_g \\ -\,r \times f_g \end{pmatrix},\qquad r = p_{\mathrm{com}} - p_{\mathrm{tcp}} .
$$

Split it along the finger axis $y$:

- **Along $y$:** carried by the gripper mechanism (the two fingers are coupled), not by friction.
- **Tangential force $f_t$ and torque $\tau_\perp$ perpendicular to $y$:** shared by the two contacts, a distance $d$ (the grasp width) apart. A torque about an axis perpendicular to $y$ is a *pair* of opposite tangential forces:

$$
t_{1,2} = \tfrac12 f_t \mp \frac{\tau_\perp \times y}{d}.
$$

- **Torque about $y$:** shared as torsional friction, $\tfrac12\tau_y$ each.

Each finger presses with $F$ along $\pm y$, so contact $i$ sees $f_i = \pm F y + t_i$. The required force $F$ is the smallest one for which both contacts satisfy the soft-finger limit. It grows linearly with the object’s mass.

### Why the surface tilt matters

The first version of the model assumed the surfaces face the fingers squarely (`F_square`). It matched MuJoCo well for grasps with small tilt, but was off by up to $4.7\times$ for grasps on the bottle’s sloped sides. On a tilted surface the squeeze force itself is partly **tangential** to the surface, so it uses up part of the friction budget before any load is applied. Checking each contact’s cone around its *real* normal (`F_pred`) fixes most of that:

| Object, grasps | `F_sim / F_square` | `F_sim / F_pred` |
|--------|------|------|
| bottle, tilt ≤ 16° | 0.7 – 1.0 | 0.7 – 1.0 |
| bottle, tilt ≈ 20° | 1.5 – 2.3 | 0.8 – 1.2 |
| soup can, tilt 15–21° | 2.7 – 5.1 | 1.5 – 2.0 |

The remaining factor up to 2 on the can's tilted grasps comes from what the model leaves out: a real pad is a 2.2 × 3.8 cm patch, not a point, and the object rotates slightly while it is squeezed, which changes the contact geometry. This is why practice multiplies a model like this by a **safety factor** — tutorial [12](12_force_aware_grasping.md) uses 2.5 (and on the 2F-85 never less than its 20 N minimum).

Even where it is off, the model **ranks** grasps well (correlation 0.91 on the bottle, 0.98 on the can). That is what [10](10_reactive_grasping.md) uses it for: prefer grasps that need little force.

### Antipodal sampling (recap)

Candidate contact pairs come from rays shot through the object (see [10](10_reactive_grasping.md), step 1). The friction-cone test there keeps only pairs whose squeeze line stays well inside the cones (`cone_frac = 0.6` of $\arctan\mu$, i.e. tilt ≤ 23° for $\mu = 0.8$) — the column `tilt` above.

---

## 3. Making the simulation trustworthy

Three MuJoCo details had to be right before the comparison meant anything. They are worth knowing for any grasping simulation:

1. **Don’t teleport the hand.** Moving a `mocap` body sets its position but gives it zero velocity. MuJoCo’s friction acts on velocities, so a “lifted” object just stays behind. The hand is a free body pulled toward the mocap target by a `weld` constraint instead, so it moves with real velocities.
2. **Friction creep.** MuJoCo’s contacts are soft: under a steady load, an object slowly slides even far inside the friction cone (~8 mm/s at 70 N without the fix). The `noslip` post-solver (`noslip_iterations`) removes that, so slipping in the sim means the friction limit was really exceeded. The grasping env of tutorials 10 and 12 uses the same setting, plus elliptic friction cones.
3. **Know what the fingers collide with.** MuJoCo collides with a mesh’s **convex hull**. Planning the hand against the original (concave) mesh made the fingers clip the bottle’s shoulders in tutorial 10.

---

## 4. How it is implemented

| Piece | Role |
|-------|------|
| `run_grasp_quality.py` | Gripper-only model (built from the FR3 model file), trials, bisection |
| `utils/grasping/parallel_jaw.py` | `sample_antipodal_grasps`, `grasp_tcp_poses`, `HandCollisionChecker`, `required_grip_force` |
| `robot_platform/sim/franka_fr3/SimFrankaFR3.py` (`_GripperSim`) | The same `move` / `grasp` emulation as the full robot |
| `configs/grasp_quality/default.yaml` | Object, $\mu$, $\gamma$, number of grasps, trial timing |

**Try:**

- Point `object_config` at `configs/objects/tomato_soup_can.yaml`. Its squarely-faced grasps need less than the sim's ~2 N minimum; the tilted ones 1.5–2× the predicted force (correlation 0.98).
- Lower `mu` to 0.5 in the YAML *and* in `fr3_robotiq_robot.xml` (the `robotiq_pad_box1` / `robotiq_pad_box2` classes) and compare the tables. Which grasps’ forces rise most? (The model predicts: the tilted ones, because a tilted surface leaves less room inside a narrower cone.)

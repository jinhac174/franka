# Force-aware picking: weigh, then squeeze

[10](10_reactive_grasping.md) squeezed with a fixed 30 N; [11](11_grasp_quality.md) showed how to compute the force a grasp really needs — *if you know the mass*. Here the robot finds the mass itself, with its joint torque sensors, and then grasps with the right force.

---

## 1. Run

```bash
python run_env.py --config configs/franka_fr3/env/grasping.yaml

python run_controller.py --config configs/franka_fr3/controllers/force_aware_grasp.yaml

python run_visualizer.py --config configs/visualizer/franka_fr3/grasping.yaml
```

Same keys as [10](10_reactive_grasping.md) (`v` start, `x` release, `o` move the object).

### What you should observe

```
Grasping: width 0.054 m, force 30.0 N (model needs 7.2 N)
Test lift (6 cm): settling, then weighing...
Weighed: m_est = 0.519 kg (env says 0.603 kg). Model needs 6.2 N -> re-grasp with 20.0 N (safety x2.5).
Lifted and holding. Press x to release.
```

The estimate is a rough scale (here 14% low; the test accepts ±25%). With the Robotiq 2F-85 the re-grasp force rarely goes below its **20 N minimum** for a bottle like this one (2.5 × 6.2 N = 15.5 N → 20 N): the weighing matters for heavier objects, where 2.5 × the model force exceeds 20 N, and for not crushing light ones with a fixed large force.

**Try:** a heavier object (raise `mass` in `configs/objects/mustard_bottle.yaml`, e.g. to 2.0 kg) and watch the re-grasp force rise above 20 N.

---

## 2. Ideas

### The FR3’s joint torque sensors

Each FR3 joint has a torque sensor *after* the gearbox, so it measures the torque actually transmitted to the link, $\tau_J$ — gear friction is not in it. At rest:

$$
\tau_J = g(q) + J(q)^\top w ,
$$

where $g(q)$ is the arm’s own gravity (known from the model) and $w$ is the wrench the arm exerts at the TCP to hold whatever it carries. The simulator publishes exactly this link-side torque (`SimFrankaFR3._link_side_torque`); on the real robot libfranka provides `tau_J` (and even `tau_ext_hat_filtered`).

### Weighing

1. **Tare.** Just before closing (gripper empty, arm still): $\tau_{\mathrm{ext}}^{0} = \tau_J - g(q)$. This removes the model’s own error (e.g. the URDF and the real gripper differ by ~0.15 Nm).
2. **Probe grasp** with a firm force (30 N; the 2F-85 cannot go below 20 N anyway) and lift **6 cm**. (With 2 cm the tall bottle, hanging slightly tilted, still touched the table, which carried most of its weight: the scale read ~0.)
3. **Settle, average** $\tau_{\mathrm{ext}}$ for 0.5 s, and solve $\Delta\tau = J^\top w$ in least squares (7 equations, 6 unknowns). The mass is $m = w_{f_z}/g$.

If the estimate is below 0.05 kg, the object did not come up (it slid out or stayed on the table): open, back off, and squeeze harder next time.

### Re-grasp with the right force

$$
F = \text{safety}\times F_{\mathrm{model}}(m_{\mathrm{est}}),\qquad \text{safety} = 2.5,
$$

with $F_{\mathrm{model}}$ from [11](11_grasp_quality.md) (it scales linearly with mass). The safety factor covers the model’s worst case found in tutorial 11.

The re-grasp targets the **measured** width minus 2 mm. `grasp()` first moves toward its target width; a target slightly *wider* than the object would open the fingers and drop it.

### Hold the weight with feed-forward

The FR3 compensates its own weight and the configured gripper, not your object. With the load wrench known, add it to the command:

$$
\tau_{\mathrm{ff}} \mathrel{+}= J(q)^\top w_{\mathrm{est}} .
$$

The PD spring then no longer has to carry the load, and the sag disappears.

### Slip and escalation

During lift and hold, `is_grasped` from the gripper is monitored. If the object is lost — or the scale reads ~0 — open, back off, and multiply the probe force by 1.5 for the next attempt.

---

## 3. How it is implemented

| Piece | Role |
|-------|------|
| `controller/ForceAwareGraspControl.py` | Subclass of tutorial 10: tare, weigh, re-grasp, load feed-forward, escalation |
| `controller/ParallelJawGraspControl.py` | Reach / approach / gripper / state machine (tutorial 10) |
| `robot_platform/sim/franka_fr3/SimFrankaFR3.py` | Link-side torque `tau_J` like the FR3 sensors |
| `configs/franka_fr3/controllers/force_aware_grasp.yaml` | `force:` section (probe force, test lift, safety factor, escalation) |

**Real robot notes:** on hardware, weigh with libfranka’s `tau_ext_hat_filtered` or `tau_J − g(q)` exactly as here; keep the tare (sensor offsets drift). Set the real gripper’s mass in Franka Desk, or the tare will hide it but feed-forward will be off.

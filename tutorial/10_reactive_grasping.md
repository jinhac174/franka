# Reactive parallel-jaw grasping

This is the first “full story” grasping tutorial: the robot **chooses** where to grasp an object, **reaches** a safe pre-grasp pose, **approaches** in a straight line, closes the Robotiq 2F-85, and lifts — while the object can still be moved around until the fingers close.

If [06](06_inverse_kinematics_quadratic_program.md) was “the gripper chases a keyboard target,” this is “the gripper chases a grasp on a live object,” with care about **which** grasp and **how** to approach it.

---

## 1. Run & LCM communication

```bash
python run_env.py --config configs/franka_fr3/env/grasping.yaml

python run_controller.py --config configs/franka_fr3/controllers/parallel_jaw_grasp.yaml

python run_visualizer.py --config configs/visualizer/franka_fr3/grasping.yaml
```

| Key | Where | Action |
|-----|--------|--------|
| `v` | controller | Start (or restart) the grasp |
| `x` | controller | Release: open the gripper and stop |
| `h` | controller | Joint-space home |
| `o` | env | Put the object somewhere else (random XY and yaw) |
| `r` | env | Full reset |
| `p` / `q` | both | Pause / quit |

### Suggested first run

Start all three processes, wait for `... candidate TCP poses for mustard_bottle`, then press `v` in the controller terminal. Watch Viser: the bottle mesh, the selected contact pair (two dots), and the goal frame of the gripper. Press `o` in the env terminal *during* the reach to see the controller re-plan.

### What you should observe

- The controller prints its state changes: `reach -> approach -> close -> lift -> hold`.
- Reach takes a few seconds; the approach is a slow 10 cm straight line along the gripper’s axis; then `Grasping: width 0.055 m, force 30.0 N (model needs 7.7 N)`.
- The bottle is lifted about 15 cm and held. The gripper is a few millimetres below its goal: the arm sags under the bottle’s weight, which nothing compensates yet ([12](12_force_aware_grasping.md) fixes that).
- Not every object pose works on the first try. You may see `no collision-free IK for the pre-grasp, next one.`, `Approach blocked ... trying another grasp.`, or, for a few poses, `No valid grasp for this object pose`. These are the fallbacks doing their job; in testing, roughly 5 out of 7 random poses end with the bottle lifted.

### Three clocks (keep this picture)

| Loop | Rate | Job |
|------|------|-----|
| **Control** | 1 kHz | Track a joint reference with stiff PD (see below) |
| **Plan** | ~50–100 Hz | Select a grasp, IK, velocity fields, collision-aware QP |
| **Gripper** | its own | `move` / `grasp` run to completion on the gripper’s own controller |

### LCM

| Channel | Direction | Role |
|---------|-----------|------|
| `franka_arm_joint_meas`, `franka_gripper_joint_meas` | env → ctrl / viz | State |
| `franka_gripper_state` | env → ctrl | Width, `is_grasped`, `busy`, result of the last command |
| `franka_object_pose` | env → ctrl / viz | Object pose in the world (on the real robot: from perception, [13](13_foundation_pose_sim.md)) |
| `franka_object_physics` | env → ctrl | Mass, COM, gravity |
| ColInfo (robot + static) | env → ctrl / viz | FCL shapes ([04](04_publish_collision_geom.md)) |
| `franka_arm_joint_ctrl` | ctrl → env | Arm command |
| `franka_gripper_cmd` | ctrl → env | `MOVE` (open) / `GRASP` |
| `franka_grasp_target_tcp`, `franka_grasp_candidates` | ctrl → viz | Goal frame, selected contact pair |

```
Env -- joints, gripper state, object pose, ColInfo --> Controller -- JointCtrl, GripperCmd --> Env
                                                          |
                                                          +-- goal frame, contact pair --> Viser
```

---

## 2. Ideas

### Step 1 — Where can two fingers hold this object? (offline)

A parallel gripper needs two contact points it can squeeze **without the fingers sliding**. Sample a surface point $p_1$ with outward normal $n_1$, shoot a ray into the object along $-n_1$, and call the exit point $p_2$ (normal $n_2$). The pair is **antipodal** if the squeeze line $a = (p_2-p_1)/\|p_2-p_1\|$ lies inside both friction cones:

$$
\angle(-n_1,\,a) \le \alpha,\qquad \angle(n_2,\,-a) \le \alpha,\qquad \alpha = c\cdot\arctan\mu ,
$$

with $c<1$ a safety margin. The width $\|p_2-p_1\|$ must also fit the 85 mm gripper (with 1 cm to spare).

Each pair becomes **several gripper poses**: the pad centres sit at the midpoint (the 2F-85's pads ride on an arc, so the TCP is shifted along $z$ by a few mm depending on the width: `robotiq_2f85.pad_z`), its $y$ axis (the finger closing direction) along $a$, and its $z$ axis (the approach direction) is swept around $a$ in 30° steps. Swapping the fingers (a 180° turn about $z$) gives the same grasp, so both are kept.

Two more offline filters:

- **The open gripper must not hit the object** (five boxes: body, two fingertips, two outer linkages; `parallel_jaw.hand_boxes`) — at the grasp *and* along the straight approach from the pre-grasp (10 cm back along $z$). The check uses the object’s **convex hull**: that is what MuJoCo collides with, and it is conservative on a real object.
- **Required grip force** for each grasp, from the model of [11](11_grasp_quality.md).

For the mustard bottle this leaves about 120 candidate poses.

### Step 2 — Which grasp, right now? (online, every plan)

Place every candidate with the live object pose $T_{\mathrm{obj}}$ and keep those that

- approach from above (approach axis at most 60° from straight down),
- keep the open fingers at least 4 cm above the table,
- are within reach.

Score the rest (lower is better):

$$
s = \frac{F_{\mathrm{req}}}{F_{\mathrm{ref}}} + w_{\mathrm{rot}}\,\angle(R_{\mathrm{tcp}}, R_{\mathrm{grasp}}) + w_{\mathrm{pos}}\,\|p_{\mathrm{tcp}}-p_{\mathrm{grasp}}\|.
$$

Re-picking the best grasp every plan makes the target **jump**. The selection is therefore **sticky** (switch only when another grasp is clearly better for a while), and once the reach toward a grasp has been planned it is **committed**: it only changes if the grasp becomes invalid or the object moves.

### Step 3 — Reach: inverse kinematics, then joint space

The pre-grasp pose is solved with the CLIK of [09](09_inverse_kinematics.md) (only the 7 arm joints, 5 cm inside the joint limits), and the arm drives there with a joint-space velocity field through the collision-aware QP of [06](06_inverse_kinematics_quadratic_program.md):

$$
\dot q^\star = K_q\,(q_{\mathrm{pre}} - q).
$$

Why not just a Cartesian field all the way? Far from the goal, a large reorientation plus a translation makes the Cartesian QP trade one against the other, and near awkward poses it crawls. Solving IK up front also catches unreachable grasps **immediately**: the IK fails (or its solution collides, or sits too close to a singularity) and the next candidate is tried.

**Singularities.** The FR3 has a shoulder singularity at $q_2 \approx 0$, where the axes of joints 1 and 3 line up: turning them in opposite directions barely moves the hand. An IK solution there makes the approach crawl, so solutions whose TCP Jacobian has a small singular value ($\sigma_{\min} < 0.08$; the ready pose has $0.22$) are rejected.

### Step 4 — Approach: a straight line

From the pre-grasp, the TCP goes straight in along its $z$ axis with a 6-D pose velocity field

$$
v^\star = \mathrm{sat}\bigl(K_p(p_{\mathrm{grasp}}-p),\, v_{\max}\bigr),\qquad
\omega^\star = \mathrm{sat}\bigl(K_r\log(R_{\mathrm{grasp}}R^\top)^\vee,\, \omega_{\max}\bigr),
$$

and the QP of [06](06_inverse_kinematics_quadratic_program.md) with the full $6\times7$ TCP Jacobian. If the object moves more than 2 cm during the approach, go back to reach.

The gripper closes only when **both** the planned pose and the **measured** TCP are within tolerance. If the fingers push on the object, the real hand lags behind the plan; closing then would grasp off-centre. An approach that makes no progress for a few seconds backs off and tries another grasp.

### Step 5 — Close and lift

`grasp(width, speed=0.1, force=30 N, ε=±1 cm)` on the gripper (the 2F-85 squeezes with 20–235 N); it reports success if the fingers stop within the window. Then lift 15 cm. While lifting and holding, `is_grasped` is monitored; if it drops, open and start over.

### Control: track a reference, don’t just damp velocity

Tutorial 06 tracked $\dot q_{\mathrm{des}}$ with damping only. That cannot hold a load: with a 0.6 kg bottle the shoulder needs about 3 Nm, and a damper only produces torque while it is moving. Here the 1 kHz loop integrates the plan into a joint reference and tracks it with libfranka’s stiff joint gains:

$$
q_{\mathrm{ref}} \mathrel{+}= \dot q_{\mathrm{des}}\,\Delta t,\qquad
\tau = K_p(q_{\mathrm{ref}}-q) + K_d(\dot q_{\mathrm{des}}-\dot q) + C(q,\dot q)\dot q .
$$

Two details matter:

- The planner plans on $q_{\mathrm{ref}}$, not on the measured $q$. Otherwise the arm’s small lag behind $q_{\mathrm{ref}}$ (load, friction) is fed back into the integrator, and the result is a slow wobble.
- No friction feed-forward here: the 600 Nm/rad spring already beats the 1.1 Nm joint friction with a 2 mrad error, and a friction term that flips sign with $\dot q_{\mathrm{des}}$ near a goal causes a stick-slip limit cycle.

Safety nets: $q_{\mathrm{ref}}$ stays within 0.05 rad of $q$ (no wind-up when blocked), and if the planner stops delivering plans for 0.2 s the reference stops moving.

---

## 3. How it is implemented

| Piece | Role |
|-------|------|
| `controller/ParallelJawGraspControl.py` | Candidates, sticky selection, IK reach, pose approach, gripper, state machine, reference tracking |
| `utils/grasping/parallel_jaw.py` | Antipodal sampling, TCP poses, hand collision check, grip-force model |
| `utils/grasping/objects.py`, `configs/objects/*.yaml` | Object mesh, mass, upright orientation |
| `env/MujocoGraspEnv.py` | Object in the scene, `o` re-placement, pose / physics publishing |
| `robot_platform/sim/franka_fr3/SimFrankaFR3.py` (`_GripperSim`) | Gripper `move` / `grasp` emulation (the real 2F-85: `robot_platform/hardware/franka_fr3/robotiq.py`) |
| `configs/franka_fr3/controllers/parallel_jaw_grasp.yaml` | All knobs (`grasp:` section) |

**Try:** grasp the tomato soup can instead. Add `--object_config configs/objects/tomato_soup_can.yaml` to both the env and the controller command (and set `object_mesh_path: assets/object_mesh/0715_tomato_soup_can/mujoco_mesh.obj` in the visualizer YAML). The can is a plain cylinder, so far more grasps are valid (about 530 candidate poses instead of 120) and it is picked up from every test position.

**Real robot notes:** the object pose comes from perception instead of the simulator ([13](13_foundation_pose_sim.md)), and its error (a few mm) is why the approach tolerance and the grasp window `ε` are not tighter. The table height `table_z` must match your cell.

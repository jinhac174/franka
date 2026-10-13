# Publishing collision geometry

Before a controller can say “stay away from the wall,” it needs a **description of shapes**. This tutorial is only about **publishing** those shapes from the environment over LCM. No avoidance yet — that comes in [05](05_collision_aware_grav_comp.md).

---

## 1. Run & LCM communication

```bash
python run_env.py --config configs/franka_fr3/env/collision.yaml

python run_visualizer.py --config configs/visualizer/franka_fr3/collision.yaml
```

You can run this env alone. The visualizer is optional here: tick **Show collision geometry** in the Viser panel to see the shapes on the arm and the walls. Controllers that need collision data subscribe to the same channels ([05](05_collision_aware_grav_comp.md)).

### Two channels, two frames

| Channel | Who uses it | Frame | Contents |
|---------|-------------|-------|----------|
| `franka_robot_col_info` | Controller / viz | **Link-local** | Simple shapes attached to robot links (capsules for the arm, two boxes for the gripper) |
| `franka_static_col_info` | Controller / viz | **World** | Floor and walls |

Each named primitive is a small record:

- a **type** (`box`, `cylinder`, `sphere`, or `capsule`),
- **size** parameters,
- a 4×4 **offset** transform.

The env republishes on a timer (`col_info_update_dt`) so a controller that starts late still receives geometry.

### Where the FR3 shapes come from

Franka ships its own coarse collision model with the robot description: one or two **capsules** per link (a cylinder with hemispherical ends). The robot firmware uses exactly these for its self-collision checks. We reuse them for the arm. The Robotiq 2F-85 gets two **boxes** fitted to its collision meshes: one around its body (`fr3_gripper`) and one around the fingers fully open (`fr3_gripper__2`, conservative when they close).

Together the shapes enclose every vertex of the arm’s and the gripper’s collision meshes at any pose (checked in `tests/unit/test_collision_model.py`).

### Why not reuse MuJoCo meshes?

MuJoCo contact meshes are for **physics**. Planning libraries such as **FCL** prefer a handful of **convex primitives** that are fast for distance queries. We deliberately keep a second, coarser geometric model for control.

---

## 2. Ideas

### Placing a robot shape in the world

A link moves with the robot. If a capsule is defined in the link frame with offset $T_{\mathrm{offset}}$, its world pose at configuration $q$ is

$$
T^{\mathrm{world}}(q)
  = {}^{W}T_{\mathrm{link}}(q)\,T_{\mathrm{offset}}.
$$

Static walls skip the FK part: their offset is already expressed in the world.

**Naming.** A primitive is named after the link frame it is attached to (`fr3_link4`). When a link carries a second shape, it is called `fr3_link5__2`: “second primitive on frame `fr3_link5`.” The FK helpers strip the `__2` before looking up the frame.

### What “distance” will mean later

Given two placed shapes, a collision library returns:

- a **signed distance** $d$ (positive = separation, negative = penetration),
- two **nearest points**, one on each shape.

Controllers turn those into inequality constraints (“do not reduce $d$ too quickly”). This tutorial only ensures the shapes exist on the bus.

### Export details people trip on

- MuJoCo boxes often store **half**-extents; FCL boxes here use **full** side lengths.
- A capsule’s size is (radius, length of the straight part); its total length is that plus $2r$. It lies along its local $z$ axis.
- An infinite ground plane is exported as a **thin box** so distance queries stay well defined.
- On the real robot, the walls must match **your** cell. The ones in `scene_default.xml` are placeholders.

---

## 3. How it is implemented

| Piece | Role |
|-------|------|
| `env/MujocoColEnv.py` | Builds primitives and publishes `ColInfo` |
| `SimFrankaFR3.get_col_info_data` / `FR3_COL_CAPSULES`, `FR3_GRIPPER_BOXES` | FR3 link capsules + two gripper boxes |
| `utils/collision/wall_geoms.py` | Floor / wall boxes, read from named MJCF geoms (`floor`, `wall_*`, `table*`, `obs_*`) |
| `utils/fcl/getter.py` | Primitive → FCL shape; `col_frame_name` for `__k` names |
| `ColInfoData` + `col_info.lcm` | Message schema |
| `configs/franka_fr3/env/collision.yaml` | Channel names, republish period |

**Try this:** in Viser, tick **Show collision geometry** and compare the capsules to the robot meshes as you move the arm in MuJoCo.

#!/usr/bin/env bash
# Start a tutorial's processes in one tmux session (one pane each).
#
#   scripts/launch.sh <target> [sim|real]       (default: sim)
#   scripts/launch.sh list
#
# sim : run_env.py (MuJoCo) + controller + visualizer
# real: run_hardware.py (FR3) [+ run_scene_info.py] + controller --env_mode real
#       + visualizer [+ camera, FoundationPose, pose bridge for grasping]
#
# Keys go to the pane that has focus (Ctrl-b + arrow keys to move). Each
# process stops with Ctrl-C (or q); `tmux kill-session -t franka` stops all.
# PYTHON overrides the interpreter (default: python); extra environment
# (LCM_DEFAULT_URL, MUJOCO_GL, ...) is inherited.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
# HEADLESS=1: no MuJoCo viewer window (machines without a display).
ENV_EXTRA=""; [[ ${HEADLESS:-0} == 1 ]] && ENV_EXTRA="--vis_mode vis_off"
SESSION=${SESSION:-franka}
TARGET=${1:-list}
MODE=${2:-sim}

E=configs/franka_fr3/env
C=configs/franka_fr3/controllers
V=configs/visualizer/franka_fr3

# target -> "env_config controller_config visualizer_config needs_scene_info"
declare -A T=(
  [grav_comp]="$E/default.yaml $C/grav_comp.yaml $V/default.yaml 0"           # 01-03
  [collision]="$E/collision.yaml - $V/collision.yaml 1"                        # 04
  [col_aware]="$E/collision.yaml $C/col_aware_grav_comp.yaml $V/collision.yaml 1"  # 05
  [ik_qp]="$E/collision.yaml $C/ik_qp.yaml $V/ik_qp.yaml 1"                    # 06
  [min_jerk]="$E/collision.yaml $C/min_jerk_p2p.yaml $V/min_jerk_p2p.yaml 1"   # 07
  [impedance]="$E/default.yaml $C/impedance.yaml $V/impedance.yaml 0"          # 08
  [ik_reach]="$E/collision.yaml $C/ik_reach.yaml $V/ik_reach.yaml 1"           # 09
  [grasp]="$E/grasping.yaml $C/parallel_jaw_grasp.yaml $V/grasping.yaml 1"     # 10
  [force_grasp]="$E/grasping.yaml $C/force_aware_grasp.yaml $V/grasping.yaml 1"  # 12
)
# Grasping from perception in the sim: camera env + FoundationPose (GPU) + bridge.
declare -A T_PERCEPTION=(
  [grasp_perception]="$C/parallel_jaw_grasp.yaml"
  [force_grasp_perception]="$C/force_aware_grasp.yaml"
)

usage() {
  echo "usage: scripts/launch.sh <target> [sim|real]"
  echo "targets:"
  for k in grav_comp collision col_aware ik_qp min_jerk impedance ik_reach grasp force_grasp; do
    echo "  $k"
  done
  echo "  grasp_perception / force_grasp_perception  (object pose from FoundationPose; needs a GPU)"
  echo "  perception        tutorial 13 (FoundationPose on the sim camera; GPU)"
  echo "  quality           tutorial 11 (single process, no tmux)"
}

panes=()
add() { panes+=("$*"); }

case "$TARGET" in
  list|-h|--help) usage; exit 0 ;;
  quality) exec $PY run_grasp_quality.py --config configs/grasp_quality/default.yaml ;;
  perception)
    add "$PY run_env.py --config configs/perception_tutorial/env/table_mesh_cam.yaml"
    add "$PY run_perception.py --config configs/perception_tutorial/foundation_pose_sim.yaml"
    add "$PY run_visualizer.py --config configs/perception_tutorial/visualizer/obj_pose.yaml"
    ;;
  grasp_perception|force_grasp_perception)
    ctrl=${T_PERCEPTION[$TARGET]}
    if [[ $MODE == sim ]]; then
      add "$PY run_env.py --config $E/grasping_perception.yaml $ENV_EXTRA"
      add "$PY run_perception.py --config configs/franka_fr3/perception/foundation_pose_sim.yaml"
    else
      add "$PY run_hardware.py --config configs/franka_fr3/hardware/fr3.yaml"
      add "$PY run_scene_info.py --config configs/franka_fr3/hardware/scene_info.yaml"
      add "$PY run_sensors.py --config configs/sensors/zed/droid.yaml"
      add "$PY run_perception.py --config configs/franka_fr3/perception/foundation_pose_real.yaml"
    fi
    add "$PY run_pose_bridge.py --config configs/franka_fr3/perception/pose_bridge.yaml"
    extra=""; [[ $MODE == real ]] && extra="--env_mode real"
    add "$PY run_controller.py --config $ctrl $extra"
    add "$PY run_visualizer.py --config $V/grasping.yaml"
    ;;
  *)
    if [[ -z ${T[$TARGET]+x} ]]; then usage; exit 1; fi
    read -r env ctrl vis scene <<<"${T[$TARGET]}"
    if [[ $MODE == sim ]]; then
      add "$PY run_env.py --config $env $ENV_EXTRA"
    elif [[ $MODE == real ]]; then
      # run_scene_info.py publishes the real cell (scene_real_cell.xml: floor,
      # wall_front, wall_right); controllers must name those obstacles.
      case $TARGET in
        col_aware|collision)
          [[ $ctrl != - ]] && ctrl=$C/col_aware_real_cell.yaml
          vis=$V/collision_real_cell.yaml ;;
        ik_qp|min_jerk|ik_reach)
          echo "$TARGET has no real-cell config yet (its obstacles are the sim's wall_back / wall_left)." >&2
          exit 1 ;;
      esac
      add "$PY run_hardware.py --config configs/franka_fr3/hardware/fr3.yaml"
      [[ $scene == 1 ]] && add "$PY run_scene_info.py --config configs/franka_fr3/hardware/scene_info.yaml"
      if [[ $TARGET == grasp || $TARGET == force_grasp ]]; then
        echo "On the robot the object pose must come from perception:" >&2
        echo "  scripts/launch.sh ${TARGET}_perception real" >&2
        exit 1
      fi
    else
      usage; exit 1
    fi
    extra=""; [[ $MODE == real ]] && extra="--env_mode real"
    [[ $ctrl != - ]] && add "$PY run_controller.py --config $ctrl $extra"
    add "$PY run_visualizer.py --config $vis"
    ;;
esac


if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' exists: tmux attach -t $SESSION (or kill it first)" >&2
  exit 1
fi

# Panes start a shell that keeps open after the process exits (read the error).
first=1
for cmd in "${panes[@]}"; do
  wrapped="echo '\$ $cmd'; $cmd; echo; echo '[exited, press Enter]'; read"
  if [[ $first == 1 ]]; then
    tmux new-session -d -s "$SESSION" -x 220 -y 60 "bash -c \"$wrapped\""
    first=0
  else
    tmux split-window -t "$SESSION" "bash -c \"$wrapped\""
    tmux select-layout -t "$SESSION" tiled >/dev/null
  fi
  sleep 1.5  # env / hardware first: later processes wait for its messages
done
tmux select-layout -t "$SESSION" tiled >/dev/null
echo "Started ${#panes[@]} processes in tmux session '$SESSION' ($TARGET, $MODE)."
[[ -n ${TMUX:-} ]] || exec tmux attach -t "$SESSION"

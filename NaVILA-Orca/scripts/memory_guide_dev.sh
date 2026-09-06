#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="${PROJECT_ROOT}/scripts"
source "${SCRIPT_DIR}/orcalab_env.sh"

STATE_DIR="${NAVILA_DEV_STATE_DIR:-${PROJECT_ROOT}/outputs/dev_stack}"
RUN_ROOT="${NAVILA_DEV_RUN_ROOT:-${PROJECT_ROOT}/outputs/memory_guide}"
RUNS_DIR="${RUN_ROOT}/runs"
LATEST_LINK="${RUN_ROOT}/latest-run"
TELEOP_LATEST_LINK="${RUN_ROOT}/latest-teleop"
ORCALAB_SCENE="${NAVILA_ORCALAB_SCENE:-SimpleMovement_DiningTable}"
ORCALAB_LAYOUT="${NAVILA_ORCALAB_LAYOUT:-${PROJECT_ROOT}/../dethread/kitchen2.json}"
ORCALAB_VK_ICD="${NAVILA_ORCALAB_VK_ICD:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
ORCALAB_MCP_PORT="${NAVILA_ORCALAB_MCP_PORT:-12345}"
ORCAGYM_PORT="${NAVILA_ORCAGYM_PORT:-50051}"
ORCALAB_EDIT_PORT="${NAVILA_ORCALAB_EDIT_PORT:-50151}"
NAVVLM_PORT="${NAVVLM_PORT:-54321}"
NAVILA_SERVER_MODE="${NAVILA_SERVER_MODE:-aws}"
NAVILA_AWS_INSTANCE_ID="${NAVILA_AWS_INSTANCE_ID:-i-066515f762428ba55}"
NAVILA_AWS_PROFILE="${NAVILA_AWS_PROFILE:-navila}"
NAVILA_AWS_REGION="${NAVILA_AWS_REGION:-ap-northeast-1}"
START_TIMEOUT="${NAVILA_DEV_START_TIMEOUT:-180}"

usage() {
  cat <<EOF
Usage:
  $0 start
  $0 status
  $0 run --query "Where are my glasses?" [runner options]
  $0 teleop [teleop options]
  $0 inspect [RUN_DIRECTORY]
  $0 stop

Environment overrides:
  NAVILA_ORCALAB_SCENE       OrcaLab scene name
  NAVILA_ORCALAB_LAYOUT      Absolute layout JSON path
  NAVILA_ORCALAB_VK_ICD      Vulkan ICD JSON used to launch OrcaLab
  NAVILA_SERVER_MODE         NaVILA backend: aws or local (default: aws)
  NAVILA_AWS_INSTANCE_ID     AWS instance hosting NaVILA
  NAVILA_AWS_PROFILE         AWS CLI profile (default: navila)
  NAVILA_AWS_REGION          AWS region (default: ap-northeast-1)
  NAVILA_DEV_START_TIMEOUT   Service startup timeout in seconds

Completed teleop data is consumed automatically from
outputs/memory_guide/latest-teleop/teleop.json and rebuilt as a semantic map
for each run. Pass --no-semantic-map to run the legacy text-only patrol.

"start" stays in the foreground as the service supervisor. Run it in a
dedicated terminal or as an attached background task.
EOF
}

port_open() {
  local port="$1"
  (
    exec 3<>"/dev/tcp/127.0.0.1/${port}" || exit 1
    exec 3>&-
  ) >/dev/null 2>&1
}

wait_for_port() {
  local label="$1"
  local port="$2"
  local deadline=$((SECONDS + START_TIMEOUT))
  until port_open "${port}"; do
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for ${label} on 127.0.0.1:${port}" >&2
      return 1
    fi
    sleep 1
  done
}

wait_for_stable_port() {
  local label="$1"
  local port="$2"
  local required_successes="${3:-3}"
  local successes=0
  local deadline=$((SECONDS + START_TIMEOUT))
  while ((successes < required_successes)); do
    if port_open "${port}"; then
      ((successes += 1))
    else
      successes=0
    fi
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for stable ${label} on 127.0.0.1:${port}" >&2
      return 1
    fi
    sleep 1
  done
}

pid_is_running() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] || return 1
  local pid
  pid="$(<"${pid_file}")"
  [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null
}

start_owned_process() {
  local name="$1"
  local log_file="$2"
  shift 2
  setsid "$@" >>"${log_file}" 2>&1 &
  echo "$!" >"${STATE_DIR}/${name}.pid"
}

stop_owned_process() {
  local name="$1"
  local pid_file="${STATE_DIR}/${name}.pid"
  if ! pid_is_running "${pid_file}"; then
    rm -f "${pid_file}"
    return
  fi
  local pid
  pid="$(<"${pid_file}")"
  kill -- "-${pid}" 2>/dev/null || kill "${pid}"
  local deadline=$((SECONDS + 20))
  while kill -0 -- "-${pid}" 2>/dev/null && ((SECONDS < deadline)); do
    sleep 1
  done
  if kill -0 -- "-${pid}" 2>/dev/null; then
    echo "${name} did not stop within 20 seconds (PID ${pid})" >&2
    return 1
  fi
  rm -f "${pid_file}"
}

resolve_orcalab_cli() {
  navila_orca_require_gui
  ORCALAB_CLI="$(dirname "${NAVILA_ORCA_ORCALAB_BIN}")/orcalab-cli"
  if [[ ! -x "${ORCALAB_CLI}" ]]; then
    echo "OrcaLab CLI is missing: ${ORCALAB_CLI}" >&2
    return 2
  fi
}

simulation_state() {
  "${ORCALAB_CLI}" get_simulation_state 2>&1
}

mcp_ready() {
  local state
  if ! state="$(simulation_state)"; then
    return 1
  fi
  [[ "${state}" == *'"state"'* || "${state}" == *'\"state\"'* ]] &&
    [[ "${state}" != *"failed to connect"* ]] &&
    [[ "${state}" != *"All connection attempts failed"* ]]
}

wait_for_mcp() {
  local deadline=$((SECONDS + START_TIMEOUT))
  until mcp_ready; do
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for the OrcaLab MCP handshake on 127.0.0.1:${ORCALAB_MCP_PORT}" >&2
      echo "See ${STATE_DIR}/orcalab.log for OrcaLab startup details." >&2
      return 1
    fi
    sleep 1
  done
}

simulation_running() {
  local state="$1"
  [[ "${state}" == *'"running": true'* ||
     "${state}" == *'\"running\": true'* ]]
}

start_external_simulation() {
  local state
  state="$(simulation_state)"
  if simulation_running "${state}"; then
    return
  fi
  "${ORCALAB_CLI}" start_simulation \
    --json '{"program_name":"external"}'
  local deadline=$((SECONDS + START_TIMEOUT))
  until simulation_running "$(simulation_state)"; do
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for OrcaLab external simulation mode" >&2
      return 1
    fi
    sleep 1
  done
}

start_navila_backend() {
  if port_open "${NAVVLM_PORT}"; then
    sleep 3
    if port_open "${NAVVLM_PORT}"; then
      return
    fi
  fi

  case "${NAVILA_SERVER_MODE}" in
    local)
      start_owned_process navila "${STATE_DIR}/navila.log" \
        "${SCRIPT_DIR}/start_navvlm_server.sh"
      ;;
    aws)
      if ! command -v aws >/dev/null 2>&1; then
        echo "AWS CLI is required for NAVILA_SERVER_MODE=aws" >&2
        return 2
      fi
      if [[ -z "${NAVILA_AWS_INSTANCE_ID}" ]]; then
        echo "NAVILA_AWS_INSTANCE_ID must be set for the AWS backend" >&2
        return 2
      fi
      if ! aws sts get-caller-identity \
        --profile "${NAVILA_AWS_PROFILE}" \
        --region "${NAVILA_AWS_REGION}" \
        >/dev/null; then
        echo "AWS profile ${NAVILA_AWS_PROFILE} is not authenticated" >&2
        return 2
      fi
      start_owned_process navila "${STATE_DIR}/navila.log" \
        aws ssm start-session \
        --target "${NAVILA_AWS_INSTANCE_ID}" \
        --profile "${NAVILA_AWS_PROFILE}" \
        --region "${NAVILA_AWS_REGION}" \
        --document-name AWS-StartPortForwardingSession \
        --parameters \
        "{\"portNumber\":[\"54321\"],\"localPortNumber\":[\"${NAVVLM_PORT}\"]}"
      ;;
    *)
      echo "NAVILA_SERVER_MODE must be 'aws' or 'local', got: ${NAVILA_SERVER_MODE}" >&2
      return 2
      ;;
  esac

  local deadline=$((SECONDS + START_TIMEOUT))
  until port_open "${NAVVLM_PORT}"; do
    if ! pid_is_running "${STATE_DIR}/navila.pid"; then
      echo "NaVILA ${NAVILA_SERVER_MODE} backend exited during startup; see ${STATE_DIR}/navila.log" >&2
      return 1
    fi
    if ((SECONDS >= deadline)); then
      echo "Timed out waiting for NaVILA ${NAVILA_SERVER_MODE} backend on 127.0.0.1:${NAVVLM_PORT}" >&2
      return 1
    fi
    sleep 1
  done
  wait_for_stable_port "NaVILA server" "${NAVVLM_PORT}"
}

cleanup_supervisor() {
  trap - EXIT INT TERM
  resolve_orcalab_cli >/dev/null 2>&1 || true
  if [[ -n "${ORCALAB_CLI:-}" ]] && port_open "${ORCALAB_MCP_PORT}"; then
    "${ORCALAB_CLI}" stop_simulation >/dev/null 2>&1 || true
  fi
  stop_owned_process navila >/dev/null 2>&1 || true
  stop_owned_process orcalab >/dev/null 2>&1 || true
  rm -f "${STATE_DIR}/supervisor.pid"
}

start_stack() {
  mkdir -p "${STATE_DIR}" "${RUNS_DIR}"
  if pid_is_running "${STATE_DIR}/supervisor.pid"; then
    echo "Memory Guide supervisor is already running (PID $(<"${STATE_DIR}/supervisor.pid"))."
    return 0
  fi
  if [[ ! -f "${ORCALAB_LAYOUT}" ]]; then
    echo "OrcaLab layout does not exist: ${ORCALAB_LAYOUT}" >&2
    return 2
  fi
  if [[ ! -f "${ORCALAB_VK_ICD}" ]]; then
    echo "OrcaLab Vulkan ICD does not exist: ${ORCALAB_VK_ICD}" >&2
    return 2
  fi
  if ! command -v setsid >/dev/null 2>&1; then
    echo "setsid is required to supervise complete service process groups" >&2
    return 2
  fi

  resolve_orcalab_cli
  echo "$$" >"${STATE_DIR}/supervisor.pid"
  trap cleanup_supervisor EXIT INT TERM
  if ! mcp_ready; then
    start_owned_process orcalab "${STATE_DIR}/orcalab.log" \
      env VK_ICD_FILENAMES="${ORCALAB_VK_ICD}" \
      "${SCRIPT_DIR}/start_orcalab_gui.sh" \
      --scene "${ORCALAB_SCENE}" \
      --layout "${ORCALAB_LAYOUT}"
  fi
  wait_for_mcp
  wait_for_port "OrcaLab edit service" "${ORCALAB_EDIT_PORT}"
  start_external_simulation
  wait_for_port "OrcaGym external simulation" "${ORCAGYM_PORT}"
  start_navila_backend

  echo "Memory Guide stack is ready."
  echo "Scene: ${ORCALAB_SCENE}"
  echo "Layout: ${ORCALAB_LAYOUT}"
  echo "NaVILA backend: ${NAVILA_SERVER_MODE}"
  echo "Logs: ${STATE_DIR}"

  while true; do
    if [[ -f "${STATE_DIR}/orcalab.pid" ]] &&
       ! pid_is_running "${STATE_DIR}/orcalab.pid"; then
      echo "Owned OrcaLab process exited; see ${STATE_DIR}/orcalab.log" >&2
      return 1
    fi
    if [[ -f "${STATE_DIR}/navila.pid" ]] &&
       ! pid_is_running "${STATE_DIR}/navila.pid"; then
      echo "NaVILA ${NAVILA_SERVER_MODE} backend exited; restarting." >&2
      rm -f "${STATE_DIR}/navila.pid"
      start_navila_backend
    elif ! port_open "${NAVVLM_PORT}"; then
      echo "NaVILA endpoint disappeared; starting ${NAVILA_SERVER_MODE} backend." >&2
      stop_owned_process navila >/dev/null 2>&1 || true
      start_navila_backend
    fi
    sleep 2
  done
}

show_status() {
  mkdir -p "${STATE_DIR}"
  resolve_orcalab_cli
  printf 'Supervisor: %s\n' \
    "$(pid_is_running "${STATE_DIR}/supervisor.pid" && echo running || echo stopped)"
  printf 'OrcaLab MCP (%s): %s\n' "${ORCALAB_MCP_PORT}" \
    "$(port_open "${ORCALAB_MCP_PORT}" && mcp_ready && echo ready || echo unavailable)"
  printf 'OrcaLab edit (%s): %s\n' "${ORCALAB_EDIT_PORT}" \
    "$(port_open "${ORCALAB_EDIT_PORT}" && echo ready || echo unavailable)"
  printf 'OrcaGym (%s): %s\n' "${ORCAGYM_PORT}" \
    "$(port_open "${ORCAGYM_PORT}" && echo ready || echo unavailable)"
  printf 'NaVILA (%s): %s\n' "${NAVVLM_PORT}" \
    "$(port_open "${NAVVLM_PORT}" && echo ready || echo unavailable)"
  if port_open "${ORCALAB_MCP_PORT}" && mcp_ready; then
    printf 'Simulation: %s\n' "$(simulation_state)"
  fi
  if [[ -L "${LATEST_LINK}" ]]; then
    printf 'Latest run: %s\n' "$(readlink -f "${LATEST_LINK}")"
  fi
}

run_mission() {
  local query=""
  local args=()
  while (($#)); do
    case "$1" in
      --query)
        [[ $# -ge 2 ]] || {
          echo "--query requires a value" >&2
          return 2
        }
        query="$2"
        shift 2
        ;;
      --query=*)
        query="${1#*=}"
        shift
        ;;
      *)
        args+=("$1")
        shift
        ;;
    esac
  done
  if [[ -z "${query}" ]]; then
    echo "run requires --query" >&2
    return 2
  fi
  for port in "${ORCALAB_MCP_PORT}" "${ORCALAB_EDIT_PORT}" \
    "${ORCAGYM_PORT}" "${NAVVLM_PORT}"; do
    if ! port_open "${port}"; then
      echo "Required service on 127.0.0.1:${port} is unavailable; start the stack first." >&2
      return 1
    fi
  done

  local run_id run_dir
  run_id="$(date -u +%Y%m%dT%H%M%S)-$$"
  run_dir="${RUNS_DIR}/${run_id}"
  mkdir -p "${run_dir}"
  ln -sfn "runs/${run_id}" "${LATEST_LINK}"

  "${SCRIPT_DIR}/run_orcalab_memory_guide.sh" \
    --query "${query}" \
    --plan-output "${run_dir}/plan.json" \
    --waypoint-output "${run_dir}/waypoints.txt" \
    --output "${run_dir}" \
    "${args[@]}" \
    > >(tee "${run_dir}/run.log") 2>&1 &
  local mission_pid=$!
  echo "${mission_pid}" >"${STATE_DIR}/mission.pid"
  trap 'kill "${mission_pid}" 2>/dev/null || true; rm -f "${STATE_DIR}/mission.pid"' INT TERM

  local result=0
  wait "${mission_pid}" || result=$?
  trap - INT TERM
  rm -f "${STATE_DIR}/mission.pid"
  if ((result != 0)); then
    echo "Mission failed with exit code ${result}; see ${run_dir}/run.log" >&2
    return "${result}"
  fi
  inspect_run "${run_dir}"
}

run_teleop() {
  local args=("$@")
  for port in "${ORCALAB_MCP_PORT}" "${ORCALAB_EDIT_PORT}" "${ORCAGYM_PORT}"; do
    if ! port_open "${port}"; then
      echo "Required service on 127.0.0.1:${port} is unavailable; start the stack first." >&2
      return 1
    fi
  done

  local run_id run_dir
  run_id="$(date -u +%Y%m%dT%H%M%S)-$$"
  run_dir="${RUNS_DIR}/teleop-${run_id}"
  mkdir -p "${run_dir}"
  ln -sfn "runs/teleop-${run_id}" "${TELEOP_LATEST_LINK}"

  NAVILA_ORCA_TELEOP_OUTPUT="${run_dir}" \
    "${SCRIPT_DIR}/run_orcalab_teleop.sh" \
    "${args[@]}" \
    > >(tee "${run_dir}/teleop.log") 2>&1
  echo "Teleop artifacts: ${run_dir}"
}

inspect_run() {
  local run_dir="${1:-${LATEST_LINK}}"
  run_dir="$(readlink -f "${run_dir}")"
  local measurements="${run_dir}/measurements.json"
  if [[ ! -f "${measurements}" ]]; then
    echo "No measurements found at ${measurements}" >&2
    return 1
  fi
  navila_orca_resolve_runtime
  "${NAVILA_ORCA_PYTHON}" - "${measurements}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
summary = {
    "run_directory": str(path.parent),
    "pipeline_status": payload.get("pipeline_status"),
    "termination_reason": payload.get("termination_reason"),
    "control_steps": payload.get("control_steps"),
    "decisions": payload.get("decisions"),
    "waypoints_completed": payload.get("waypoints_completed"),
    "waypoint_count": payload.get("waypoint_count"),
    "captured_frames": payload.get("runtime", {}).get("captured_frames"),
    "metrics": payload.get("metrics"),
    "final_position": payload.get("final_state", {}).get("root_pos_world"),
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

stop_stack() {
  mkdir -p "${STATE_DIR}"
  if pid_is_running "${STATE_DIR}/mission.pid"; then
    stop_owned_process mission
  fi
  if pid_is_running "${STATE_DIR}/supervisor.pid"; then
    local supervisor_pid
    supervisor_pid="$(<"${STATE_DIR}/supervisor.pid")"
    kill "${supervisor_pid}"
    local deadline=$((SECONDS + 30))
    while kill -0 "${supervisor_pid}" 2>/dev/null && ((SECONDS < deadline)); do
      sleep 1
    done
    if kill -0 "${supervisor_pid}" 2>/dev/null; then
      echo "Supervisor did not stop within 30 seconds (PID ${supervisor_pid})" >&2
      return 1
    fi
  else
    stop_owned_process navila
    stop_owned_process orcalab
  fi
  rm -f "${STATE_DIR}/supervisor.pid"
  echo "Memory Guide stack stopped."
}

case "${1:-}" in
  start)
    shift
    start_stack "$@"
    ;;
  status)
    shift
    show_status "$@"
    ;;
  run)
    shift
    run_mission "$@"
    ;;
  teleop)
    shift
    run_teleop "$@"
    ;;
  inspect)
    shift
    inspect_run "$@"
    ;;
  stop)
    shift
    stop_stack "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
# Parallel, resumable full RoboDojo evaluation for Pi_05.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_DIR="${ROOT_DIR}/XPolicyLab/policy/Pi_05"
DEFAULT_CKPT_ROOT="${ROOT_DIR}/.cache/robodojo_ckpt_modelscope_repo/ckpt/RoboDojo/Pi_05"

ckpt_root="${DEFAULT_CKPT_ROOT}"
single_ckpt=""
policy_env="/root/miniconda3/envs/pi05"
eval_env="/root/miniconda3/envs/RoboDojo"
env_cfg="arx_x5"
action_type="joint"
seeds="0,1,2"
gpu_ids=""
eval_num="native"
only=""
run_prefix="pi05_full"
output_dir=""
dry_run="false"

usage() { cat <<'EOF'
Usage: bash scripts/run_pi05_parallel_eval.sh [options]

Runs all runnable RoboDojo Pi_05 task configurations for seeds 0, 1, and 2.
The default checkpoints are:
  .cache/robodojo_ckpt_modelscope_repo/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-{0,1,2}

Completed task/seed results are skipped.  Incomplete results from an earlier
run are resumed with their stable run id; completed layouts recorded in their
result file are synchronized to the resume manifest before continuation.
If the simulator exits without producing every requested episode, the job is
automatically restarted (up to 10 times by default) from that partial result.

Options:
  --ckpt-root PATH       Checkpoint parent directory (default shown above)
  --ckpt PATH            Use one checkpoint for every selected seed (legacy override)
  --seeds 0,1,2          Seeds to evaluate (default: 0,1,2)
  --policy-env PATH      Pi_05 conda/uv environment (default: /root/miniconda3/envs/pi05)
  --eval-env PATH        RoboDojo simulator environment (default: /root/miniconda3/envs/RoboDojo)
  --env-cfg NAME         Environment config (default: arx_x5)
  --action-type NAME     Action type (default: joint)
  --gpus 0,2             Physical GPU subset (default: detect all GPUs)
  --eval-num NUM|native  Episode count override; native uses 50 or 25 for Generalization halves
  --only a,b,c           Optional runnable-task subset
  --run-prefix ID        Stable result identity prefix (default: pi05_full)
  --output-dir PATH      Default: eval_runs/<run-prefix>_all_seeds_<timestamp>
  --dry-run              Print the scheduled/skipped work without evaluating
EOF
}

need_value() { [[ $# -ge 2 && "$2" != --* ]] || { echo "Missing value for $1" >&2; exit 2; }; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt-root) need_value "$@"; ckpt_root="$2"; shift 2 ;;
    --ckpt) need_value "$@"; single_ckpt="$2"; shift 2 ;;
    --seeds) need_value "$@"; seeds="$2"; shift 2 ;;
    --policy-env) need_value "$@"; policy_env="$2"; shift 2 ;;
    --eval-env) need_value "$@"; eval_env="$2"; shift 2 ;;
    --env-cfg) need_value "$@"; env_cfg="$2"; shift 2 ;;
    --action-type) need_value "$@"; action_type="$2"; shift 2 ;;
    --gpus) need_value "$@"; gpu_ids="$2"; shift 2 ;;
    --eval-num) need_value "$@"; eval_num="$2"; shift 2 ;;
    --only) need_value "$@"; only="$2"; shift 2 ;;
    --run-prefix) need_value "$@"; run_prefix="$2"; shift 2 ;;
    --output-dir) need_value "$@"; output_dir="$2"; shift 2 ;;
    --dry-run) dry_run="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v flock >/dev/null || { echo "flock is required" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq is required to inspect/resume results" >&2; exit 1; }
[[ -f "${POLICY_DIR}/eval.sh" ]] || { echo "Pi_05 eval.sh not found: ${POLICY_DIR}/eval.sh" >&2; exit 1; }
[[ -n "${single_ckpt}" || -d "${ckpt_root}" ]] || { echo "Checkpoint root not found: ${ckpt_root}" >&2; exit 1; }

declare -a policy_gpus isaac_gpus detected_gpus seed_list
if [[ -z "${gpu_ids}" ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required for automatic GPU detection" >&2; exit 1; }
  mapfile -t detected_gpus < <(env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index --format=csv,noheader,nounits | sed 's/[[:space:]]//g')
else
  IFS=',' read -r -a detected_gpus <<< "${gpu_ids}"
fi
[[ ${#detected_gpus[@]} -gt 0 ]] || { echo "No NVIDIA GPUs detected" >&2; exit 1; }
declare -A used_physical wanted
for gpu in "${detected_gpus[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "Invalid physical GPU id '${gpu}'" >&2; exit 2; }
  [[ -z "${used_physical[${gpu}]:-}" ]] || { echo "Duplicate GPU id '${gpu}'" >&2; exit 2; }
  used_physical[${gpu}]=1
  policy_gpus+=("${gpu}")
  isaac_gpus+=("${gpu}")
done

IFS=',' read -r -a seed_list <<< "${seeds}"
[[ ${#seed_list[@]} -gt 0 ]] || { echo "No seeds selected" >&2; exit 2; }
for seed in "${seed_list[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || { echo "Invalid seed '${seed}'" >&2; exit 2; }
done

if [[ -n "${only}" ]]; then
  IFS=',' read -r -a selected <<< "${only}"
  for task in "${selected[@]}"; do wanted["${task}"]=1; done
fi
mapfile -t inventory < <(python3 "${ROOT_DIR}/scripts/internal/task_inventory.py" --only-runnable)
tasks=()
for task in "${inventory[@]}"; do
  if [[ -z "${only}" || -n "${wanted[${task}]:-}" ]]; then
    tasks+=("${task}")
    unset "wanted[${task}]"
  fi
done
[[ -z "${only}" || -z "${wanted[*]-}" ]] || { echo "Unknown/unrunnable task(s): ${!wanted[*]}" >&2; exit 2; }
[[ ${#tasks[@]} -gt 0 ]] || { echo "No tasks selected" >&2; exit 1; }

is_generalization_task() {
  case "$1" in
    stack_bowls|stack_bowls_random|push_T|push_T_random|pack_objects_into_box|pack_objects_into_box_random|fold_clothes|fold_clothes_random|hang_mugs|hang_mugs_random|sweep_blocks|sweep_blocks_random|pour_liquid_into_cup|pour_liquid_into_cup_random|make_toast|make_toast_random|arrange_largest_number|arrange_largest_number_random|sort_nesting_dolls_by_size|sort_nesting_dolls_by_size_random|store_laptop_and_headphones|store_laptop_and_headphones_random|stack_blocks|stack_blocks_random) return 0 ;;
    *) return 1 ;;
  esac
}

episode_target() {
  if [[ "${eval_num}" != "native" ]]; then
    printf '%s\n' "${eval_num}"
  elif is_generalization_task "$1"; then
    printf '25\n'
  else
    printf '50\n'
  fi
}

run_parent() {
  local task="$1" seed="$2" ckpt="$3"
  printf '%s\n' "${ROOT_DIR}/eval_result/RoboDojo/${task}/Pi_05/${env_cfg}/${seed}_ckpt_name=${ckpt},action_type=${action_type}"
}

# Exit 0 if any saved result for this exact task/seed/checkpoint has enough
# numeric layouts. Absolute checkpoint paths intentionally form nested result
# directories, hence the recursive search.
has_complete_result() {
  local parent="$1" target="$2" result count
  [[ -d "${parent}" ]] || return 1
  while IFS= read -r -d '' result; do
    count="$(jq -r 'if (.details | type) == "object" then [.details | keys[] | select(test("^[0-9]+$"))] | length else 0 end' "${result}" 2>/dev/null || printf '0')"
    if [[ "${count}" =~ ^[0-9]+$ ]] && (( count >= target )); then
      return 0
    fi
  done < <(find "${parent}" -type f -name '_result.json' -print0)
  return 1
}

# Preserve all layouts present in a partial _result.json before resuming. This
# repairs a stale manifest after a hard process death and also creates one if
# the result was flushed but the manifest was not.
sync_partial_resume() {
  local parent="$1" run_id="$2" task="$3" seed="$4" ckpt="$5"
  local result="${parent}/${run_id}/_result.json"
  local manifest="${parent}/_resume_${run_id}.json"
  [[ -f "${result}" ]] || return 0
  python3 - "${result}" "${manifest}" "${run_id}" "${task}" "${env_cfg}" "${seed}" "ckpt_name=${ckpt},action_type=${action_type}" <<'PY'
import json
import os
import sys

result_path, manifest_path, run_id, task, config_name, seed, additional_info = sys.argv[1:]
try:
    with open(result_path, encoding="utf-8") as fh:
        result = json.load(fh)
    details = result.get("details") or {}
    if not isinstance(details, dict) or not details:
        raise SystemExit(0)
    with open(manifest_path, encoding="utf-8") as fh:
        previous = json.load(fh)
except FileNotFoundError:
    previous = {}
except (OSError, json.JSONDecodeError):
    previous = {}

def layouts(data):
    return {
        int(value.get("layout_id", key))
        for key, value in data.items()
        if isinstance(value, dict)
        and str(value.get("layout_id", key)).lstrip("-").isdigit()
    }

if len(layouts(details)) <= len(layouts(previous.get("details") or {})):
    raise SystemExit(0)

values = [value for value in details.values() if isinstance(value, dict)]
successes = sum(bool(value.get("success", False)) for value in values)
total_score = sum(float(value.get("score", 0.0) or 0.0) for value in values)
previous.update({
    "run_id": run_id,
    "save_dir": os.path.dirname(result_path),
    "task_name": task,
    "policy_name": "Pi_05",
    "config_name": config_name,
    "eval_seed": int(seed),
    "additional_info": additional_info,
    "success_nums": successes,
    "fail_nums": len(values) - successes,
    "total_score": total_score,
    "completed_layout_ids": sorted(layouts(details)),
    "details": details,
})
os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
tmp_path = manifest_path + ".tmp"
with open(tmp_path, "w", encoding="utf-8") as fh:
    json.dump(previous, fh, indent=2)
os.replace(tmp_path, manifest_path)
print(f"[pi05-parallel] synchronized {len(values)} completed layouts into {manifest_path}")
PY
}

output_dir="${output_dir:-${ROOT_DIR}/eval_runs/${run_prefix}_all_seeds_$(date +%Y-%m-%d_%H-%M-%S)}"
mkdir -p "${output_dir}/logs" "${output_dir}/jax_cache"
queue="${output_dir}/task_queue.tsv"
status="${output_dir}/status.tsv"
lock="${output_dir}/.queue.lock"
status_lock="${output_dir}/.status.lock"
: > "${queue}"
printf 'seed\ttask\tepisodes\tpolicy_gpu\tisaac_gpu\texit_code\tlog\n' > "${status}"

scheduled=0
skipped=0
for seed in "${seed_list[@]}"; do
  if [[ -n "${single_ckpt}" ]]; then
    ckpt="${single_ckpt}"
  else
    ckpt="${ckpt_root}/RoboDojo-sim-arx_x5-joint-${seed}"
  fi
  [[ -d "${ckpt}" ]] || { echo "Missing checkpoint for seed ${seed}: ${ckpt}" >&2; exit 1; }
  for task in "${tasks[@]}"; do
    episodes="$(episode_target "${task}")"
    [[ "${episodes}" =~ ^[0-9]+$ ]] || { echo "Invalid --eval-num '${episodes}'" >&2; exit 2; }
    parent="$(run_parent "${task}" "${seed}" "${ckpt}")"
    if has_complete_result "${parent}" "${episodes}"; then
      echo "[pi05-parallel] SKIP seed=${seed} task=${task}: complete result exists"
      skipped=$((skipped + 1))
      continue
    fi
    # Keep compatibility with the user's earlier seed-0 `pi05_full_<task>` runs.
    # Other seeds use stable, non-colliding names and are resumable on rerun.
    if [[ "${seed}" == "0" ]]; then run_id="${run_prefix}_${task}"; else run_id="${run_prefix}_seed${seed}_${task}"; fi
    if [[ "${dry_run}" != "true" ]]; then
      sync_partial_resume "${parent}" "${run_id}" "${task}" "${seed}" "${ckpt}"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "${seed}" "${task}" "${episodes}" "${ckpt}" "${run_id}" >> "${queue}"
    scheduled=$((scheduled + 1))
  done
done

echo "[pi05-parallel] scheduled=${scheduled} skipped=${skipped} workers=${#policy_gpus[@]} GPUs=${policy_gpus[*]}"
echo "[pi05-parallel] logs/status: ${output_dir}"
if [[ "${dry_run}" == "true" || "${scheduled}" -eq 0 ]]; then
  exit 0
fi

worker() {
  local worker="$1" pgpu="$2" egpu="$3" seed task episodes ckpt run_id log parent rc attempt max_attempts
  exec 9>"${lock}"
  exec 8>"${status_lock}"
  while :; do
    flock 9
    if [[ -s "${queue}" ]]; then
      IFS=$'\t' read -r seed task episodes ckpt run_id < "${queue}" || true
      sed -i '1d' "${queue}"
    else
      seed=""
    fi
    flock -u 9
    [[ -n "${seed}" ]] || break
    parent="$(run_parent "${task}" "${seed}" "${ckpt}")"
    max_attempts="${ROBODOJO_MAX_JOB_ATTEMPTS:-10}"
    [[ "${max_attempts}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid ROBODOJO_MAX_JOB_ATTEMPTS=${max_attempts}" >&2; max_attempts=10; }
    attempt=0
    rc=2
    while (( attempt < max_attempts )); do
      attempt=$((attempt + 1))
      # A new simulator process is the official recovery path for transient
      # PhysX failures.  Preserve every already-counted layout before retrying.
      sync_partial_resume "${parent}" "${run_id}" "${task}" "${seed}" "${ckpt}"
      log="${output_dir}/logs/${task}.seed${seed}.policy${pgpu}.isaac${egpu}.attempt${attempt}.log"
      echo "[worker ${worker}] RUN seed=${seed} task=${task} episodes=${episodes} attempt=${attempt}/${max_attempts} (policy=${pgpu}, isaac=${egpu})"
      set +e
      env -u CUDA_VISIBLE_DEVICES ROBODOJO_ISAAC_GPU_MODE=physical CUROBO_VERSION="${CUROBO_VERSION:-v0.8.0-no-tag}" EVAL_NUM="${episodes}" ROBODOJO_RUN_ID="${run_id}" ROBODOJO_WS_PING_INTERVAL_S=60 ROBODOJO_WS_PING_TIMEOUT_S=600 ROBODOJO_WS_REQUEST_TIMEOUT_S=600 JAX_COMPILATION_CACHE_DIR="${output_dir}/jax_cache/policy_${pgpu}" \
        bash "${POLICY_DIR}/eval.sh" RoboDojo "${task}" "${ckpt}" "${env_cfg}" "${action_type}" "${seed}" "${pgpu}" "${egpu}" "${policy_env}" "${eval_env}" > "${log}" 2>&1
      rc=$?
      set -e
      if has_complete_result "${parent}" "${episodes}"; then
        rc=0
        break
      fi
      # The bundled client may catch an exception and return 0.  Do not let
      # that mask a partial evaluation or prevent its fresh-process retry.
      [[ "${rc}" -eq 0 ]] && rc=2
      if (( attempt < max_attempts )); then
        echo "[worker ${worker}] INCOMPLETE seed=${seed} task=${task}; restarting from saved layouts (attempt ${attempt}/${max_attempts})" >&2
      else
        echo "[worker ${worker}] INCOMPLETE seed=${seed} task=${task}: result has fewer than ${episodes} episodes after ${max_attempts} attempts" >&2
      fi
    done
    flock 8
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${seed}" "${task}" "${episodes}" "${pgpu}" "${egpu}" "${rc}" "${log}" >> "${status}"
    flock -u 8
    [[ "${rc}" -eq 0 ]] && echo "[worker ${worker}] PASS seed=${seed} task=${task}" || echo "[worker ${worker}] FAIL seed=${seed} task=${task}; ${log}" >&2
  done
}

pids=()
trap 'for pid in "${pids[@]:-}"; do kill "${pid}" 2>/dev/null || true; done' INT TERM
for i in "${!policy_gpus[@]}"; do
  worker "$((i + 1))" "${policy_gpus[$i]}" "${isaac_gpus[$i]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}" || true; done
fails="$(awk -F '\t' 'NR > 1 && $6 != 0 {n++} END {print n + 0}' "${status}")"
echo "[pi05-parallel] complete; failures=${fails}; status=${status}"
[[ "${fails}" -eq 0 ]]

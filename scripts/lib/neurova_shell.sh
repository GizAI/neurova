#!/usr/bin/env bash

neurova_q() {
  printf "%q" "$1"
}

neurova_remote_exec() {
  local host="$1"
  local root="$2"
  local env_name="$3"
  shift 3
  local remote_cmd
  remote_cmd="cd $(neurova_q "$root") && source ~/miniconda3/etc/profile.d/conda.sh && conda activate $(neurova_q "$env_name") && $*"
  if [[ "$host" == "local" || "$host" == "$(hostname)" ]]; then
    bash -lc "$remote_cmd"
  else
    ssh "$host" "bash -lc $(neurova_q "$remote_cmd")"
  fi
}

neurova_pick_existing_file_script() {
  local target_var="$1"
  shift
  local script="$target_var='$(neurova_q "${1:-}")'"
  shift || true
  local candidate
  for candidate in "$@"; do
    script+="; if [[ ! -f \"\$$target_var\" && -f $(neurova_q "$candidate") ]]; then $target_var=$(neurova_q "$candidate"); fi"
  done
  printf "%s" "$script"
}

neurova_join_args() {
  local out=""
  local arg
  for arg in "$@"; do
    out+=" $(neurova_q "$arg")"
  done
  printf "%s" "$out"
}

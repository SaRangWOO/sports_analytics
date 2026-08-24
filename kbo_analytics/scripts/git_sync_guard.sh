#!/usr/bin/env bash
set -euo pipefail

sync_main_with_origin() {
  local branch local_head remote_head

  branch="$(git branch --show-current)"
  if [ "$branch" != "main" ]; then
    echo "[Error] KBO automation requires the main branch; current=$branch" >&2
    return 1
  fi

  git fetch origin main
  local_head="$(git rev-parse HEAD)"
  remote_head="$(git rev-parse origin/main)"
  if [ "$local_head" = "$remote_head" ]; then
    return 0
  fi

  if git merge-base --is-ancestor "$local_head" "$remote_head"; then
    git merge --ff-only origin/main
    return 0
  fi

  if git merge-base --is-ancestor "$remote_head" "$local_head"; then
    git push origin main
    git fetch origin main
    test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
    return
  fi

  echo "[Error] local main and origin/main diverged; refusing to generate new outputs" >&2
  return 1
}

push_main_update() {
  git push origin main
}

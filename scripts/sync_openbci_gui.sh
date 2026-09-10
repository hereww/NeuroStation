#!/usr/bin/env bash
set -euo pipefail

action="${1:-sync}"
case "$action" in
  sync|status|overlay) ;;
  *) echo "Usage: $0 [sync|status|overlay]" >&2; exit 2 ;;
esac

script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
project_root="$(CDPATH= cd -- "$script_dir/.." && pwd)"
lock_path="$project_root/integrations/openbci_gui/upstream.lock.json"

repository="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["repository"])' "$lock_path")"
revision="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["revision"])' "$lock_path")"
source_directory="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["source_directory"])' "$lock_path")"
source_path="$project_root/$source_directory"

if [[ "$action" == "overlay" ]]; then
  python3 "$project_root/scripts/apply_openbci_gui_overlay.py" apply --source "$source_path"
  exit $?
fi

get_upstream_remote() {
  for remote in $(git -C "$source_path" remote); do
    if [[ "$(git -C "$source_path" remote get-url "$remote")" == "$repository" ]]; then
      echo "$remote"
      return 0
    fi
  done
  echo "No Git remote points to the locked upstream '$repository'." >&2
  return 1
}

if [[ "$action" == "status" ]]; then
  if [[ ! -d "$source_path/.git" ]]; then
    echo "OpenBCI GUI source is not present. Run: $0 sync" >&2
    exit 1
  fi
  echo "Source: $source_path"
  upstream_remote="$(get_upstream_remote)"
  echo "Upstream remote: $upstream_remote"
  echo "Upstream URL: $(git -C "$source_path" remote get-url "$upstream_remote")"
  echo "Expected revision: $revision"
  echo "Current revision: $(git -C "$source_path" rev-parse HEAD)"
  echo "Branch: $(git -C "$source_path" branch --show-current || true)"
  if [[ -z "$(git -C "$source_path" status --short)" ]]; then
    echo "Clean: true"
  else
    echo "Clean: false"
  fi
  if python3 "$project_root/scripts/apply_openbci_gui_overlay.py" status --source "$source_path" >/dev/null 2>&1; then
    echo "zh-CN overlay applied: true"
  else
    echo "zh-CN overlay applied: false"
  fi
  exit 0
fi

if [[ ! -d "$source_path/.git" ]]; then
  mkdir -p "$(dirname -- "$source_path")"
  git clone --no-checkout "$repository" "$source_path"
  upstream_remote="origin"
else
  upstream_remote="$(get_upstream_remote)"
  if [[ -n "$(git -C "$source_path" status --short)" ]]; then
    echo "OpenBCI GUI source has local changes. Commit or stash them before syncing." >&2
    exit 1
  fi
  current_revision="$(git -C "$source_path" rev-parse HEAD)"
  branch="$(git -C "$source_path" branch --show-current)"
  if [[ "$current_revision" != "$revision" && -n "$branch" ]]; then
    echo "OpenBCI GUI is on branch '$branch'. Switch to detached HEAD before changing the locked revision." >&2
    exit 1
  fi
fi

git -C "$source_path" fetch --depth 1 "$upstream_remote" "$revision"
git -C "$source_path" checkout --detach "$revision"
echo "OpenBCI GUI source is ready at $source_path"
echo "Locked revision: $revision"

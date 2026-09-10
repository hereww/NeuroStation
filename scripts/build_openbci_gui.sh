#!/usr/bin/env sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_command=${PYTHON:-python3}
source_root="$project_root/.vendor/OpenBCI_GUI"

if [ ! -d "$source_root/.git" ]; then
  echo "OpenBCI GUI source is missing. Run scripts/sync_openbci_gui.sh sync first." >&2
  exit 2
fi
"$python_command" "$project_root/scripts/apply_openbci_gui_overlay.py" status

cache_root=${XDG_CACHE_HOME:-"$HOME/.cache"}/neurostation/openbci-build
mkdir -p "$cache_root"
os_name=$(uname -s)
machine=$(uname -m)
case "$os_name" in
  Linux)
    processing_root="$cache_root/processing-4.2"
    processing_command="$processing_root/processing-java"
    case "$machine" in
      x86_64|amd64)
        archive="$cache_root/processing-4.2-linux-x64.tgz"
        url="https://github.com/processing/processing4/releases/download/processing-1292-4.2/processing-4.2-linux-x64.tgz"
        variant="linux-amd64"
        ;;
      aarch64|arm64)
        archive="$cache_root/processing-4.2-linux-arm64.tgz"
        url="https://github.com/processing/processing4/releases/download/processing-1292-4.2/processing-4.2-linux-arm64.tgz"
        variant="linux-arm64"
        ;;
      *)
        echo "Unsupported Linux architecture: $machine" >&2
        exit 2
        ;;
    esac
    runtime_name="linux"
    output_name="application.linux64"
    ;;
  Darwin)
    processing_root="$cache_root/Processing.app"
    processing_command="$processing_root/Contents/MacOS/processing-java"
    case "$machine" in
      x86_64|amd64)
        archive="$cache_root/processing-4.2-macos-x64.zip"
        url="https://github.com/processing/processing4/releases/download/processing-1292-4.2/processing-4.2-macos-x64.zip"
        variant="macos-x86_64"
        ;;
      arm64|aarch64)
        archive="$cache_root/processing-4.2-macos-aarch64.zip"
        url="https://github.com/processing/processing4/releases/download/processing-1292-4.2/processing-4.2-macos-aarch64.zip"
        variant="macos-aarch64"
        ;;
      *)
        echo "Unsupported macOS architecture: $machine" >&2
        exit 2
        ;;
    esac
    runtime_name="macos"
    output_name="application.macosx"
    ;;
  *)
    echo "Unsupported platform: $os_name" >&2
    exit 2
    ;;
esac

if [ ! -x "$processing_command" ]; then
  curl -L --fail --retry 3 -o "$archive" "$url"
  if [ "$os_name" = "Linux" ]; then
    tar -xzf "$archive" -C "$cache_root"
  else
    unzip -q -o "$archive" -d "$cache_root"
  fi
fi

if [ "$os_name" = "Darwin" ]; then
  library_root="$HOME/Documents/Processing/libraries"
else
  library_root="$HOME/sketchbook/libraries"
fi
mkdir -p "$library_root"
cp -R "$source_root/OpenBCI_GUI/libraries/." "$library_root/"

stage_root="$cache_root/stage-$(date +%Y%m%d%H%M%S)-$$"
mkdir -p "$stage_root"
cp -R "$source_root/OpenBCI_GUI" "$stage_root/OpenBCI_GUI"
output="$stage_root/$output_name"
"$processing_command" --force --sketch="$stage_root/OpenBCI_GUI" --output="$output" --variant="$variant" --export

runtime_root="$project_root/integrations/openbci_gui/runtime/$runtime_name"
mkdir -p "$runtime_root"
find "$runtime_root" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
find "$output" -mindepth 1 -maxdepth 1 ! -name source -exec cp -R -- {} "$runtime_root/" \;
echo "OpenBCI GUI export passed: $runtime_root"

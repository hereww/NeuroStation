#!/usr/bin/env sh
set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_command=${PYTHON:-python3}
mode=${MODE:-standalone}
deploy_command=$("$python_command" -c 'import pathlib,sys; print(pathlib.Path(sys.executable).parent / "pyside6-deploy")')

if [ ! -x "$deploy_command" ]; then
  echo "pyside6-deploy is missing beside $python_command. Install requirements.txt first." >&2
  exit 2
fi

cd "$project_root"
"$deploy_command" -c "$project_root/pysidedeploy.spec" --mode "$mode" --force "$@"

if [ "$mode" = "standalone" ]; then
  brainflow_lib=$("$python_command" -c 'import pathlib,brainflow; print(pathlib.Path(brainflow.__file__).resolve().parent / "lib")')
  case "$(uname -s)" in
    Darwin)
      artifact_root="$project_root/dist/NeuroStation.app/Contents/MacOS"
      runtime_name="macos"
      ;;
    Linux)
      artifact_root="$project_root/dist/NeuroStation.dist"
      runtime_name="linux"
      ;;
    *)
      echo "Unsupported standalone platform." >&2
      exit 2
      ;;
  esac
  mkdir -p "$artifact_root/brainflow/lib"
  cp -R "$brainflow_lib/." "$artifact_root/brainflow/lib/"
  openbci_source="$project_root/integrations/openbci_gui/runtime/$runtime_name"
  if [ -d "$openbci_source" ]; then
    openbci_destination="$artifact_root/integrations/openbci_gui/runtime/$runtime_name"
    mkdir -p "$openbci_destination"
    cp -R "$openbci_source/." "$openbci_destination/"
  fi
fi

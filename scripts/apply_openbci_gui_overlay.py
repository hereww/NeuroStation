"""Apply or verify the NeuroStation localization overlay on locked OpenBCI GUI source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "integrations" / "openbci_gui" / "upstream.lock.json"
OVERLAY_ROOT = ROOT / "integrations" / "openbci_gui" / "overlay"

ALLOWED_CHANGES = {
    "OpenBCI_GUI/ControlPanel.pde",
    "OpenBCI_GUI/OpenBCI_GUI.pde",
    "OpenBCI_GUI/TopNav.pde",
    "OpenBCI_GUI/WorkstationI18n.pde",
    "OpenBCI_GUI/data/workstation-i18n/en-US.json",
    "OpenBCI_GUI/data/workstation-i18n/zh-CN.json",
}

REPLACEMENTS: dict[str, tuple[tuple[str, str], ...]] = {
    "OpenBCI_GUI/OpenBCI_GUI.pde": (
        (
            'public final static String stopButton_pressToStop_txt = "Stop Data Stream";\r\n'
            'public final static String stopButton_pressToStart_txt = "Start Data Stream";',
            'public String stopButton_pressToStop_txt = "Stop Data Stream";\r\n'
            'public String stopButton_pressToStart_txt = "Start Data Stream";',
        ),
        (
            "    copyPaste = new CopyPaste();",
            '    copyPaste = new CopyPaste();\r\n\r\n'
            '    setupWorkstationI18n();\r\n'
            '    stopButton_pressToStop_txt = tr("stream.stop");\r\n'
            '    stopButton_pressToStart_txt = tr("stream.start");',
        ),
        ('createFont("fonts/Raleway-SemiBold.otf",', 'createFont(workstationUIFont("fonts/Raleway-SemiBold.otf"),'),
        ('createFont("fonts/Raleway-Regular.otf",', 'createFont(workstationUIFont("fonts/Raleway-Regular.otf"),'),
        ('createFont("fonts/Montserrat-Regular.otf",', 'createFont(workstationUIFont("fonts/Montserrat-Regular.otf"),'),
        ('createFont("fonts/OpenSans-Semibold.ttf",', 'createFont(workstationUIFont("fonts/OpenSans-Semibold.ttf"),'),
        ('createFont("fonts/OpenSans-Regular.ttf",', 'createFont(workstationUIFont("fonts/OpenSans-Regular.ttf"),'),
    ),
    "OpenBCI_GUI/TopNav.pde": (
        ('createControlPanelCollapser("System Control Panel",', 'createControlPanelCollapser(tr("nav.system_control_panel"),'),
        ('createTutorialsButton("Help",', 'createTutorialsButton(tr("nav.help"),'),
        ('createIssuesButton("Issues",', 'createIssuesButton(tr("nav.issues"),'),
        ('createShopButton("Shop",', 'createShopButton(tr("nav.shop"),'),
        ('createUpdateGuiButton("Update",', 'createUpdateGuiButton(tr("nav.update"),'),
        ('createTopNavSettingsButton("Settings",', 'createTopNavSettingsButton(tr("nav.settings"),'),
        ('createFiltersButton("Filters",', 'createFiltersButton(tr("nav.filters"),'),
        ('createLayoutButton("Layout",', 'createLayoutButton(tr("nav.layout"),'),
        ('createExpertModeButton("expertMode", "Turn Expert Mode On",', 'createExpertModeButton("expertMode", tr("settings.expert_on"),'),
        ('createSaveSettingsButton("saveSessionSettings", "Save",', 'createSaveSettingsButton("saveSessionSettings", tr("settings.save"),'),
        ('createLoadSettingsButton("loadSessionSettings", "Load",', 'createLoadSettingsButton("loadSessionSettings", tr("settings.load"),'),
        ('createDefaultSettingsButton("defaultSessionSettings", "Default",', 'createDefaultSettingsButton("defaultSessionSettings", tr("settings.default"),'),
        ('createClearAllSettingsButton("clearAllGUISettings", "Clear All",', 'createClearAllSettingsButton("clearAllGUISettings", tr("settings.clear_all"),'),
        ('createClearSettingsNoButton("clearAllSettingsNo", "No",', 'createClearSettingsNoButton("clearAllSettingsNo", tr("settings.no"),'),
        ('createClearSettingsYesButton("clearAllSettingsYes", "Yes",', 'createClearSettingsYesButton("clearAllSettingsYes", tr("settings.yes"),'),
        ('setText("Turn Expert Mode Off")', 'setText(tr("settings.expert_off"))'),
        ('setText("Turn Expert Mode On")', 'setText(tr("settings.expert_on"))'),
    ),
    "OpenBCI_GUI/ControlPanel.pde": (
        ('text("DATA SOURCE",', 'text(tr("control.data_source"),'),
        ('text("SERIAL CONNECT",', 'text(tr("control.serial_connect"),'),
        ('text("SERIAL/COM PORT",', 'text(tr("control.serial_port"),'),
        ('setText("SEARCHING...")', 'setText(tr("action.searching"))'),
        ('setText("REFRESH LIST")', 'setText(tr("action.refresh"))'),
        ('setText("START SEARCH")', 'setText(tr("action.start_search"))'),
        ('text("BLE DEVICES",', 'text(tr("control.ble_devices"),'),
        ('text("WIFI SHIELDS",', 'text(tr("control.wifi_shields"),'),
        ('text("ENTER IP ADDRESS",', 'text(tr("control.enter_ip"),'),
        ('text("PICK TRANSFER PROTOCOL",', 'text(tr("control.transfer_protocol"),'),
        ('text("SESSION DATA",', 'text(tr("control.session_data"),'),
        ('text("Name",', 'text(tr("control.name"),'),
        ('text("Max File Duration",', 'text(tr("control.max_file_duration"),'),
        ('text("CHANNEL COUNT ",', 'text(tr("control.channel_count") + " ",'),
        ('text("CHANNEL COUNT",', 'text(tr("control.channel_count"),'),
        ('text("SAMPLE RATE ",', 'text(tr("control.sample_rate") + " ",'),
        ('text("PLAYBACK HISTORY",', 'text(tr("control.playback_history"),'),
        ('text("BRAINFLOW STREAMER",', 'text(tr("control.brainflow_streamer"),'),
        ('text("Location",', 'text(tr("control.location"),'),
        ('text("IP",', 'text(tr("control.ip"),'),
        ('text("Port",', 'text(tr("control.port"),'),
        ('text("PLAYBACK FILE",', 'text(tr("control.playback_file"),'),
        ('text("WRITE TO SD CARD?",', 'text(tr("control.write_sd"),'),
        ('text("RADIO CONFIGURATION",', 'text(tr("control.radio_configuration"),'),
    ),
}


def read_preserving_newlines(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return stream.read()


def write_preserving_newlines(path: Path, value: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(value)


def git_output(source: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), *arguments], text=True, encoding="utf-8"
    ).strip()


def changed_paths(source: Path) -> set[str]:
    paths: set[str] = set()
    output = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain=v1", "-uall"],
        text=True,
        encoding="utf-8",
    )
    for line in output.splitlines():
        if line:
            paths.add(line[3:].replace("\\", "/"))
    return paths


def apply_replacements(source: Path) -> None:
    unexpected = changed_paths(source) - ALLOWED_CHANGES
    if unexpected:
        raise RuntimeError("Unrelated OpenBCI GUI changes are present: " + ", ".join(sorted(unexpected)))
    for relative, replacements in REPLACEMENTS.items():
        path = source / relative
        value = read_preserving_newlines(path)
        newline = "\r\n" if "\r\n" in value else "\n"
        value = value.replace("\r\n", "\n")
        for old, new in replacements:
            old = old.replace("\r\n", "\n")
            new = new.replace("\r\n", "\n")
            if new in value:
                continue
            if old not in value:
                raise RuntimeError(f"Overlay anchor not found in {relative}: {old[:70]}")
            value = value.replace(old, new)
        write_preserving_newlines(path, value.replace("\n", newline))
    for relative in (
        "OpenBCI_GUI/WorkstationI18n.pde",
        "OpenBCI_GUI/data/workstation-i18n/en-US.json",
        "OpenBCI_GUI/data/workstation-i18n/zh-CN.json",
    ):
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OVERLAY_ROOT / relative, destination)


def verify(source: Path) -> None:
    for relative, replacements in REPLACEMENTS.items():
        value = read_preserving_newlines(source / relative)
        for _old, new in replacements:
            if new.replace("\r\n", "\n") not in value.replace("\r\n", "\n"):
                raise RuntimeError(f"Overlay replacement is missing in {relative}: {new[:70]}")
    for relative in (
        "OpenBCI_GUI/WorkstationI18n.pde",
        "OpenBCI_GUI/data/workstation-i18n/en-US.json",
        "OpenBCI_GUI/data/workstation-i18n/zh-CN.json",
    ):
        if (source / relative).read_bytes() != (OVERLAY_ROOT / relative).read_bytes():
            raise RuntimeError(f"Overlay file differs: {relative}")
    english = json.loads((source / "OpenBCI_GUI/data/workstation-i18n/en-US.json").read_text(encoding="utf-8"))
    chinese = json.loads((source / "OpenBCI_GUI/data/workstation-i18n/zh-CN.json").read_text(encoding="utf-8"))
    if set(english) != set(chinese):
        raise RuntimeError("OpenBCI GUI English and Chinese catalogs have different keys")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "status"), nargs="?", default="apply")
    parser.add_argument("--source", type=Path)
    arguments = parser.parse_args()
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    source = (arguments.source or ROOT / lock["source_directory"]).resolve()
    if not (source / ".git").exists():
        parser.error(f"OpenBCI GUI Git source is missing: {source}")
    revision = git_output(source, "rev-parse", "HEAD")
    if revision != lock["revision"]:
        parser.error(f"Expected locked revision {lock['revision']}, found {revision}")
    try:
        if arguments.action == "apply":
            apply_replacements(source)
        verify(source)
    except RuntimeError as error:
        print(f"OpenBCI GUI overlay error: {error}", file=sys.stderr)
        return 1
    print(f"OpenBCI GUI zh-CN overlay: applied ({len(ALLOWED_CHANGES)} files)")
    print(f"Source: {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

[app]
title = NeuroStation
project_dir = .
input_file = workstation.py
exec_directory = dist
project_file = pyproject.toml
icon =

[python]
python_path =
packages = Nuitka
android_packages = buildozer==1.5.0,cython==0.29.33

[qt]
qml_files =
excluded_qml_plugins =
modules = Core,Gui,Widgets
plugins = iconengines,imageformats,platforms,platformthemes,styles

[android]
wheel_pyside =
wheel_shiboken =
plugins =

[nuitka]
macos.permissions =
mode = standalone
extra_args = --quiet --assume-yes-for-downloads --product-name=NeuroStation --company-name=NeuroStation --file-version=1.0.3 --product-version=1.0.3 --file-description=NeuroStation_MVP1.0.3_EEG_workstation --include-package=brainflow --include-package=scipy --include-data-file=configs/protocols/ssvep_four_target_v2.json=configs/protocols/ssvep_four_target_v2.json --include-data-file=configs/ssvep_config_v1.json=configs/ssvep_config_v1.json --include-data-file=configs/channel_config_v1_auto.json=configs/channel_config_v1_auto.json --include-data-file=configs/channel_config_v1_template.json=configs/channel_config_v1_template.json --include-data-file=configs/denoise_pipeline_v1.json=configs/denoise_pipeline_v1.json --include-data-file=integrations/openbci_gui/upstream.lock.json=integrations/openbci_gui/upstream.lock.json --include-data-dir=assets=assets --include-data-dir=apps/workstation_ui/locales=apps/workstation_ui/locales

[buildozer]
mode = debug
recipe_dir =
jars_dir =
ndk_path =
sdk_path =
local_libs =
arch =

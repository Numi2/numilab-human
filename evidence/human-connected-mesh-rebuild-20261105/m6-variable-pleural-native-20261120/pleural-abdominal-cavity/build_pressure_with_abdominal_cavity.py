"""Build the existing Matter and Human owners with the staged boundary admission."""
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
build = root / "pressure-with-abdominal-cavity-build"
directory = Path.home() / "Work/skin-fat-native-integration-209/build-native"
for command in json.loads((build / "build-commands.json").read_text()):
    subprocess.run(command, cwd=directory, check=True)
subprocess.run(["/usr/bin/python3", root / "build_floor_scene.py"], check=True)

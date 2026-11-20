"""Bind reference volumes and donor masses to the exact native clearance pose."""
from pathlib import Path
import json
import os
import subprocess
import sys
from time import monotonic

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from build_hand_trial import digest

reference = ROOT / "exact-shell-bone-audit-032/native-evidence/launch.json"
launch = json.loads(reference.read_text())
output = HERE / "compiled-reference"
output.mkdir()
settings = launch["environment"]
cache_key = "NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE"
old_cache = settings[cache_key]
key_digest = launch["pins"].pop(old_cache)
cache = ROOT / "operator-resident-030/initial.mrvpack"
settings[cache_key] = str(cache)
launch["pins"][str(cache)] = key_digest
for path, sha in launch["pins"].items():
    assert digest(path) == sha, path
settings.pop("NUMI_HUMAN_INITIAL_BONE_GEOMETRY_AUDIT")
settings.pop("NUMI_HUMAN_BATCH_CONTACT_WITNESS_CSV")
settings["NUMI_HUMAN_INITIAL_SKIN_FEM_NODES_CSV"] = str(output / "initial-skin-fem.csv")
settings["NUMI_HUMAN_SKIN_FAT_PARENT_MASS_LEDGER_CSV"] = str(output / "shell-mass.csv")
launch["argv"][4] = str(output / "native")
launch["variant"] = "native-rest-frame-and-mass-capture"
launch["comparison_launch_sha256"] = digest(reference)
(output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
environment = {key: value for key, value in os.environ.items()
               if not key.startswith(("NM_", "NUMI_"))}
environment.update(settings)
start = monotonic()
with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
    result = subprocess.run(launch["argv"], cwd=launch["cwd"], env=environment,
                            stdout=out, stderr=err, check=False, timeout=120)
receipt = {"exit": result.returncode, "wall_seconds": monotonic() - start}
(output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
if result.returncode == 0:
    assert digest(output / "initial-skin-fem.csv") == digest(HERE / "inputs/initial-skin-fem.csv")
    for name in ("shell-mass.csv", "shell-mass.csv.nodes.csv", "shell-mass.csv.tets.csv", "shell-mass.csv.source.csv"):
        path = output / name
        receipt[name] = {"size_bytes": path.stat().st_size, "sha256": digest(path)}
    (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
raise SystemExit(result.returncode)

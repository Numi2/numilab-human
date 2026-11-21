"""Cook the pinned registered ligament catalog at its recorded entrance pose."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bake(launch_path, entrance_log, output):
    config = json.loads(launch_path.read_text())
    rigid, bones = map(Path, (config["argv"][1], config["argv"][3]))
    for path in (rigid, bones):
        assert config["pins"][str(path)] == sha(path)
    pose = next(json.loads(s.partition("=")[2]) for s in entrance_log.open()
                if s.startswith("registered_ligament_actual_pose_json="))
    assert pose["bonePayloadSha256"] == sha(bones) == "43a31ef4771e2a383ae2c7caab1b994f0c28076c2a43501860cd1ba3fe7708c0"
    assert not pose["sourceBodyMap"] or pose["sourceBodyMap"] == list(range(len(pose["sourceBodyMap"])))
    coords = "\n".join(" ".join(map(str, row)) for row in pose["bodyPoses"])
    result = subprocess.run(["./cook_catalog", str(bones), str(rigid)], input=coords, text=True,
                            capture_output=True)
    if result.returncode:
        raise RuntimeError(f"catalog cook failed {result.returncode}: {result.stderr}")
    assert len(result.stdout.splitlines()) == 35
    output.write_text(result.stdout)


if __name__ == "__main__":
    bake(*map(Path, sys.argv[1:]))

"""Preserve exact native capture files while reclaiming their physical space."""
from argparse import ArgumentParser
import importlib.util
import json
from pathlib import Path

from build_hand_trial import HERE, digest


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("audit", type=Path)
    args = parser.parse_args()
    trial = args.trial.resolve()
    audit = args.audit.resolve()
    assert trial.parent == HERE and audit.parent == HERE
    assert json.loads((trial/"execution.json").read_text())["stop"]=="exit"
    native = trial/"native"
    script = Path.home()/"storage-cleanup-20261101-transparent-compression-001/compress_authorized_inputs.py"
    spec = importlib.util.spec_from_file_location("verified_compression",script)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.OUT=trial/"compression-receipts"
    helper.CUR_MAN=trial/"launch.json"
    helper.CUR_SPEC=audit
    helper.PIN=trial/"execution.json"
    paths = [
        ("initial-geometry",native/"resting-human.mrvpack"),
        ("registered-skin",trial/"registered-skin-native.csv"),
        ("stdout",trial/"stdout.log"),
    ]
    results=[]
    for label,path in paths:
        response=helper.compress_single(label,path,digest(path))
        results.append({"label":label,"status":response["status"],
                        "allocated_bytes_saved":response.get("allocated_bytes_saved",0),
                        "verified_sha256":digest(path)})
    with (trial/"storage.json").open("x") as stream:
        json.dump(results,stream,indent=2)
        stream.write("\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()

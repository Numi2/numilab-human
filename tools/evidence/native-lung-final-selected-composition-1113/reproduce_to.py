#!/usr/bin/env python3
"""Reproduce the pinned 1113 source composition into a new output directory."""
import hashlib, importlib.util, json, pathlib, subprocess, sys

HERE=pathlib.Path(__file__).resolve().parent
PINS=json.loads((HERE/"source-pins.json").read_text())
COMPOSER=HERE/"compose_candidate_v8.py"

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4<<20),b""): h.update(block)
    return h.hexdigest()

def main():
    if len(sys.argv)!=2:
        raise SystemExit(f"usage: {sys.argv[0]} NEW_OUTPUT_DIRECTORY")
    source=pathlib.Path(PINS["source_checkout"]["path"])
    commit=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(source),"status","--porcelain"],text=True).strip()
    if commit!=PINS["source_checkout"]["commit"] or dirty:
        raise SystemExit("pinned source checkout revision/status mismatch")
    if sha(COMPOSER)!=PINS["composer"]["sha256"]:
        raise SystemExit("pinned composer checksum mismatch")
    for item in PINS["direct_owner_imports"]:
        if sha(item["path"])!=item["sha256"]:
            raise SystemExit(f"pinned owner module checksum mismatch: {item['module']}")
    out=pathlib.Path(sys.argv[1]).resolve()
    if out.exists(): raise SystemExit(f"refusing to overwrite existing output: {out}")
    spec=importlib.util.spec_from_file_location("numi_lung_composer_v8",COMPOSER)
    if spec is None or spec.loader is None: raise SystemExit("cannot load pinned composer")
    composer=importlib.util.module_from_spec(spec); spec.loader.exec_module(composer)
    composer.OUT=out
    composer.main()

if __name__=="__main__": main()

#!/usr/bin/env python3
"""Reproduce the retained native forty-three-surface anatomy increment on the SSH Mini.

The retained owner runtime and source assets are required. This is a launch
adapter, not an asset installer or a physics implementation.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,tempfile
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--prepare-only",action="store_true");args=p.parse_args()
 inventory=json.loads((HERE/"retained-files.json").read_text())
 for relative in ["native/baseline/run-declaration.json","native/launch_arm.py"]:
  if sha(HERE/relative)!=inventory[relative]["sha256"]:raise SystemExit("published launch input changed: "+relative)
 declaration=json.loads((HERE/"native/baseline/run-declaration.json").read_text())
 for path,digest in declaration["immutable_assets"].items():
  if not Path(path).is_file() or sha(path)!=digest:raise SystemExit("retained Mini input missing or changed: "+path)
 destination=Path(tempfile.mkdtemp(prefix="forty-three-surface-native-reproduction-",dir="/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276"))
 (destination/"baseline").mkdir()
 launch=destination/"launch_arm.py";shutil.copyfile(HERE/"native/launch_arm.py",launch)
 old=declaration["owner_cli_preview"]["native_argv"][4];new=str(destination/"baseline/native-run")
 def replace(value):
  if isinstance(value,str):return value.replace(old,new)
  if isinstance(value,list):return [replace(x) for x in value]
  if isinstance(value,dict):return {k:replace(v) for k,v in value.items()}
  return value
 declaration["argv"]=replace(declaration["argv"])
 declaration["owner_cli_preview"]=replace(declaration["owner_cli_preview"])
 declaration["launch_adapter"]={"path":str(launch),"sha256":sha(launch)}
 dp=destination/"baseline/run-declaration.json";dp.write_text(json.dumps(declaration,indent=2,sort_keys=True)+"\n")
 guard={"launcher_sha256":sha(launch),"arms":{"baseline":{"run_declaration_sha256":sha(dp),"native_output":new}}}
 (destination/"launch-guard.json").write_text(json.dumps(guard,indent=2,sort_keys=True)+"\n")
 command=["/usr/bin/python3",str(launch),"baseline"]
 print(json.dumps({"prepared_directory":str(destination),"launch_argv":command,"launch_requested":not args.prepare_only}),flush=True)
 return 0 if args.prepare_only else subprocess.run(command,cwd=destination).returncode
if __name__=="__main__":raise SystemExit(main())

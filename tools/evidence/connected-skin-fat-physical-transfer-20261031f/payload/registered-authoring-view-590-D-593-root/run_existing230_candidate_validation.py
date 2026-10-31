#!/usr/bin/env python3
"""Invoke the exact existing230 validate_candidate owner for one authoring view."""
from __future__ import annotations
import argparse, hashlib, importlib, json, sys, traceback
from pathlib import Path

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidate',required=True,type=Path)
    ap.add_argument('--exporter',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--role',default='D')
    ap.add_argument('--source-commit',default='e962da30')
    args=ap.parse_args()
    source_dir=args.exporter.parent
    sys.path.insert(0,str(source_dir))
    module=importlib.import_module('export_authoring_registered_surface_sidecars_230')
    result={'schema':'existing230-single-candidate-validation.v1','candidate_path':str(args.candidate),
            'candidate_sha256':sha(args.candidate),'role':args.role,'validator_path':str(args.exporter),
            'validator_sha256':sha(args.exporter),'validator_git_commit':args.source_commit,
            'validator_function':'validate_candidate(path, layer, require_f32=True)',
            'validation':'PENDING','native_steps':0,'physical_admission':False}
    try:
        validated=module.validate_candidate(args.candidate,args.role,require_f32=True)
        result.update(validation='PASS',counts={'points':int(validated['vertexCount']),
            'faces':int(validated['faceCount']),'source_original_ids':validated['sourceOriginalFaceIDs'],
            'source_schema_ids':len(validated['sourceSchema']),
            'source_ancestry_ids':sorted(map(int,validated['faceAncestry']))})
        code=0
    except Exception as exc:
        result.update(validation='FAIL',error_type=type(exc).__name__,error=str(exc),
                      traceback=traceback.format_exc())
        code=1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,sort_keys=True))
    return code

if __name__=='__main__': raise SystemExit(main())

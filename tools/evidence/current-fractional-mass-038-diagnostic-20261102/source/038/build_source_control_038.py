#!/usr/bin/env python3
"""Build a pinned 030d source-control wrapper and include closure for canonical 038."""
from __future__ import annotations
import argparse, hashlib, json, re, shlex
from pathlib import Path

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
NATIVE = ROOT / "current-fractional-semantic-native-1047"
TRIAL = NATIVE / "optimization-trial-030d-contiguous-preference-lookup"
P029 = NATIVE / "optimization-trial-029-canonical-source-tet-sites"
D030C = NATIVE / "optimization-trial-030c-gzip-fullcut-witness-closed-include"
VARIANT = "030d-contiguous-preference-lookup"
CONTROL_SCHEMA = "numi.h706.current-fractional-cli-controls.v1"
CLOSURE_SCHEMA = "numi.h706.allocator-source-closure.v1"
NOT_RUN = {"method":"implicit_dinic_feasibility_on_exact_residual_graph","status":"NOT_RUN_FULL_ALLOCATOR","optimality_certified":False,"scale_factor":1.0,"postsolve_rescaling":False}
PINNED = {
 "030d_source_control":"07a6585b8de55629d7433b66d7a6d9e7beb79234df10cdd0c7c13c8a60850f04",
 "030d_compile_control":"73f2832e9a1671231a07584a6ce380baa35ad6d102be83f6e27abb3501134197",
 "030d_focused":"32eaaad8c045aa423fffd3712b0322d7554f12f264e502751b3fe8ece5549d59",
 "030d_carrier":"61da25945b181a8cd395cec11ead521c78d99c0a410adffd5bfdaea1c0c65b2b",
 "030d_flow":"2b4bd3eaa26b0ca3f7c7826c8f84fd1ce631918dcb61690258ec8c42699b1ac2",
 "030d_semantic":"21328dd75135cef43d1aff019a9eadda022cf56132800cb152593a7717a8c996",
 "030d_gzip":"a75468c5fd6590609f070c621e910c758551d3de0643a068b27e0e43bf381b0d",
 "030d_storage":"2c3298857abaadfff552eaee52eb8fac515b9f9e4c556ab50e5b93ad5b6b4be1",
 "030c_control":"2dde5dbc5a9c100e1c1149f689837c7dd2a5dcf546f208a22849d0050a5e2f3d",
 "029_control":"e21af9025b6073ff981bdb13cd2f4ca6796fb75c2f4df9d730d852cc5edc8929",
}
class SourceControlError(ValueError): pass

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def pin(path: Path, expected: str|None=None) -> dict:
    path=path.resolve(strict=True)
    if not path.is_file(): raise SourceControlError(f"not a file: {path}")
    ref={"path":str(path),"sha256":sha(path),"bytes":path.stat().st_size}
    if expected and ref["sha256"]!=expected: raise SourceControlError(f"frozen input SHA changed: {path}")
    return ref

def read(path: Path) -> dict:
    x=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(x,dict): raise SourceControlError(f"expected JSON object: {path}")
    return x

def check_ref(ref: dict,label: str) -> Path:
    if not isinstance(ref,dict) or set(ref) not in ({"path","sha256"},{"path","sha256","bytes"}):
        raise SourceControlError(f"{label} pin schema mismatch")
    p=Path(ref["path"])
    if not p.is_absolute() or not p.is_file() or sha(p)!=ref["sha256"]:
        raise SourceControlError(f"{label} path/SHA mismatch: {p}")
    if "bytes" in ref and (type(ref["bytes"]) is not int or ref["bytes"]!=p.stat().st_size):
        raise SourceControlError(f"{label} byte count mismatch")
    return p.resolve()

def depfile_paths(path: Path) -> set[Path]:
    text=path.read_text(encoding="utf-8").replace("\\\n"," ")
    if ":" not in text: raise SourceControlError("depfile target separator missing")
    return {Path(t).resolve(strict=True) for t in shlex.split(text.split(":",1)[1]) if t}

def compiler_contract(argv: list[str]) -> dict:
    fixed=["/usr/bin/clang++","-std=c++20","-O3","-Wall","-Wextra","-Werror","-DNUMI_HUMAN_FULL_REFERENCE_SKIN_CAPACITY=1"]
    if not isinstance(argv,list) or argv[:7]!=fixed or "-MMD" not in argv or "-MF" not in argv or "-lz" not in argv or "-o" not in argv:
        raise SourceControlError("030d compiler/flags differ from frozen source build")
    dirs=[x[2:] for x in argv if x.startswith("-I")]
    if len(dirs)!=5 or dirs[0]!=str(TRIAL): raise SourceControlError("030d include-directory order mismatch")
    if Path(argv[argv.index("-lz")-1]).name!="NumiHumanCurrentFractionalMassMappingCLI_030d.cpp":
        raise SourceControlError("030d source identity mismatch")
    return {"command":argv[0],"standard":"c++20","optimization":"-O3","warnings":["-Wall","-Wextra","-Werror"],
            "defines":["NUMI_HUMAN_FULL_REFERENCE_SKIN_CAPACITY=1"],"include_directories":dirs,"libraries":["z"]}

def build(out_dir: Path) -> tuple[dict,dict]:
    c=TRIAL/"controls"
    paths={
      "030d_source_control":c/"source-control-manifest-030d.json",
      "030d_compile_control":c/"compile-control-030d.json",
      "030d_focused":c/"focused-controls-030d.json",
      "030d_carrier":c/"source-carrier-parity-report.json",
      "030d_flow":c/"lookup-flow-parity-report.json",
      "030d_semantic":c/"lookup-semantic-parity-report.json",
      "030d_gzip":c/"gzip-writer-control-030d.json",
      "030d_storage":c/"lookup-storage-benchmark-receipt-030d.json",
      "030c_control":D030C/"source-control-manifest-030c.json",
      "029_control":P029/"source-control-manifest-029.json"}
    refs={k:pin(v,PINNED[k]) for k,v in paths.items()}
    objs={k:read(v) for k,v in paths.items()}
    raw,cc=objs["030d_source_control"],objs["030d_compile_control"]
    focused,carrier,flow=objs["030d_focused"],objs["030d_carrier"],objs["030d_flow"]
    semantic,gzip,storage=objs["030d_semantic"],objs["030d_gzip"],objs["030d_storage"]
    upstream,parent=objs["030c_control"],objs["029_control"]
    if raw.get("variant")!=VARIANT or raw.get("scope")!="source-only performance candidate; no full allocator run":
        raise SourceControlError("raw 030d manifest scope/variant mismatch")
    if cc.get("variant")!=VARIANT or cc.get("status")!="PASS_COMPILED_SOURCE_ONLY_NOT_RUN":
        raise SourceControlError("030d compile control is not compile-only")
    if cc.get("source_control_manifest")!={"path":refs["030d_source_control"]["path"],"sha256":refs["030d_source_control"]["sha256"]}:
        raise SourceControlError("compile control does not bind raw 030d source-control")
    if focused.get("status")!="PASS_EXACT_PARITY" or focused.get("variant_pair")!=["030c frozen baseline","030d contiguous preference lookup"]:
        raise SourceControlError("030d focused parity is not the frozen 030c comparison")
    if carrier.get("status")!="PASS_SOURCE_CARRIER_PARITY": raise SourceControlError("source-carrier parity failed")
    if flow.get("status")!="PASS_EXACT_FLOW_PARITY" or flow.get("byte_identical_stdout") is not True or flow.get("byte_identical_stderr") is not True:
        raise SourceControlError("flow parity is not byte-identical")
    if semantic.get("status")!="PASS_EXACT_SEMANTIC_SUPPORT_PARITY" or semantic.get("byte_identical_stdout") is not True or semantic.get("byte_identical_stderr") is not True:
        raise SourceControlError("semantic parity is not byte-identical")
    if gzip.get("status")!="PASS_EXACT_GZIP_ROUNDTRIP" or storage.get("status")!="PASS_EXACT_SCORE_PARITY":
        raise SourceControlError("gzip/lookup controls are not passing")
    if upstream.get("schema")!="numi.h706.current-fractional-cli-source-control-030c.v1" or upstream.get("status")!="PASS_SOURCE_CONTROL_COMPILE_ONLY_FULL_ALLOCATOR_NOT_RUN":
        raise SourceControlError("030c upstream lineage must remain compile-only")
    if parent.get("variant")!="029-canonical-source-tet-sites" or parent.get("solver_contract")!=NOT_RUN:
        raise SourceControlError("P029 source-control lineage is invalid")
    source,binary,support=raw["source"],raw["binary"],raw["optimized_header"]
    depfile=raw["dependency_file"]
    for name,ref in (("source",source),("binary",binary),("support",support),("depfile",depfile)): check_ref(ref,"030d "+name)
    if Path(source["path"]).name!="NumiHumanCurrentFractionalMassMappingCLI_030d.cpp" or Path(binary["path"]).name!="current-fractional-mass-mapping-cli-030d-o3":
        raise SourceControlError("raw source/binary path is not 030d")
    if Path(support["path"]).name!="ConservativeSpatialFractionalSemanticSupport.hpp":
        raise SourceControlError("optimized support header name mismatch")
    deps=raw.get("dependencies")
    if not isinstance(deps,list) or raw.get("dependency_count")!=len(deps) or len(deps)!=32:
        raise SourceControlError("030d dependency count mismatch")
    actual=depfile_paths(Path(depfile["path"]))
    declared=set()
    for i,ref in enumerate(deps): check_ref(ref,f"030d dependency {i}"); declared.add(Path(ref["path"]).resolve())
    if len(declared)!=len(deps) or declared!=actual: raise SourceControlError("depfile/dependency pins differ")
    mass_header=next((x for x in deps if Path(x["path"]).name=="NumiHumanPhysicalSkinChildMassMapping.hpp"),None)
    if mass_header is None: raise SourceControlError("mass-transport header missing from depfile")
    argv=cc.get("compile_command")
    compiler=compiler_contract(argv)
    compile_command=shlex.join(argv)
    if raw.get("compile_command")!=argv or cc.get("source")!=source or cc.get("binary")!=binary or cc.get("optimized_header")!=support or cc.get("dependency_file")!=depfile:
        raise SourceControlError("raw source-control and compile-control differ")
    if cc.get("stdout",{}).get("sha256")!=hashlib.sha256(b"").hexdigest() or cc.get("stderr",{}).get("sha256")!=hashlib.sha256(b"").hexdigest():
        raise SourceControlError("compile-only logs are not empty")
    for log in (cc["stdout"],cc["stderr"]):
        check_ref(log,"030d compile log")
        if Path(log["path"]).read_bytes(): raise SourceControlError("compile-only logs must be empty")
    rows=[
      {"source":pin(Path(carrier["source"]["path"])),"binary":pin(Path(carrier["variants"]["030d_candidate"]["binary_path"])),"status":"PASS"},
      {"source":pin(Path(flow["fixture"]["path"])),"binary":pin(Path(flow["variants"]["030d_candidate"]["binary_path"])),"status":"PASS"},
      {"source":pin(Path(semantic["fixture"]["path"])),"binary":pin(Path(semantic["variants"]["030d_candidate"]["binary_path"])),"status":"PASS"},
      {"source":pin(Path(gzip["source"]["path"])),"binary":pin(Path(gzip["variants"]["030d_candidate"]["binary"]["path"])),"status":"PASS"},
      {"name":"production-cli-o3-compile","source":source,"binary":binary,"compile_command":compile_command,"status":"PASS_COMPILE_ONLY"}]
    out_dir.mkdir(parents=True,exist_ok=True)
    sc=out_dir/"source-control-manifest-038.json"; cl=out_dir/"source-closure-038.json"
    if sc.exists() or cl.exists(): raise SourceControlError("refusing to overwrite 038 source-control outputs")
    parent_refs=[refs[k] for k in ("029_control","030d_source_control","030d_compile_control","030d_focused","030d_carrier","030d_flow","030d_semantic","030d_gzip","030d_storage","030c_control")]
    control={"schema":CONTROL_SCHEMA,"status":"PASS_SOURCE_CONTROLS_FULL_ALLOCATOR_NOT_RUN","variant":VARIANT,
      "source":source,"support_header":support,"mass_transport_header":mass_header,"binary":binary,
      "admission_adapter":parent["admission_adapter"],"source_controls_006":parent["source_controls_006"],
      "compiler":compiler,"solver_contract":dict(NOT_RUN),"focused_controls":rows,
      "receipt_builder":parent["receipt_builder"],"receipt_builder_test":parent["receipt_builder_test"],
      "compile_command":compile_command,"parent_manifests":parent_refs,"threshold_policy":parent["threshold_policy"]}
    sc.write_text(json.dumps(control,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    closure={"schema":CLOSURE_SCHEMA,"variant":VARIANT,"source_control_manifest":pin(sc),
      "parent_source_control_manifest":refs["030d_source_control"],"compiler":compiler,"compile_command":compile_command,
      "sources":[{"path":str(Path(x["path"]).resolve(strict=True)),
                  "sha256":x["sha256"],"bytes":x["bytes"]} for x in deps]}
    cl.write_text(json.dumps(closure,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return control,closure

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",required=True,type=Path)
    a=p.parse_args()
    try: build(a.output_dir.resolve())
    except Exception as e: p.error(str(e))
    print(f"source_control_038=PASS_SOURCE_CONTROLS_FULL_ALLOCATOR_NOT_RUN output={a.output_dir.resolve()}")
    return 0
if __name__=="__main__": raise SystemExit(main())

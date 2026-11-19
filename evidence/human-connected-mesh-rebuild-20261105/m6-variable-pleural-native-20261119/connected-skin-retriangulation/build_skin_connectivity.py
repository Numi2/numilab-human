"""Admit one audited payload identity while retaining the source thickness gate."""
import json
import os
from pathlib import Path
import subprocess
import sys

from register_tissue_bellies import B, HERE, LATE, hash_file

sys.path.insert(0,str(B.parent/"resting-clearance-and-atrium-004"))
from archive_bed_preflight import archive

FAILED=HERE/"rejected-unregistered-connectivity"
BASE=B/"trimmed-guarded-pressure-skin-build"
OUT=HERE/"connected-skin-build"


def main():
    initial=HERE/"connected-skin-native"
    if initial.exists() and not FAILED.exists():
        result=json.loads((initial/"execution.json").read_text())
        assert result["exit_code"]==1
        assert "regional thickness requires an exact detailed skin source" in (initial/"stderr.log").read_text()
        initial.rename(FAILED)
        n=FAILED/"native"
        # Keep the original path recoverable in the preserved startup trial.
        for name in ("skin-source-influences.bin","resting-human.mrvpack"):
            archive(n/name)
    proof=json.loads((HERE/"connected-skin-diagonals.json").read_text())
    candidate=HERE/"connected-skin-diagonals.nhskin"
    assert proof["candidate_steps"] and proof["introduced_pairs"]==0
    assert hash_file(candidate)==proof["candidate_sha256"]
    assert proof["protected_source_vertices"]==9588
    launch=json.loads((LATE/"launch.json").read_text())
    assert hash_file(BASE/"numi-human-native")==launch["pins"][str(BASE/"numi-human-native")]
    OUT.mkdir()
    visual="NumiHumanRestingVisual.hpp"
    for f in [*BASE.glob("*.hpp"),BASE/"numilab_human_myosim_visual_probe.mm"]:
        if f.name!=visual:
            os.link(f,OUT/f.name)
    old=(BASE/visual).read_text()
    expected=proof["source_sha256"]
    before=f'retainedSourceSkinSHA256 == "{expected}") &&'
    after=f'retainedSourceSkinSHA256 == "{expected}" ||\n                     retainedSourceSkinSHA256 == "{proof["candidate_sha256"]}") &&'
    assert old.count(before)==1
    (OUT/visual).write_text(old.replace(before,after))
    cmds=json.loads((BASE/"build-commands.json").read_text())
    paths=("numilab_human_myosim_visual_probe.mm","numi-human-native.o","numi-human-native")
    mapping={str(BASE/name):str(OUT/name) for name in paths}
    prepared=[[mapping.get(p,p) for p in row] for row in cmds]
    (OUT/"build-commands.json").write_text(json.dumps(prepared,indent=2)+"\n")
    cwd=Path("/Users/numi/Work/skin-fat-native-integration-209/build-native")
    for argv in prepared:
        subprocess.run(argv,cwd=cwd,check=True)
    receipt={str(p.relative_to(OUT)):hash_file(p) for p in (OUT/visual,OUT/"numi-human-native",OUT/"build-commands.json")}
    receipt["same_runtime_object_sha256"]=hash_file(BASE/"runtime.mm.o")
    (OUT/"build-verification.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(receipt,flush=True)


if __name__=="__main__":
    main()

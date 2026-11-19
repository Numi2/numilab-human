"""Record the actual section contributors on the verified resting skin."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/"resting-clearance-and-atrium-006"
RUN=PREVIOUS/"source-clearance-patches-native"
BASE=PREVIOUS/"source-clearance-patches-build"
BUILD=HERE/"bed-dependency-build"
APP="numi-human-native"
HEADER="NumiHumanRestingVisual.hpp"
UNIT="numilab_human_myosim_visual_probe.mm"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    with path.open("x") as output:
        json.dump(value,output,indent=2)
        output.write("\n")


def build():
    BUILD.mkdir()
    for f in [*BASE.glob("*.hpp"),BASE/UNIT]:
        if f.name!=HEADER:
            os.link(f,BUILD/f.name)
    inp=BASE/HEADER
    assert digest(inp)==read(BASE/"build-verification.json")[HEADER]
    s=inp.read_text()
    match="for (unsigned v = 0; v < posedSkin.size(); ++v) {\n                    if (tributaryAreas[v] == 0) continue;"
    assert s.count(match)==1
    s=s.replace(match,"std::vector<unsigned> chosenThicknessVertex(posedSkin.size(), MR_INVALID_INDEX);\n                "+match)
    match="skinThicknessByRow[row] += tributaryAreas[v]"
    assert s.count(match)==1
    s=s.replace(match,"chosenThicknessVertex[v] = kWholeBodySkinSourceVertices[nearest];\n                    "+match)
    match="                auto* uploadedMaps = static_cast<MRHumanRestingVertexMap*>(mapping.contents);"
    assert s.count(match)==1
    s=s.replace(match,r'''                if (std::getenv("NUMI_HUMAN_TEST_BED_SECTION_FOOTPRINT")) {
                    std::ofstream footprint(outputDirectory / "resting-bed-section-footprint.csv");
                    footprint << "vertex,region,x,y,z,region_min,tributary_area,nearest_thickness_vertex\n"
                              << std::setprecision(17);
                    for (unsigned v=0; v<posedSkin.size(); ++v) {
                        const auto row=regionForVertex[v];
                        footprint << v << ',' << row << ',' << posedSkin[v].x << ',' << posedSkin[v].y
                                  << ',' << posedSkin[v].z << ',' << patchMinimum[row] << ','
                                  << tributaryAreas[v] << ',' << chosenThicknessVertex[v] << '\n';
                    }
                    require(bool(footprint), "bed section footprint receipt write failed");
                }
'''+match)
    (BUILD/HEADER).write_text(s)
    m={str(BASE/f):str(BUILD/f) for f in (APP,APP+".o",UNIT)}
    cmds=[[m.get(a,a) for a in cmd] for cmd in read(BASE/"build-commands.json")]
    write(BUILD/"build-commands.json",cmds)
    for cmd in cmds:
        subprocess.run(cmd,cwd="/Users/numi/Work/skin-fat-native-integration-209/build-native",check=True)
    (BUILD/"metalrobo").mkdir()
    for f in (BASE/"metalrobo").iterdir():
        os.link(f,BUILD/"metalrobo"/f.name)


def native(binary,skin,out):
    out.mkdir()
    cfg=read(RUN/"launch.json")
    args=cfg["argv"]
    pos=args.index("--skin-payload")+1
    args[0],args[4],args[pos]=str(binary),str(out/"native"),str(skin)
    args[args.index("--muscle-step-count")+1]="1"
    e=cfg["environment"]
    for k in ("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS","NUMI_HUMAN_RESTING_CAPTURE_REUSE_SOURCE_DIR"):
        e.pop(k,None)
    e.update(NUMI_HUMAN_TEST_BED_SECTION_FOOTPRINT="1",
             NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=str(out/"failure.json"))
    pins={path:sha for path,sha in cfg["pins"].items() if path not in
          (str(BASE/APP),str(PREVIOUS/"source-clearance-patches.nhskin"))}
    pins.update({str(path):digest(path) for path in (binary,skin,*sorted((binary.parent/"metalrobo").iterdir()))})
    for path,sha in pins.items():
        assert digest(Path(path))==sha,path
    # Initial pack is generated from the actual candidate source. Only the
    # duplicate control is eligible for the existing exact-byte startup cache.
    if skin==PREVIOUS/"source-clearance-patches.nhskin":
        proof=read(RUN/"native/resting-human.mrvpack.zst.recovery.json")
        e["NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256"]=proof["sha256"]
        cached=True
    else:
        cached=False
    with tempfile.TemporaryDirectory(prefix="dependency-preflight-",dir=HERE) as temp:
        start=Path(temp)/"resting-human.mrvpack"
        if cached:
            with start.open("wb") as file:
                subprocess.run(["/opt/homebrew/bin/zstd","-dc",RUN/"native/resting-human.mrvpack.zst"],
                               stdout=file,check=True)
            assert digest(start)==proof["sha256"]
            e["NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE"]=str(start)
            pins[str(start)]=proof["sha256"]
        cfg["pins"]=pins
        write(out/"launch.json",cfg)
        env={k:v for k,v in os.environ.items() if not k.startswith(("NUMI_","NM_"))}
        with (out/"stdout.log").open("wb") as stdout,(out/"stderr.log").open("wb") as stderr:
            p=subprocess.run(args,cwd=cfg["cwd"],env=env|e,stdout=stdout,stderr=stderr,timeout=95)
        write(out/"execution.json",dict(exit=p.returncode,requested_steps=1,variable_pleural_pressure=True))
        p.check_returncode()
        file=out/"native/resting-human.mrvpack"
        if cached:
            assert digest(file)==proof["sha256"]
            compressed=out/"native/resting-human.mrvpack.zst"
            os.link(RUN/"native/resting-human.mrvpack.zst",compressed)
            assert digest(compressed)==digest(RUN/"native/resting-human.mrvpack.zst")
            proof["restore_command"]=["/opt/homebrew/bin/zstd","-d",str(compressed),"-o",str(file)]
            write(out/"native/resting-human.mrvpack.zst.recovery.json",proof)
            file.unlink()
    print(digest(out/"native/resting-bed-section-footprint.csv"),flush=True)


if __name__=="__main__":
    build()
    native(BUILD/APP,PREVIOUS/"source-clearance-patches.nhskin",HERE/"bed-dependency-control")

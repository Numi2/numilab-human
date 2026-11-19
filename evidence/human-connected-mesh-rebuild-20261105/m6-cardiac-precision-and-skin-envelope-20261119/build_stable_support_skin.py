"""Retain source-row region identity across the guarded NHSKIN position repair."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/"resting-clearance-and-atrium-004"
sys.path.insert(0,str(PREVIOUS))
from run_segment_control import PRIOR, digest

def main():
    kind=os.environ.get("NUMI_HUMAN_TRIAL_SKIN_SELF","1")
    assert kind in ("1","2")
    variant="skin-self-native" if kind=="1" else "skin-live-pinned-native"
    package=HERE/(variant+".nhskin")
    receipt=json.loads((HERE/(variant+"-payload.json")).read_text())
    candidate_hash=receipt["candidate_sha256"]
    assert digest(package)==candidate_hash and not receipt["new_source_pairs"]
    footprint=PREVIOUS/"bed-preflight-baseline/native/resting-bed-section-footprint.csv"
    reference=json.loads((PREVIOUS/"bed-section-pins.json").read_text())
    assert digest(footprint)==reference["base_footprint_sha256"]
    with footprint.open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==61950
    assert all(int(row["vertex"])==i for i,row in enumerate(rows))
    regions=bytes(int(row["region"]) for row in rows)
    assert set(regions)==set(range(32))
    work=HERE/("skin-stable-support-build" if kind=="1" else "skin-live-pinned-build")
    work.mkdir()
    position=work/"source-region-map.bin"
    position.write_bytes(regions)
    for file in ("NumiHumanAorticDiscretePreload.hpp","NumiHumanLocalBedLayers.hpp",
                 "NumiHumanRestingCoupling.hpp","NumiHumanRestingTransactionProbe.hpp",
                 "human_resting_runtime.hpp","numilab_human_myosim_visual_probe.mm","runtime.mm.o","build.py"):
        os.link(PRIOR/file,work/file)
    for directory in ("include","metalrobo"):
        (work/directory).symlink_to(PRIOR/directory,target_is_directory=True)
    source=PRIOR/"NumiHumanRestingVisual.hpp"
    data=source.read_text()
    old='retainedSourceSkinSHA256 == "00709724af1ae4684f057255e0f98297add7e83671d3e229f040a3d91daeb1c8"'
    assert data.count(old)==1
    data=data.replace(old,f'retainedSourceSkinSHA256 == "{candidate_hash}"')
    find="        if(deriveFootManifold) {\n            for(unsigned parentIndex="
    assert data.count(find)==1
    init=f'''
        // Recovery fixture only. Hold material-region IDs across a topology
        // preserving position edit while the contact kernel still sees every
        // reconstructed candidate position and live contact reaction.
        require(!organSupportActive&&!deriveFootManifold&&!skin.usesRegisteredDirectWorldSurface&&
                sourceRegionCount==32u&&skin.header.vertexCount==61950u&&
                retainedSourceSkinSHA256=="{candidate_hash}",
                "stable support region fixture received an unexpected skin variant");
        {{
            NSData* canonical=[NSData dataWithContentsOfFile:@"{position}"];
            require(canonical!=nil&&canonical.length==regionForVertex.size()&&
                    loadedKneeSHA256Hex(loadedKneeSHA256(
                        static_cast<const std::uint8_t*>(canonical.bytes),canonical.length))==
                    "{digest(position)}",
                    "retained source skin region map differs from the original compiled footprint");
            const auto* region=static_cast<const std::uint8_t*>(canonical.bytes);
            const auto regionSize=regionForVertex.size();
            unsigned reassigned=0;
            for(unsigned i=0; i<regionSize; ++i) {{
                require(region[i]<sourceRegionCount,
                        "retained skin contact region points outside the source owner table");
                reassigned+=unsigned(region[i]!=regionForVertex[i]);
                --regionCounts[regionForVertex[i]];
                regionForVertex[i]=region[i];
                ++regionCounts[regionForVertex[i]];
            }}
            std::cout<<"test_retained_source_support_regions reassigned="<<reassigned<<"\\n";
        }}
'''
    (work/source.name).write_text(data.replace(find,init+find))
    (work/"support-map-contract.json").write_text(json.dumps(
        dict(candidate_sha256=candidate_hash,reference_footprint_sha256=digest(footprint),
             region_map_sha256=digest(position),rows=len(regions),region_count=32,
             geometry_remains_live=True),indent=2)+"\n")
    subprocess.run([sys.executable,str(work/"build.py")],check=True)

if __name__=="__main__":
    main()

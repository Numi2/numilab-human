"""Bind the exact guarded skin candidate to the existing bed compatibility check."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

from run_segment_control import HERE, PRIOR, digest

def main():
    output="guarded-footprint" if os.environ.get("NUMI_AUTHOR_BED_FOOTPRINT")=="1" else "guarded-skin"
    package=HERE/(output+".nhskin")
    receipt=json.loads((HERE/(output+"-payload.json")).read_text())
    key=receipt["candidate_sha256"]
    assert digest(package)==key and receipt["support_anchors_byte_equal"]==32
    assert not receipt["new_source_pairs"]
    work=HERE/(output+"-build")
    work.mkdir()
    for file in ("NumiHumanAorticDiscretePreload.hpp","NumiHumanLocalBedLayers.hpp",
                 "NumiHumanRestingCoupling.hpp","NumiHumanRestingTransactionProbe.hpp",
                 "human_resting_runtime.hpp","numilab_human_myosim_visual_probe.mm","runtime.mm.o","build.py"):
        os.link(PRIOR/file,work/file)
    for directory in ("include","metalrobo"):
        (work/directory).symlink_to(PRIOR/directory,target_is_directory=True)
    source=PRIOR/"NumiHumanRestingVisual.hpp"
    old='retainedSourceSkinSHA256 == "00709724af1ae4684f057255e0f98297add7e83671d3e229f040a3d91daeb1c8"'
    data=source.read_text()
    assert data.count(old)==1
    (work/source.name).write_text(data.replace(old,f'retainedSourceSkinSHA256 == "{key}"'))
    subprocess.run([sys.executable,str(work/"build.py")],check=True)

if __name__=="__main__":
    main()

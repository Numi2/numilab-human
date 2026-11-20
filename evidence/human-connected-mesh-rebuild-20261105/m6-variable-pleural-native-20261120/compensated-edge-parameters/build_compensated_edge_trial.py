"""Use the same compensated closest segment parameters in contact and barrier evaluation."""
from pathlib import Path
import difflib
import json
import os
import subprocess

here=Path.cwd()
source=here/"edge-barrier-derivative"
destination=here/"edge-barrier-compensated-parameters"
(destination/"src").mkdir(parents=True)
for original in (source/"src").rglob("*"):
    path=destination/original.relative_to(source)
    if original.is_dir():
        path.mkdir(exist_ok=True,parents=True)
    elif original.is_file():
        os.link(original.resolve(),path)
name="src/metal/contact.metalinc"
before=(source/name).read_text()
start=before.index("// Keep the closest-point parameters produced")
after=before.index("inline bool contactCompensatedSegmentWitness(\n    const float3 p0",start)
patched=before[:start]+(here/"compensated-edge-parameters.inc").read_text()+before[after:]
(destination/name).unlink()
(destination/name).write_text(patched)
(here/"compensated-edge-parameters.patch").write_text("".join(difflib.unified_diff(
    before.splitlines(True),patched.splitlines(True),"a/matter/"+name,"b/matter/"+name)))
commands=json.loads((source/"build-commands.json").read_text())
commands=[[x.replace(str(source)+"/",str(destination)+"/") for x in command] for command in commands]
(destination/"build-commands.json").write_text(json.dumps(commands,indent=2)+"\n")
with (here/"compensated-edge-parameters-build.log").open("wb") as log:
    for command in commands:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
Path(commands[0][-1]).unlink()
print(destination/"NumiMatter.metallib")

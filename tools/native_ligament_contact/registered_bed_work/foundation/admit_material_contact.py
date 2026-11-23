"""A material contact is active exactly when the existing quadrature carries pressure."""
from pathlib import Path
import json,subprocess,sys
from export_contact_columns import change
from material_context import sha
def build(owner):
    ids=json.loads((owner/'identities.json').read_text())
    if any(sha(owner/k)!=d for k,d in ids.items()):raise ValueError('changed material owner')
    p=owner/'NumiHumanStand.metal'; s=p.read_text()
    s=change(s,'''    // An open unilateral column carries no traction in the current linearization.
    if (support.planePoint.z == -2.0f && support.planePoint.w > 0.0f && support.planePoint.x == 0.0f)
        return -INFINITY;''','''    // Each material point has already tested its actual closure. The pressure
    // centroid is a virtual-work application point and can be above the plane.
    if (support.planePoint.z == -2.0f && support.planePoint.w > 0.0f)
        return support.planePoint.x == 0.0f ? -INFINITY : INFINITY;''')
    p.write_text(s)
    cmds=json.loads((owner/'shader-build-commands.json').read_text());old=Path(cmds[0][cmds[0].index('-c')+1]).parent
    cmds=[[v.replace(str(old)+'/',str(owner)+'/') for v in c] for c in cmds]
    for cmd in cmds:subprocess.run(cmd,check=True)
    (owner/'shader-build-commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
    (owner/'identities.json').write_text(json.dumps({k:sha(owner/k) for k in ids},indent=2)+'\n')
if __name__=='__main__':build(Path(sys.argv[1]).resolve())

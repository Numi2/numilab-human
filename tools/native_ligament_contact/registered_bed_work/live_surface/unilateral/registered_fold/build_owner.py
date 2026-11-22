"""Compile the authenticated loaded state into the same full native Human owner."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from supported_material_admission import certify, installed_topology, stress_limits
from material_context import sha


def build(root):
    def read(p):return json.loads(p.read_text())
    parent=root.parent/'feasible-bed'
    case=root/'material';old=parent/'owner';dest=root/'owner'
    meta=read(case/'material-identity.json');ident=read(old/'identities.json')
    x,y,t,tag,pins=inputs(parent/'two')
    xx,yy=[np.loadtxt(case/n,skiprows=1,usecols=(1,2,3)) for n in ('reference.node','current.node')]
    rows=np.asarray(meta['rows']);delta=np.array(meta['moves_m'])
    if pins!=meta['source'] or not np.array_equal(xx[rows]-x[rows],delta) or not np.array_equal(yy[rows]-y[rows],delta):
        raise ValueError('Material state differs from emitted repair')
    if not np.array_equal(np.flatnonzero((xx!=x).any(1)),rows) or not np.array_equal(np.flatnonzero((yy!=y).any(1)),rows):
        raise ValueError('Unrelated points changed')
    if np.min(y[rows,2])<=0.01:
        raise ValueError('The measured outer fold overlaps the registered bed footprint')
    baseline=read(parent/'two/launch.json')
    t,tag=installed_topology(t,tag,baseline['environment'])
    props,hashes=materials(old)
    admission=certify(xx,yy,t,tag,props,stress_limits(hashes))
    if not admission['passed']:raise ValueError(admission)
    header='NumiHumanLoadedSkinPoseData.inc'
    main='numilab_human_myosim_visual_probe.mm'
    loader=ident['loaded_pose_override']
    if sha(old/header)!=loader['header_sha256'] or sha(old/main)!=ident['source'] or sha(case/'mesh/mesh.node')!=meta['reference'] or sha(case/'current.node')!=meta['current']:
        raise ValueError('Unmatched native source or geometry pins')
    subprocess.run(['cp','-cR',str(old),str(dest)],check=True)
    text=(dest/header).read_text()
    for before,after in [(pins['installed_pose_sha256'],meta['current']),(pins['reference_sha256'],meta['reference'])]:
        if text.count(before)!=1:raise ValueError('Ambiguous physical digest')
        text=text.replace(before,after)
    (dest/header).write_text(text)
    compile_cmd,link_cmd=ident['commands']
    old_compile=Path(compile_cmd[compile_cmd.index('-c')+1]).parent
    if sha(old_compile/main)!=sha(old/main) or sha(old_compile/header)!=sha(old/header):
        raise ValueError('Compilation inputs diverged since parent link')
    cmds=[[v.replace(str(prefix)+'/',str(dest)+'/') for v in c] for prefix,c in [(old_compile,compile_cmd),(old,link_cmd)]]
    for c in cmds:subprocess.run(c,check=True,cwd=baseline['cwd'])
    exe=dest/'numi-human-native';binary=sha(exe)
    ident.update(binary=binary,binary_sha256=binary,binary_path=str(exe),commands=cmds,installed_tissue_admission=admission,
                 generated_header=sha(dest/header))
    ident['loaded_pose_override']=dict(loader,path=str(case/'current.node'),sha256=meta['current'],header_sha256=sha(dest/header),
            immediate_parent_sha256=pins['installed_pose_sha256'],fit_sha256=meta['fit_sha256'],nodes=rows.tolist())
    (dest/'identities.json').write_text(json.dumps(ident,indent=2)+'\n')
    (root/'admission.json').write_text(json.dumps(dict(binary=binary,input=meta,material=admission),indent=2)+'\n')


if __name__=='__main__':build(Path(sys.argv[1]).resolve())

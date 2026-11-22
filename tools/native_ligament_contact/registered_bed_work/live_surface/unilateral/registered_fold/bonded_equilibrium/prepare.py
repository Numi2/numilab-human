"""Install a checked, loaded tissue configuration into the full release-certified Human owner."""
from pathlib import Path
import csv
import json
import re
import os
import subprocess
import sys
import numpy as np
from emit_constrained_material import write_coordinates
from fit_dorsal_material import inputs
from material_context import sha
from supported_material_admission import check_installed

def read(p):return json.loads(p.read_text())
def prepare(root,reader,predicate,release_witness=False):
    out=root/'bonded-equilibrium';old=root/'material-direction/release/owner';dest=out/'owner'
    if release_witness:
        out=out/'free-interface';dest=out/'owner'
    run=root/'fold-repair/accepted-2';launch=read(run/'launch.json');env=launch['environment']
    prior=read(old/'identities.json');fit=read(out/'fit.json');x,y,src,tags,pins=inputs(run)
    if pins!=fit['source'] or not fit['moved'] or fit['emitted']['free_force_l2_N']>=fit['initial']['free_force_l2_N']:raise ValueError('Unqualified mechanical start')
    with np.load(out/'preload.npz') as z:rows=z['rows'];values=z['coordinates']
    if not np.array_equal(rows,np.unique(rows)) or len(rows)!=fit['moved'] or np.any((rows<0)|(rows>=len(y))) or values.shape!=(len(rows),3) or values.dtype!=np.float32:raise ValueError('Incomplete tissue update')
    prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    path=Path(env[prefix+'COMBINED_NODE_MAP_CSV'])
    if sha(path)!=fit['registered_map']:raise ValueError('Changed physical interface')
    wanted=set(map(int,rows));seen=set()
    with path.open() as f:
        for v in csv.DictReader(f):
            n=int(v['joint_node_id'])-1
            if n in wanted:
                if v['registered_surface_layer'] not in ('','I'):raise ValueError('A physical boundary would move')
                seen.add(n)
    header='NumiHumanLoadedSkinPoseData.inc';main='numilab_human_myosim_visual_probe.mm'
    body=(old/header).read_text();anchors=re.search(r'constexpr Anchor anchors\[\]=\{(.*?)\};',body,re.S)
    protected=set(map(int,re.findall(r'\{\s*\d+u?\s*,\s*(\d+)u?\s*,',anchors[1])))
    if len(protected)!=14 or wanted&protected or wanted!=seen:raise ValueError('Attachment or ancestry mismatch')
    prop=Path(env['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ'])
    with np.load(prop) as z:
        fixed=np.unique(np.r_[z['replacement_tetrahedra'].ravel(),src[np.setdiff1d(np.arange(len(src)),z['retained_input_cell_rows'])].ravel()])
    affected=np.intersect1d(rows,fixed)
    if release_witness:
        target=read(root/'material-direction/release/result.json')['source_cell']
        if not np.array_equal(affected,np.sort(src[target])):raise ValueError('Released vertices differ from registered witness')
    elif len(affected):raise ValueError('Cavity/mass-replacement rows moved')
    original=Path(env['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE'])
    write_coordinates(original,out/'current.node',rows,values)
    cur=np.loadtxt(out/'current.node',skiprows=1,usecols=(1,2,3))
    if not np.array_equal(cur[rows],values) or not np.array_equal(np.flatnonzero((cur!=y).any(1)),rows):raise ValueError('Emitted geometry differs')
    admission=check_installed(x,cur,src,tags,env,old)
    if not admission['passed']:raise ValueError(admission)
    subprocess.run([sys.executable,str(reader),str(run/'launch.json'),str(predicate),str(out/'layers.json'),'--accepted-nodes',str(run/'physical-reference.node'),str(out/'current.node')],check=True,env=dict(os.environ,PYTHONPATH=str(predicate.parent)+os.pathsep+os.environ.get('PYTHONPATH','')))
    if np.any(read(out/'layers.json')['pair_counts']):raise ValueError('Initial tissue layers intersect')
    if not dest.exists():subprocess.run(['cp','-cR',str(old),str(dest)],check=True)
    loader=prior['loaded_pose_override'];oldhash=sha(original);newhash=sha(out/'current.node')
    if sha(old/header)!=loader['header_sha256'] or oldhash!=loader['sha256'] or body.count(oldhash)!=1:raise ValueError('Different registered pose owner')
    (dest/header).write_text(body.replace(oldhash,newhash))
    compile_cmd,link_cmd=prior['commands'];source=Path(compile_cmd[compile_cmd.index('-c')+1]).parent
    if sha(source/header)!=sha(old/header) or sha(source/main)!=sha(old/main):raise ValueError('Stale source during build')
    commands=[[v.replace(str(prefix)+'/',str(dest)+'/') for v in c] for prefix,c in [(source,compile_cmd),(old,link_cmd)]]
    if release_witness:
        earlier=root/'material-direction/release/build'
        qualified=root/'bonded-equilibrium/bed-stop/paired/build'
        commands[1]=[v.replace(str(earlier)+'/',str(qualified)+'/') for v in commands[1]]
    for cmd in commands:subprocess.run(cmd,cwd=launch['cwd'],check=True)
    exe=dest/'numi-human-native';binary=sha(exe)
    prior.update(binary=binary,binary_sha256=binary,binary_path=str(exe),commands=commands,generated_header=sha(dest/header),installed_tissue_admission=admission,parent_pose_owner_sha256=sha(old/'numi-human-native'))
    prior['loaded_pose_override']=dict(loader,path=str(out/'current.node'),sha256=newhash,header_sha256=sha(dest/header),immediate_parent_sha256=oldhash,fit_sha256=sha(out/'fit.json'),nodes=rows.tolist())
    (dest/'identities.json').write_text(json.dumps(prior,indent=2)+'\n')
    (out/'preparation.json').write_text(json.dumps(dict(material=admission,initial_layers=sha(out/'layers.json'),fit=sha(out/'fit.json'),loaded_pose=newhash,source_pose=oldhash,native=binary,protected_attachments=len(protected),reference=pins['reference_sha256']),indent=2)+'\n')
if __name__=='__main__':prepare(*(Path(v).resolve() for v in sys.argv[1:4]),release_witness='--free-interface' in sys.argv[4:])

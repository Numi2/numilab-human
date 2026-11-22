"""Bind a regenerated source volume, mass ledger and starting pose to the existing native scene."""
import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
from fit_dorsal_material import inputs
from material_context import sha
from relax_bonded_interface import loaded_mass_moments

def bind(native,parent,case,proposal,old_flow,owner):
    conf=json.loads((native/'launch.json').read_text())
    trial=json.loads((case/'qualified-parent/launch.json').read_text())
    fit=json.loads((case/'material-identity.json').read_text())
    reg=json.loads((case/'material/registration/receipt.json').read_text())
    reader=json.loads((case/'material/native-role-verification.json').read_text())
    x,y,t,tag,pins=inputs(native)
    if (fit['native_source']!=pins or sha(proposal)!=conf['environment']['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256']
        or not reader['passed'] or sha(case/'proposal.npz')!=reg['proposal_sha256'] or reg['proposal_sha256']!=fit['proposal']
        or trial['connected_material_volume']['role_reader_sha256']!=sha(case/'material/native-role-verification.json')):
        raise ValueError('Unmatched production readers, material or cavity geometry')
    X,Y=[np.loadtxt(case/f,skiprows=1,usecols=(1,2,3)) for f in ('mesh/mesh.node','current.node')]
    if (sha(case/'current.node')!=fit['current'] or sha(case/'mesh/mesh.node')!=fit['reference']
        or trial['environment']['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE_SHA256']!=fit['reference']):
        raise ValueError('Unmatched emitted coordinates')
    delta=X-x.astype(float)
    if not np.array_equal(Y-y.astype(float),delta):raise ValueError('Loaded and reference material edits differ')
    ids=np.flatnonzero(np.any(delta!=0,1))
    explicit=np.zeros_like(delta)
    for region in fit['regions']:explicit[region['rows'],region['axis']]=region['movement_m']
    if not np.array_equal(explicit,delta):raise ValueError('An unregistered point moved')
    mreport=json.loads((case/'fine-mass-routed/mass-transfer.json').read_text())
    flow=case/'fine-mass-routed/current-density-closed-flow.npz'
    if (mreport['flow_sha256']!=sha(flow) or mreport['parent_flow_sha256']!=sha(old_flow) or mreport['density_outliers']
        or mreport['maximum_carrier_mass_change_kg']>2e-17 or not mreport['archived_compensation_unchanged']):raise ValueError('Mass assignment failed')
    with np.load(old_flow) as f:before=f['element_mass_kg']
    with np.load(flow) as f:after=f['element_mass_kg']
    with np.load(proposal) as f:
        ret=np.setdiff1d(np.arange(len(t)),f['retained_input_cell_rows'])
        if np.any(before[ret]!=after[ret]) or np.isin(t[ret],ids).any():raise ValueError('Existing replacement cavity changed')
        moments=[loaded_mass_moments(s,t[ret],after[ret],f['replacement_tetrahedra'],f['replacement_mass_kg']) for s in (X,Y)]
    if not all(r['passed'] for r in moments):raise ValueError('Existing cavity mass moments failed')
    donor=np.loadtxt(case/'mesh/current-child.labels.csv',delimiter=',',skiprows=1,usecols=2,dtype='i4')
    affected=(before!=after)|np.any(np.isin(t,ids),axis=1);results=[]
    for d in np.unique(donor[affected]):
        a=np.flatnonzero(affected&(donor==d));old=before[a];new=after[a]
        r=dict(donor=int(d),mass_delta_kg=float(sum(new-old)),moment_deltas_kg_m=[])
        for original,current in ((x,X),(y,Y)):
            u=original[t[a]].astype(float).mean(1);v=current[t[a]].mean(1)
            r['moment_deltas_kg_m'].append(((new-old)@u+new@(v-u)).tolist())
        if abs(r['mass_delta_kg'])>1e-10 or np.max(abs(np.array(r['moment_deltas_kg_m'])))>1e-10:raise ValueError('Material donor mass or first moment changed')
        results.append(r)
    oldmanifest=json.loads((owner/'identities.json').read_text())
    if conf['argv'][0]!=str(owner/'numi-human-native') or sha(owner/'numi-human-native')!=oldmanifest['binary_sha256']:
        raise ValueError('Wrong compiled owner')
    dest=case/'native-owner'
    if dest.exists():raise ValueError('Output owner already exists')
    subprocess.run(['cp','-cR',str(owner),str(dest)],check=True)
    header='NumiHumanLoadedSkinPoseData.inc';src=(dest/header).read_text()
    anchors=[int(a) for a in re.findall(r'\{\s*\d+u?\s*,\s*(\d+)u?\s*,',re.search(r'constexpr Anchor anchors\[\]=\{(.*?)\};',src,re.S)[1])]
    if len(anchors)!=14 or np.isin(ids,anchors).any():raise ValueError('Physical attachment changed')
    for old,new in ((pins['installed_pose_sha256'],fit['current']),(pins['reference_sha256'],fit['reference'])):
        if src.count(old)!=1:raise ValueError('Unknown installed material')
        src=src.replace(old,new)
    (dest/header).write_text(src)
    cmds=[[v.replace(str(owner)+'/',str(dest)+'/') for v in cmd] for cmd in oldmanifest['commands']]
    for cmd in cmds:subprocess.run(cmd,check=True,cwd=conf['cwd'])
    key='NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE';old=trial['pins'][trial['environment'][key]]
    if old!=oldmanifest['loaded_pose_override']['parent_sha256']:raise ValueError('Changed launch ancestor')
    exe=dest/'numi-human-native';pin=sha(exe)
    oldmanifest.update(commands=cmds,binary=pin,binary_sha256=pin,binary_path=str(exe),parent_pose_owner_sha256=sha(owner/'numi-human-native'),
        loaded_pose_override=dict(key=key,parent_sha256=old,path=str(case/'current.node'),sha256=fit['current'],header_sha256=sha(dest/header),nodes=ids.tolist()))
    (dest/'identities.json').write_text(json.dumps(oldmanifest,indent=2)+'\n')
    (case/'material-bind.json').write_text(json.dumps(dict(source=pins,changes=fit['regions'],production_role_reader=sha(case/'material/native-role-verification.json'),
        cavity_moments=moments,regional_mass=results,matched_anchor_count=len(anchors),parent_launch=sha(parent/'launch.json'),native_binary=pin),indent=2)+'\n')
if __name__=='__main__':bind(*(Path(a).resolve() for a in sys.argv[1:]))

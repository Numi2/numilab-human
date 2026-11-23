"""Feed the solved positive transfers through the existing native feasibility reader and role writer."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path
import numpy as np
from register_material_surfaces import sha


def run(launch,mesh,flows,solver_result,owner,out):
    e=json.loads(launch.read_bytes())['environment']
    r=json.loads((mesh/'receipt.json').read_bytes())
    verified=json.loads((flows/'exact-audit.json').read_bytes())
    solved=json.loads(solver_result.read_bytes())['materials']
    out.mkdir()
    lab=np.loadtxt(mesh/'mesh.ele',skiprows=1,usecols=5,dtype='i4')
    n=len(lab);carrier=[];targets=[];flow=[];source=[];total=np.zeros(n);start=0;augment=0
    for tag,mat in (('SKIN',15),('FAT',16)):
        file=flows/(tag+'-physical-flow.npz')
        if not verified[tag]['verified'] or sha(file)!=verified[tag]['flow_sha256']:raise ValueError('Unverified positive flow')
        f=np.load(file)
        k='NUMI_H706_FRACTIONAL_CARRIERS_'+tag
        packet=Path(e[k+'_NPZ'])
        if sha(packet)!=e[k+'_SHA256']:raise ValueError('Wrong source packet')
        p=np.load(packet)
        hist=np.repeat(p['global_source330_tet_rows'],np.diff(p['parent_offsets']))
        source.append(hist)
        q,g,w=(f[name] for name in ('carrier','role_cell','mass'))
        role_rows=np.flatnonzero(lab==mat)
        carrier.append(start+q);targets.append(role_rows[g]);flow.append(w)
        total[role_rows]=np.bincount(g,weights=w,minlength=len(role_rows))
        start+=len(hist)
        for line in solved[tag]['solver']:
            augment+=int(re.search(r'augmentations=(\d+)',line).group(1))
    if np.any(total<=0) or augment<=0:raise ValueError('Incomplete native flow')
    packet=out/'grouped.npz'
    np.savez_compressed(packet,carrier_row=np.concatenate(carrier).astype('i4'),group=np.concatenate(targets).astype('i4'),
                        group_for_cell=np.arange(n,dtype='i4'),source_parent=np.concatenate(source).astype('i8'),
                        mass_flow_kg=np.concatenate(flow),element_mass_kg=total)
    state=json.loads(Path(e['NUMI_H706_FRACTIONAL_P1_SKIN_NPZ']).parents[1].joinpath('current-paired-volume-state.json').read_bytes())
    src=state['source330_identity']
    for k in ('mass','labels','tets','nodes'):
        v=src['source330_'+k]
        if sha(Path(v['path']))!=v['sha256']:raise ValueError('Unmatched historical carrier geometry')
    exe=owner/'validate_complete_fine_p1'
    pre='NUMI_H706_FRACTIONAL_'
    cmd=[str(exe),src['source330_mass']['path'],src['source330_labels']['path'],str(mesh/'mesh.ele'),
         e[pre+'STATE_SHA256'],src['source330_mass']['sha256']]
    for tag in ('SKIN','FAT'):cmd.extend(e[pre+k] for k in ('P1_'+tag+'_NPZ','P1_'+tag+'_SHA256','GEOMETRY_'+tag+'_SHA256'))
    cmd.extend([str(n),str(mesh/'mesh.node'),src['source330_tets']['path'],src['source330_nodes']['path']])
    for tag in ('SKIN','FAT'):cmd.extend((e[pre+'CARRIERS_'+tag+'_NPZ'],e[pre+'CARRIERS_'+tag+'_SHA256']))
    for tag in ('SKIN','FAT'):cmd.extend((e[pre+'CARRIER_WITNESS_'+tag+'_CSV'],e[pre+'CARRIER_WITNESS_'+tag+'_SHA256']))
    labels=mesh/'current-child.labels.csv'
    cmd.extend([str(r['radius_m']),src['source330_tets']['sha256'],src['source330_nodes']['sha256'],str(labels),
                sha(labels),str(r['body_count']),r['frame'],str(packet),sha(packet),str(augment)])
    art=out/'role-flows';art.mkdir()
    env=dict(os.environ,NUMI_HUMAN_NATIVE_ROLE_FLOW_OUTPUT=str(art),NUMI_HUMAN_NATIVE_REFERENCE_VOLUME_OUTPUT=str(out/'native-volume.f64'))
    ident=dict(command=cmd,binary=sha(exe),sha256={k:sha(owner/k) for k in ('validate_complete_fine_p1.cpp','replay_grouped_paired_flow.hpp')},
               accounted_augmentations=augment,solver_result_sha256=sha(solver_result))
    (out/'launch.json').write_text(json.dumps(ident,indent=2)+'\n')
    with (out/'stdout.log').open('w') as f,(out/'stderr.log').open('w') as err:
        result=subprocess.run(cmd,env=env,stdout=f,stderr=err,check=False)
    stats=dict(exit=result.returncode,receipts={p.name:sha(p) for p in art.glob('*recorded-flow-receipt.json')})
    (out/'result.json').write_text(json.dumps(stats,indent=2)+'\n')
    return result.returncode


if __name__=='__main__':sys.exit(run(*(Path(arg).resolve() for arg in sys.argv[1:])))

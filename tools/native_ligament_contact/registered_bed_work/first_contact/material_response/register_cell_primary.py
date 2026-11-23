"""Register the primary cell label to the integrated historical corner support on a repaired mesh."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from register_material_surfaces import pinned,sha,tetgen_positions


def run(launch,mesh,support,packets,flow,out):
    env=json.loads(launch.read_bytes())['environment']
    rec=json.loads((mesh/'receipt.json').read_bytes())
    boundary=min(i['first_cell'] for i in rec['regions'])
    src=pinned(env,'PARENT_LABELS_CSV','PARENT_LABELS_SHA256')
    parent=np.loadtxt(src,dtype='i8',delimiter=',',skiprows=1)
    labels=np.loadtxt(mesh/'current-child.labels.csv',dtype='i8',delimiter=',',skiprows=1)
    old=labels.copy()
    out.mkdir()
    report={}
    xyz=tetgen_positions(mesh/'mesh.node').astype('f4').astype('f8')
    tet=np.loadtxt(mesh/'mesh.ele',dtype='i4',skiprows=1,usecols=(1,2,3,4))-1
    ref=[]
    for source_key,cols,dtype in [('NODES',(2,3,4),'f8'),('TETS',(3,4,5,6),'i4')]:
        key='NUMI_H706_FRACTIONAL_SOURCE330_'+source_key
        path=Path(env[key])
        if sha(path)!=env[key+'_SHA256']:raise ValueError('Changed historical mesh')
        ref.append(np.loadtxt(path,dtype=dtype,delimiter=',',skiprows=1,usecols=cols))
    centers=ref[0][ref[1]].mean(axis=1)
    audit=json.loads((flow/'exact-audit.json').read_bytes())
    with (out/'ownership-source.csv').open('w') as stream:
        wr=csv.writer(stream,lineterminator='\n')
        wr.writerow(['cell','old_donor','old_region','old_face','donor','region','face','P1_numerator','P1_denominator',
                     'equal_max_groups','matched_parent_rows','source_face_ids','actual_mass_from_selected_group_kg'])
        for mat,tag in ((15,'SKIN'),(16,'FAT')):
            rows=np.flatnonzero(labels[:,1]==mat)
            sr=json.loads((support/(tag.lower()+'-p1-receipt.json')).read_bytes())
            pf=Path(sr['p1_one_hop_path']);ff=flow/(tag+'-physical-flow.npz')
            if sha(pf)!=sr['p1_one_hop_sha256'] or sha(ff)!=audit[tag]['flow_sha256'] or not audit[tag]['verified']:
                raise ValueError('Changed or unsupported mass provenance')
            a=np.load(pf)
            tab=a['group_tuples'];off=a['cell_group_offsets'];groups=a['cell_group_ids'];num=a['cell_group_p1_score_numerators'];den=a['cell_group_p1_score_denominators']
            po,pr,fo,fr=(a[k] for k in ('cell_group_parent_tet_offsets','cell_group_parent_tet_rows','cell_group_stable_face_offsets','cell_group_stable_face_rows'))
            maxnum=np.maximum.reduceat(num,off[:-1]);cell=np.repeat(np.arange(len(rows)),np.diff(off))
            eq=np.all(tab[groups]==labels[rows[cell],1:4],axis=1)
            bad=np.flatnonzero(np.maximum.reduceat(np.where(eq,num,0),off[:-1])!=maxnum)
            if (rows[bad]<boundary).any() or np.any(den!=den[off[:-1]][cell]):raise ValueError('Unexpected material preference domain')
            data=np.load(ff);flowc,flowq,floww=(data[k] for k in ('role_cell','carrier','mass'))
            pack=np.load(packets/(tag.lower()+'-carriers.npz'))
            qparent=np.repeat(pack['global_source330_tet_rows'],np.diff(pack['parent_offsets']))
            total_ties=0;zero_flow=0
            for i in bad:
                gi=rows[i];slots=np.arange(off[i],off[i+1]);top=slots[num[slots]==maxnum[i]]
                if len(top)>1:total_ties+=1
                l,h=np.searchsorted(flowc,[i,i+1])
                src_tuples=parent[qparent[flowq[l:h]],1:4]
                center=xyz[tet[gi]].mean(axis=0)
                choices=[]
                for slot in top:
                    hist=pr[po[slot]:po[slot+1]];faces=fr[fo[slot]:fo[slot+1]]
                    group=groups[slot];tu=tab[group]
                    if not len(hist) or not np.all(parent[hist,1:4]==tu) or not np.array_equal(np.unique(parent[hist,4]),faces):
                        raise ValueError('Projected source donor/face lacks historical ancestry')
                    kg=float(floww[l:h][np.all(src_tuples==tu,axis=1)].sum())
                    dist=float(np.linalg.norm(centers[hist]-center,axis=1).min())
                    # The existing P1 compiler records multi-face support using a
                    # derived group identity. Every contributing face is retained below.
                    face=int(faces[0]) if len(faces)==1 else 0x80000000+int(group)
                    choices.append((-kg,dist,int(group),slot,face))
                negmass,_,group,slot,face=min(choices)
                zero_flow+=negmass==0
                labels[gi,2:4]=tab[group][1:];labels[gi,4]=face
                wr.writerow([gi,*old[gi,2:],*labels[gi,2:],num[slot],den[slot],len(top),
                    ' '.join(map(str,pr[po[slot]:po[slot+1]])),' '.join(map(str,fr[fo[slot]:fo[slot+1]])),-negmass])
            report[tag]=dict(rebound=len(bad),tied_maximum=total_ties,without_positive_mass_from_primary=zero_flow)
    dest=out/'current-child.labels.csv'
    with dest.open('w') as stream:
        with (mesh/dest.name).open() as srcfile:stream.write(srcfile.readline())
        np.savetxt(stream,labels,delimiter=',',fmt='%d')
    for n in ('mesh.node','mesh.ele','material-source-indices.npz','cell-mass.f64','node-mass.f64'):os.link(mesh/n,out/n)
    for e in rec['entries']:
        if sha(mesh/e['file'])!=e['sha256']:raise ValueError('Changed original tissue registration')
        e['sha256']=sha(out/e['file'])
    rec['prior_label_sha256']=sha(mesh/dest.name)
    rec['p1_cell_primary_ancestry']=sha(out/'ownership-source.csv')
    (out/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n')
    native=json.loads((mesh/'native-import-build.json').read_bytes())
    if sha(Path(native['launch'][0]))!=native['binary']:raise ValueError('Changed native importer')
    cmd=native['launch'].copy();cmd[1]=str(out)
    check=subprocess.run(cmd,text=True,capture_output=True)
    if check.returncode:raise ValueError(check.stderr or check.stdout)
    (out/'native-import.json').write_text(check.stdout)
    (out/'primary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(report)


if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))

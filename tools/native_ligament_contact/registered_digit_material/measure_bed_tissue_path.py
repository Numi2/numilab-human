"""Inspect which registered muscle volumes lie between the bed-facing skin and cortex on accepted support rays."""
import csv
import json
import runpy
import subprocess
import sys
from pathlib import Path
import numpy as np
from material_context import sha

def hits(points, triangles, samples, require_oriented=False):
    tri=points[triangles].astype(float)
    lo,hi=tri[:,:,:2].min(1),tri[:,:,:2].max(1)
    out={}
    for key,xy in samples.items():
        idx=np.flatnonzero(((lo<=xy)&(xy<=hi)).all(1))
        if not len(idx):continue
        a,b,c=tri[idx].transpose(1,0,2)
        u,v=b-a,c-a
        d=xy-a[:,:2]
        det=u[:,0]*v[:,1]-u[:,1]*v[:,0]
        scale=abs(u[:,0]*v[:,1])+abs(u[:,1]*v[:,0])
        ok=abs(det)>64*np.finfo(float).eps*scale
        # A ray lying on a projected degenerate edge has ambiguous occupancy.
        if (~ok).any():
            line=np.where((np.linalg.norm(u[~ok,:2],axis=1)>np.linalg.norm(v[~ok,:2],axis=1))[:,None],u[~ok,:2],v[~ok,:2])
            if np.any(np.abs(d[~ok,0]*line[:,1]-d[~ok,1]*line[:,0])<1e-15):
                out[key]=None
                continue
        p=(d[ok,0]*v[ok,1]-d[ok,1]*v[ok,0])/det[ok]
        q=(u[ok,0]*d[ok,1]-u[ok,1]*d[ok,0])/det[ok]
        bary=np.column_stack((1-p-q,p,q))
        inside=bary.min(1)>=-1.e-10
        if not inside.any():continue
        if np.any(bary[inside].min(1)<=1.e-10):
            out[key]=None
        else:
            z=(bary[inside]*tri[idx[ok][inside],:,2]).sum(1)
            order=np.argsort(z)
            sign=np.sign(det[ok][inside][order])
            ambiguous=(np.diff(z[order])<1e-10).any() or (require_oriented and (len(sign)%2 or (sign[1:]==sign[:-1]).any()))
            out[key]=None if ambiguous else z[order].tolist()
    return out

def closed(points,tris):
    _,remap=np.unique(points,axis=0,return_inverse=True)
    t=remap[tris]
    e=np.concatenate((t[:,[0,1]],t[:,[1,2]],t[:,[2,0]]))
    key,edge,counts=np.unique(np.sort(e,axis=1),axis=0,return_inverse=True,return_counts=True)
        # A sealed seam must have exactly two opposing edge orientations.
    balance=np.bincount(edge,weights=np.sign(e[:,1]-e[:,0]),minlength=len(key))
    return bool((key[:,0]!=key[:,1]).all() and (counts==2).all() and (balance==0).all())

def union_length(segments):
    end=-np.inf;length=0.
    for lo,hi in sorted(segments):
        if hi>max(lo,end):length+=hi-max(lo,end)
        end=max(end,hi)
    return length

def measure(run,reader):
    import hashlib
    rec=json.loads((run/'terminal.receipt.json').read_text())
    launch=json.loads((run/'launch.json').read_text());argv=launch['argv']
    pin=json.loads((run/'frame-recovery.json').read_text())
    step=rec['accepted_step'];dt=float(argv[argv.index('--muscle-step-seconds')+1])
    if rec['physical_endpoint']!='accepted' or abs(step*dt-rec['accepted_time_s'])>5e-6:
        raise ValueError('Inconsistent accepted clock')
    files=['resting-support-point-motion.csv','resting-com-support-impulses.csv','resting-local-bed-sections.csv']
    def sampled(name):
        with (run/name).open() as f: rows=[r for r in csv.DictReader(f) if int(r.get('accepted_step',step))==step]
        if len(rows)!=32 or len({int(r['contact_index']) for r in rows})!=len(rows):raise ValueError('Incomplete terminal support sections')
        return {int(r['contact_index']):r for r in rows}
    points,loads,sectors=map(sampled,files)
    data={};samples={}
    for k,r in points.items():
        s=loads[k]
        if not r['source_geometry_index']==s['source_geometry_index']==sectors[k]['source_geometry_index'] or int(r['pre_step_index'])!=step-1:
            raise ValueError('Mismatched load and point source')
        normal=[float(s['normal_'+a]) for a in 'xyz']
        if s['row_class']!='external_support_contact' or not np.allclose(normal,[0,0,1],atol=1e-7):
            raise ValueError('Unknown bed contact normal')
        force=float(s['normal_impulse_ns'])/dt
        if force<=0:continue
        xy=np.array([float(r['point_world_'+a+'_m']) for a in 'xy'])
        samples[k]=xy;data[k]=dict(row=k,source_geometry=int(r['source_geometry_index']),donor=int(r['body_index']),xy_m=xy.tolist(),normal_force_N=force,
            skin=[],cortex=[],muscle=[],ambiguous_solid_banks=[])
    pack=next(run.glob('*.mrvpack.zst'))
    if sha(pack)!=pin['compressed_sha256']:raise ValueError('Changed terminal archive')
    raw=subprocess.check_output(['zstd','-qdc',str(pack)])
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=pin['source_sha256'] or digest!=rec['pack_file_sha256']:raise ValueError('Changed terminal capture')
    sys.path.insert(0,str(reader.parent/'src'))
    banks=runpy.run_path(str(reader))['primitives']
    poses={r['body_index']:r for r in rec['accepted_body_poses']}
    tissue=Path(argv[argv.index('--soft-tissue-payload')+1]);manifest=json.loads(tissue.with_suffix('.manifest.json').read_text())
    if sha(tissue)!=manifest['payload']['sha256']:raise ValueError('Changed muscle payload')
    names={r['stable_id']:r.get('label',r.get('member_id')) for r in manifest['source']['surfaces'] if r['layer']=='muscle'}
    for (sem,stable),prim,v,f in banks(raw,poses):
        if sem not in (51004,51005,51007):continue
        low,high=v[:,:2].min(0),v[:,:2].max(0)
        use={k:q for k,q in samples.items() if ((low<=q)&(q<=high)).all()}
        if not use:continue
        z=hits(v,f,use,require_oriented=sem==51005)
        watertight=sem!=51005 or not z or closed(v,f)
        for k,h in z.items():
            label={51007:'skin',51004:'cortex',51005:'muscle'}[sem]
            if h is None:
                data[k]['ambiguous_solid_banks'].append([label,stable,'boundary ray'])
            elif sem==51005:
                if len(h)%2 or not watertight:
                    data[k]['ambiguous_solid_banks'].append([label,stable,'open bank or odd crossings'])
                else:
                    data[k][label].extend([[stable,names[stable],lo,hi] for lo,hi in zip(h[::2],h[1::2])])
            else:
                data[k][label].extend([[stable,depth] for depth in h])
    represented=0.;total=sum(r['normal_force_N'] for r in data.values())
    for r in data.values():
        skin=sorted(z for _,z in r.pop('skin'));bones=sorted((z,i) for i,z in r.pop('cortex'))
        muscles=r.pop('muscle');r['skin_ray_crossings']=len(skin)
        if not skin or any(k in ('skin','cortex') for k,_,_ in r['ambiguous_solid_banks']):
            r['result']='no unambiguous skin-to-cortex segment';continue
        if not bones or bones[0][0]<skin[0] or len(skin)<2 or bones[0][0]>=skin[1]:
            r['result']='no cortex within first skin interval';continue
        cut,first=bones[0];floor=skin[0]
        chords=[[i,name,max(lo,floor),min(hi,cut)] for i,name,lo,hi in muscles if min(hi,cut)>max(lo,floor)]
        length=union_length((lo,hi) for _,_,lo,hi in chords)
        r.update(result='resolved',first_cortex_stable_id=first,skin_to_cortex_mm=1000*(cut-floor),
            closed_muscle_union_mm=1000*length,closed_muscle_chords_mm=[[i,name,(hi-lo)*1000] for i,name,lo,hi in chords])
        if length>0:represented+=r['normal_force_N']
    return dict(step=step,time_s=rec['accepted_time_s'],frame_sha256=digest,reader_sha256=sha(reader),tissue_sha256=sha(tissue),
        pins={n:sha(run/n) for n in [*files,'launch.json','terminal.receipt.json']},sum_normal_N=total,
        normal_N_on_rays_with_confirmed_muscle=represented,rows=list(data.values()),
        scope='Vertical rays at the final interval contact query coordinates, through the accepted terminal anatomy. Each normal impulse is divided by its physical step. Chord lengths describe geometry. They do not determine force fractions in tissue or the footprint away from each sample.')

if __name__=='__main__':print(json.dumps(measure(*(Path(p).resolve() for p in sys.argv[1:])),indent=2,allow_nan=False))

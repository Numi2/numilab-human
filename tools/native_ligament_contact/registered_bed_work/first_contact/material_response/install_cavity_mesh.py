"""Rebind physical face and node consumers and compile the connected native owners."""
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
import csv, hashlib, json, os, subprocess, sys
import numpy as np
from compile_connected_joint import digest
from build_cortical_skin_registration import write_contact_include
from compile_cortical_skin_contact import run as quadrature
from register_material_surfaces import pinned

def run(job,h,out):
    read=lambda p:json.loads(p.read_bytes())
    control=job/'verified-connected-material';vol=out/'volume';folder=out/'binding';folder.mkdir()
    cfg=read(control/'scene/launch.json');env=cfg['environment'];pre='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    newvol=read(vol/'transfer.json');orig=job/'compiled-owner';oldbuild=read(orig/'build.json')
    if digest(vol/'mesh.ele')!=newvol['elements'] or read(out/'native-replay/result.json')['exit']!=0:
        raise ValueError('Uncertified material')
    o=np.load(vol/'overlay.npz');mapping=o['old_to_new'];cut=o['source'];dest=o['target']
    raw=np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    t=raw[:,:4]-1;m=raw[:,4];changed=np.unique(dest)
    old=np.loadtxt(pinned(env,'TETGEN_ELEMENTS','TETGEN_ELEMENTS_SHA256'),skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    ot=old[:,:4]-1;byface={};incident={}
    for c in changed:
        for s in range(4):
            ns=tuple(sorted(np.delete(t[c],s)))
            byface.setdefault((m[c],ns),[]).append((c,s))
        for s,n in enumerate(t[c]):incident.setdefault(n,[]).append((c,s))
    def recover_face(c,nodes):
        nodes=tuple(sorted(nodes))
        if not set(nodes).issubset(ot[c]):raise ValueError('Stale parent face')
        k=mapping[c]
        if k>=0:
            if not np.array_equal(t[k],ot[c]):raise ValueError('Stale unchanged cell')
            return k,next(i for i,n in enumerate(t[k]) if n not in nodes)
        options=byface.get((old[c,4],nodes),[])
        if len(options)!=1:raise ValueError('Ambiguous surface owner for '+str((c,nodes)))
        return options[0]
    def do_contacts_and_build():
        cortex=folder/'cortex';cortex.mkdir()
        f='registered-cortical-facets.npz';z=dict(np.load(control/'cortex'/f));a=z['cell_side_material_bank_nodes'].copy()
        for row in a:
            c,s=map(int,row[:2])
            if set(np.delete(ot[c],s))!=set(row[4:]):raise ValueError('Wrong cortical side')
            row[:2]=recover_face(c,row[4:])
        z['cell_side_material_bank_nodes']=a;np.savez_compressed(cortex/f,**z)
        cr=read(control/'cortex/result.json');cr.update(mapping=digest(cortex/f),prior_mapping=digest(control/'cortex'/f))
        (cortex/'result.json').write_text(json.dumps(cr,indent=2)+'\n')
        contact=folder/'contact'
        quadrature(h,job,job/'physical-weight-frames',contact,vol,vol/'current.node',cortex)
        sample='material-contact.npz';before=np.load(control/'contact'/sample)['samples'];after=np.load(contact/sample)['samples']
        if not np.array_equal(before[:,:5],after[:,:5]) or not np.isfinite(after).all() or np.any(after[:,5]<=0):raise ValueError('Cortical nodes, bank, body, tributary area or thickness invalid')
        model=folder/'app';model.mkdir();source=control/'app';lp='NumiHumanLoadedSkinPoseData.inc';cp='NumiHumanCorticalSkinBinding.inc'
        for p in source.iterdir():
            if p.suffix in ('.h','.hpp','.inc','.mm') and p.name not in (lp,cp):os.link(p,model/p.name)
        def substitute(s,old,new):
            if s.count(old)!=1:raise ValueError('Unexpected native admission expression: '+old[:80])
            return s.replace(old,new)
        src=(source/lp).read_text();src=substitute(src,oldbuild['source_elements'],newvol['elements'])
        (model/lp).write_text(substitute(src,'unsigned cells='+str(oldbuild['tetrahedra'])+';','unsigned cells='+str(len(t))+';'))
        write_contact_include(contact,model)
        rt=folder/'runtime';rt.mkdir();os.link(orig/'bed_response.inc',rt/'bed_response.inc')
        arr=np.column_stack((t,m)).astype('<u4');ordered=hashlib.sha256(arr.tobytes()).hexdigest()
        for name in ('runtime.mm','app-runtime.mm'):
            s=(orig/name).read_text()
            for a,b in [(oldbuild['source_elements'],newvol['elements']),(f"object.elementCount!={oldbuild['tetrahedra']}u",f"object.elementCount!={len(t)}u"),(oldbuild['ordered_nodes_and_material'][:16]+'ull',ordered[:16]+'ull'),(','.join(map(str,bytes.fromhex(oldbuild['ordered_nodes_and_material']))),','.join(map(str,bytes.fromhex(ordered))))]:
                s=substitute(s,a,b)
            (rt/name).write_text(s)
        cmds=oldbuild['commands'][:3]+read(control/'build.json')['commands']
        cmds=[[v.replace(str(orig)+'/',str(rt)+'/').replace(str(source)+'/',str(model)+'/') for v in c] for c in cmds]
        def execute(i):
            with (folder/f'build-{i}.log').open('w') as f:subprocess.run(cmds[i],stdout=f,stderr=subprocess.STDOUT,check=True)
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(execute,[0,1]))
        execute(2);execute(3);execute(4)
        (folder/'build.json').write_text(json.dumps(dict(commands=cmds,model=digest(model/'numi-human-native'),library=digest(rt/'libmetalrobo.dylib'),ordered_nodes_and_material=ordered,maximum_sample_thickness_change_m=float(max(abs(before[:,5]-after[:,5])))),indent=2)+'\n')
    def tables():
        labels=np.loadtxt(vol/'current-child.labels.csv',delimiter=',',skiprows=1,dtype='i8')
        src=pinned(env,'COMBINED_NODE_MAP_CSV','COMBINED_NODE_MAP_SHA256');node=folder/'combined-node-map.csv';res=0
        with src.open() as f,node.open('x') as g:
            g.write(next(f))
            for line in f:
                r=line.rstrip().split(',');c=int(r[43]);n=int(r[0])-1;k=mapping[c]
                if k>=0:r[43]=str(k)
                else:
                    cand=incident[n]
                    # The source registration and donor frame remain attached to the same geometric node.
                    same=[(a,b) for a,b in cand if m[a]==old[c,4]]
                    k,b=min(same,key=lambda ab:(not np.array_equal(labels[ab[0],2:4],[int(r[40]),int(r[41])]),ab))
                    if any(float(v)!=0 for v in r[48:52]):raise ValueError('Required nontrivial operational projection')
                    r[40:42]=[str(int(v)) for v in labels[k,2:4]];r[43]=str(k);r[44:48]=[str(int(v==b)) for v in range(4)];res+=1
                g.write(','.join(r)+'\n')
        src=pinned(env,'FACET_ANCESTRY_CSV','FACET_ANCESTRY_SHA256');face=folder/'facet-ancestry.csv';count=0
        with src.open() as f,face.open('x') as g:
            rd=csv.reader(f);wr=csv.writer(g,lineterminator='\n');wr.writerow(next(rd))
            for r in rd:
                c=int(r[18]);k=mapping[c]
                if k<0:k,_=recover_face(c,[int(n)-1 for n in r[19:22]]);count+=1
                if int(r[17])!=c+1:raise ValueError('Invalid element identifier')
                r[17:19]=[int(k+1),int(k)]
                if 'headwall' in r[3] and r[16]=='[-1,-1,-1]':
                    if json.loads(r[23])!=[c]:raise ValueError('Unknown headwall ancestry')
                    r[23]=json.dumps([int(k)])
                wr.writerow(r)
        pins={}
        for k,path in [('COMBINED_NODE_MAP',node),('FACET_ANCESTRY',face)]:
            pins[pre+k+'_CSV']=str(path);pins[pre+k+'_SHA256']=digest(path)
        (folder/'tables.json').write_text(json.dumps(dict(environment=pins,rebound_corners=res,rebound_faces=count),indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(tables),pool.submit(do_contacts_and_build)]
        for f in futures:f.result()
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))

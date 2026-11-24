"""Bind an admitted same-connectivity interior edit to the existing full native scene."""
import argparse,csv,json,os,re,subprocess
from itertools import combinations
from pathlib import Path
import numpy as np
from compile_connected_joint import digest
from build_cortical_skin_registration import write_contact_include

def closest(point,tet):
    best=None
    for count in range(1,5):
        for indices in combinations(range(4),count):
            pts=tet[list(indices)]
            weights=np.ones(1)
            if count>1:
                v=np.linalg.lstsq((pts[1:]-pts[0]).T,point-pts[0],rcond=None)[0]
                weights=np.r_[1-v.sum(),v]
                if weights.min() < -1e-12:continue
                weights=np.maximum(weights,0); weights/=weights.sum()
            w=np.zeros(4);w[list(indices)]=weights
            distance=np.linalg.norm(w@tet-point)
            if best is None or distance<best[0]:best=(distance,w)
    return best

def run(parent,stage):
    d=parent/'witness-material';m=stage/'witness-material';v=m/'volume';old=d/'volume';src=d/'binding'
    bind=m/'binding';app=bind/'app';tab=bind/'tables';con=bind/'contact'
    for path in (app,tab,con):path.mkdir(exist_ok=True,parents=True)
    read=lambda f:json.loads(f.read_bytes())
    vol=read(v/'result.json');prev=read(src/'build.json');env=read(parent/'accepted-material-07/launch.json')['environment']
    pr='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    def file(key):
        p=Path(env[pr+key+'_CSV'])
        if digest(p)!=env[pr+key+'_SHA256']:raise ValueError('Source pin mismatch '+key)
        return p
    X=np.loadtxt(v/'mesh.node',skiprows=1,usecols=(1,2,3));Y=np.loadtxt(v/'current.node',skiprows=1,usecols=(1,2,3))
    xx=np.loadtxt(old/'mesh.node',skiprows=1,usecols=(1,2,3));yy=np.loadtxt(old/'current.node',skiprows=1,usecols=(1,2,3))
    moved=sorted(r['vertex'] for r in vol['updated_sites'])
    if not np.array_equal(np.flatnonzero((X!=xx).any(axis=1)),moved) or not np.array_equal(np.flatnonzero((Y!=yy).any(axis=1)),moved):raise ValueError('Unexpected coordinate changes')
    if any(digest(v/f)!=digest(old/f) for f in ('mesh.ele','current-child.labels.csv','material-walls.npz')):raise ValueError('Topology or surfaces changed')
    tables=read(src/'tables/result.json')['environment'].copy()
    P=np.loadtxt(env['NUMI_H706_FRACTIONAL_SOURCE330_NODES'],delimiter=',',skiprows=1,usecols=tuple(range(2,8)))
    T=np.loadtxt(env['NUMI_H706_FRACTIONAL_SOURCE330_TETS'],delimiter=',',skiprows=1,usecols=(3,4,5,6),dtype=np.int32)
    labels=np.loadtxt(file('PARENT_LABELS'),skiprows=1,delimiter=',',dtype=np.int64)
    with file('DONOR_FRAMES').open() as f:donors={int(r['donor']):r for r in csv.DictReader(f)}
    checked=[];outfile=tab/'combined-node-map.csv'
    with file('COMBINED_NODE_MAP').open() as inp,outfile.open('w') as dst:
        dst.write(inp.readline())
        for line in inp:
            num=int(line.split(',',1)[0])-1
            if num not in moved:dst.write(line);continue
            r=line.rstrip('\r\n').split(',')
            if r[8]!='interior' or r[18]!='none' or any(float(k) for k in r[48:52]):raise ValueError('Protected physical interface')
            before=int(r[29]); donor=donors[int(r[12])]
            if np.linalg.norm(np.array(r[9:12],float)-xx[num])>1e-10:raise ValueError('Old registration is inconsistent')
            choose=np.flatnonzero((labels[:,1:]==labels[before,1:]).all(axis=1))
            options=[]
            for j in choose:
                distance,b=closest(X[num],P[T[j],3:])
                options.append((distance,int(j),b))
            distance,j,b=min(options,key=lambda o:(o[0],o[1]))
            delta=X[num]-b@P[T[j],:3]
            if np.linalg.norm(delta)>float(r[37])+np.linalg.norm(X[num]-xx[num])+1e-7:raise ValueError('Excessive parent registration error')
            ties=sum(abs(opt[0]-distance)<=1e-12 for opt in options)-1
            R=np.array([float(donor[f'current_R{a}{k}'])for a in range(3)for k in range(3)]).reshape(3,3)
            origin=np.array([float(donor['current_'+k])for k in 'xyz'])
            local=np.linalg.solve(R,X[num]-origin)
            fmt=lambda n:format(n,'.17g')
            r[9:12]=map(fmt,X[num]);r[14:17]=map(fmt,local);r[29:39]=[str(j),*map(fmt,b),*map(fmt,delta),fmt(np.linalg.norm(delta)),str(ties)]
            dst.write(','.join(r)+'\n');checked.append(dict(vertex=num,old_parent=before,parent=j,source_face=int(labels[j,4]),residual_m=float(np.linalg.norm(delta))))
    if sorted(r['vertex'] for r in checked)!=moved:raise ValueError('Missing interior point mapping')
    tables.update({pr+'COMBINED_NODE_MAP_CSV':str(outfile),pr+'COMBINED_NODE_MAP_SHA256':digest(outfile)})
    (tab/'result.json').write_text(json.dumps(dict(environment=tables,sites=checked),indent=2)+'\n')
    contact=read(src/'contact/result.json')
    if contact['loaded']!=digest(old/'current.node') or contact['material_reference']!=digest(old/'mesh.node'):raise ValueError('Stale cortical registration')
    program=(src/'contact/material-contact.npz').resolve();data=np.load(program)
    if np.isin(data['facets'][:,5:8],moved).any():raise ValueError('Cortical surface moved')
    contact.update(loaded=digest(v/'current.node'),material_reference=digest(v/'mesh.node'))
    (con/'result.json').write_text(json.dumps(contact,indent=2)+'\n')
    if not (con/program.name).exists():(con/program.name).symlink_to(program)
    baseapp=src/'app'
    for p in baseapp.iterdir():
        if p.suffix in ('.h','.hpp','.mm','.inc') and p.name not in ('NumiHumanLoadedSkinPoseData.inc','NumiHumanCorticalSkinBinding.inc'):
            subprocess.run(['/bin/cp','-cp',p,app/p.name],check=True)
    write_contact_include(con,app)
    n='NumiHumanLoadedSkinPoseData.inc';code=(baseapp/n).read_text()
    anchors=set(int(i)for i in re.findall(r'^\{\d+u,(\d+)u,\d+u,\{',code,flags=re.M))
    if not anchors or set(moved)&(anchors|{24331,24538,26199}):raise ValueError('A registered physical attachment moved')
    for key,name in [('nodeSHA','current.node'),('referenceSHA','mesh.node')]:
        token=key+'[]="'+digest(old/name)+'"'
        if code.count(token)!=1:raise ValueError('Prior compiled identity mismatch')
        code=code.replace(token,key+'[]="'+digest(v/name)+'"')
    (app/n).write_text(code)
    runtime=src/'runtime/libmetalrobo.dylib'
    if digest(runtime)!=prev['library']:raise ValueError('Native library changed')
    commands=[]
    for priorcmd in prev['commands'][3:]:
        cmd=[str(baseapp/'app-runtime.mm.o') if s==str(baseapp/'app-runtime.mm.o') else s.replace(str(baseapp)+'/',str(app)+'/') for s in priorcmd]
        commands.append(cmd)
        with (bind/('build-'+str(len(commands))+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    (bind/'build.json').write_text(json.dumps(dict(commands=commands,app=digest(app/'numi-human-native'),library=prev['library'],loaded=contact['loaded'],ordered=prev['ordered'],elements=prev['elements'],anchors=len(anchors),sites=checked),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for a in ('parent','stage'):p.add_argument(a,type=lambda v:Path(v).resolve())
    run(**vars(p.parse_args()))

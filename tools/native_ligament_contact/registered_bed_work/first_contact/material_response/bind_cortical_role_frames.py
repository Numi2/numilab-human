"""Express embedded material points in the dominant native frame of their full weighted binding."""
import csv,json,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest

def run(job,source,out):
    out.mkdir();d=json.loads((source/'result.json').read_bytes());pose=json.loads((job/'posed-app/scene/registered-initial-bodies.json').read_bytes())
    B=np.array([x['body_index'] for x in pose['bodies']]);dirs=np.array([x['current']['local_to_world_row_major_f32'] for x in pose['all_bodies']]).reshape(-1,3,3);inv=np.linalg.inv(dirs)
    origin=np.array([x['current']['position_world_m'] for x in pose['all_bodies']],dtype='f4').astype(float);z=np.load(job/'registered-boundary/registered-boundary.npz')
    fmt=lambda x:format(x,'.17g');changed={};joint={}
    def latest(key):
        rec=d['files'][key];f=Path(rec['path'])
        if digest(f)!=rec['sha256']:raise ValueError('Stale '+key)
        return f
    def save(key,path):d['files'][key]=dict(path=str(path),sha256=digest(path))
    for k in 'ID':
        cut=len(z[k+'_retained_source_node']);nd=z[k+'_node_joint'];weights=[]
        with latest('ROLE_'+k+'_FULL86').open() as f:
            next(f)
            for r,line in enumerate(f):
                if r>=cut:weights.append(np.array(line.rstrip().split(',')[1:],float))
        if len(weights)!=len(nd)-cut:raise ValueError('Wrong embedded weight count')
        src=latest('SURFACE_'+k+'_NODES');dst=out/(k+'-nodes.csv');diff=0
        with src.open() as f,dst.open('x') as g:
            head=f.readline();g.write(head);names=head.strip().split(',');cols={k:i for i,k in enumerate(names)}
            for r,line in enumerate(f):
                if r<cut:g.write(line);continue
                v=line.strip().split(',');lane=int(np.argmax(weights[r-cut]));body=int(B[lane]);n=nd[r]
                if int(v[cols['donor']])==body:g.write(line);continue
                p=np.array([float(v[cols['w'+a]]) for a in 'xyz']);l=inv[body]@(p-origin[body])
                if np.linalg.norm(dirs[body]@l+origin[body]-p)>1e-8:raise ValueError('Frame reconstruction')
                v[cols['donor']]=str(body)
                for i,a in enumerate('xyz'):
                    v[cols['l'+a]]=fmt(l[i])
                    for j in range(3):v[cols['m'+str(i)+str(j)]]=fmt(inv[body,i,j])
                g.write(','.join(v)+'\n');joint[int(n)]=(body,lane,list(map(fmt,l)));diff+=1
        changed[k]=diff;save('SURFACE_'+k+'_NODES',dst)
    src=latest('COMBINED_NODE_MAP');dst=out/'combined-node-map.csv';counts={}
    with src.open() as f,dst.open('x') as g:
        g.write(f.readline())
        for line in f:
            n=int(line[:line.index(',')])-1
            if n not in joint:g.write(line);continue
            v=line.rstrip().split(',');b,lane,p=joint[n];v[12:17]=[str(b),str(lane),*p];g.write(','.join(v)+'\n');counts[n]=counts.get(n,0)+1
    if counts.keys()!=joint.keys():raise ValueError('Unpaired volume binding')
    save('COMBINED_NODE_MAP',dst);d['dominant_frame_updates']=changed;(out/'result.json').write_text(json.dumps(d,indent=2)+'\n');print(changed)
if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:]))

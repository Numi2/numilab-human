"""Register the existing O/I/D surface world points in the actual native donor frames."""
from pathlib import Path
import csv,json,sys
import numpy as np
from compile_connected_joint import digest
PRE='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
M=['m'+str(i)+str(j) for i in range(3) for j in range(3)]
def read(path):return json.loads(path.read_bytes())
def run(trial,out):
    out.mkdir();source=read(trial/'launch.json')['environment']
    pose=read(trial/'registered-initial-bodies.json');pframes=pose['bodies'];frames={};changed=set();receipt={};updates={}
    def files(k):
        p=Path(source[PRE+k+'_CSV'])
        if digest(p)!=source[PRE+k+'_SHA256']:raise ValueError('Input hash '+k)
        return p,out/p.name
    def install(k,new):
        updates[PRE+k+'_CSV']=str(new);updates[PRE+k+'_SHA256']=digest(new)
    old,new=files('DONOR_FRAMES')
    with old.open() as inp,new.open('x') as dest:
        rr=csv.DictReader(inp);writer=csv.DictWriter(dest,rr.fieldnames,lineterminator='\n');writer.writeheader()
        for row in rr:
            binding,donor=map(int,[row['binding'],row['donor']]);p=pframes[binding]
            if p['body_index']!=donor or donor in frames:raise ValueError('Changed native donor identity')
            for key in ('current','rest'):
                origin=np.array(p[key]['position_world_m'],'f4').astype('f8')
                R=np.array(p[key]['local_to_world_row_major_f32']).reshape(3,3)
                if not np.array_equal(R,R.astype('f4')) or abs(np.linalg.det(R)-1)>8*np.finfo('f4').eps:raise ValueError('Invalid native frame')
                keys=[key+'_'+c for c in 'xyz']+[key+'_R'+str(i)+str(j) for i in range(3) for j in range(3)]
                vals=np.r_[origin,R.ravel()]
                if not np.array_equal([float(row[k]) for k in keys],vals):changed.add(donor)
                for k,v in zip(keys,vals):row[k]=format(v,'.17g')
                if key=='current':frames[donor]=origin,R,np.linalg.inv(R)
            writer.writerow(row)
    if len(frames)!=pose['binding_count']:raise ValueError('Missing donor')
    install('DONOR_FRAMES',new)
    for role in 'OID':
        key='SURFACE_'+role+'_NODES';old,new=files(key)
        count=0;maxerr=0.;total=0
        with old.open() as f,new.open('x') as dest:
            head=f.readline();dest.write(head);cols=head.strip().split(',');idx={k:i for i,k in enumerate(cols)}
            for line in f:
                v=line.rstrip('\r\n').split(',');donor=int(v[idx['donor']]);total+=1
                if donor not in changed:dest.write(line);continue
                if len(v)!=len(cols):raise ValueError('Unexpected native CSV quoting')
                point=np.array([float(v[idx['w'+c]]) for c in 'xyz']);p,R,inv=frames[donor];local=inv@(point-p)
                for k,val in zip(['l'+c for c in 'xyz']+M,np.r_[local,inv.ravel()]):v[idx[k]]=format(val,'.17g')
                maxerr=max(maxerr,float(np.max(abs(R@local+p-point))));count+=1;dest.write(','.join(v)+'\n')
        if maxerr>2e-12:raise ValueError('Donor transform lost the registered world point')
        receipt[role]=dict(source=digest(old),rows=total,local_frames_changed=count,max_roundtrip_m=maxerr);install(key,new)
    (out/'registration.json').write_text(json.dumps(dict(pose=digest(trial/'registered-initial-bodies.json'),changed_donors=sorted(changed),layers=receipt,environment=updates),indent=2)+'\n')
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))

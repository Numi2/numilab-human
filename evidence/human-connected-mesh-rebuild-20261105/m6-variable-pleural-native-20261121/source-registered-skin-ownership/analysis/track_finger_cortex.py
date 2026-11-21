"""Check sampled cortical motion and bound fourth-to-fifth phalanx separation."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

def gap(a,b):
    # Any unit projection with disjoint intervals provides a lower bound.
    # Improve the axis on the difference of the two convex hulls. The best
    # observed vertex-to-vertex distance supplies an independent upper bound.
    u=a.mean(0)-b.mean(0);lower=0.
    for _ in range(96):
        norm=np.linalg.norm(u)
        if norm<1e-11:break
        n=u/norm;i=(a@n).argmin();j=(b@n).argmax();v=a[i]-b[j]
        lower=max(lower,float(v@n));v-=u;den=float(v@v)
        if den<1e-24:break
        alpha=np.clip(-float(u@v)/den,0.,1.)
        if alpha<1e-14:break
        u+=alpha*v
    upper=float(cKDTree(a).query(b)[0].min())
    if lower>upper+1e-10:raise ValueError('Invalid distance interval')
    return lower,upper
def run(work,case,initialreceipt,terminalreceipt,series,output):
    sys.path[:0]=[str(work/'resting-surface-clearance-001/src'),str(work/'resting-surface-clearance-001'),str(work/'resting-clearance-and-atrium-005')]
    import audit
    from fast_skin_intersection_audit import install_for_float32
    install_for_float32()
    start=json.loads((case/'initial-cortical.json').read_text());a=np.load(case/'initial-cortical.npz')
    r0=json.loads(initialreceipt.read_text());rt=json.loads(terminalreceipt.read_text())
    if start['pack_sha256']!=r0['pack_file_sha256']:raise ValueError('Starting capture differs')
    ids={x[0]:x[1] for x in start['member_by_stable_id'].values()};pairs=[(x['first'],x['second']) for x in start['pairs']]
    poses={v['body_index']:v for v in r0['accepted_body_poses']};local={};f={}
    for name,body in ids.items():
        ref=poses[body];local[name]=Rotation.from_quat(ref['quaternion_xyzw']).inv().apply(a[name+'_v'].astype(float)-ref['position_m'])
        f[name]=a[name+'_f']
    records={};keys={body for body in ids.values()}
    for r in csv.DictReader(series.open()):
        k=int(r['accepted_step']);body=int(r['body_index'])
        if body not in keys:continue
        xyz=[float(r['root_'+j+'_m'])+float(r['com_relative_root_'+j+'_m']) for j in 'xyz']
        quat=[float(r['orientation_'+j]) for j in 'xyzw']
        records.setdefault(k,{})[body]=(xyz,quat)
    if rt['accepted_step']!=max(records):raise ValueError('Series does not reach terminal capture')
    direct=case/'terminal-cortical.npz';observed=np.load(direct) if direct.is_file() else None
    results=[];er=0.
    for k in [0,*sorted(records)]:
        pos={};banks={}
        for name,b in ids.items():
            if k==0:v=a[name+'_v']
            else:
                t,q=records[k][b];v=(Rotation.from_quat(q).apply(local[name])+t).astype('<f4')
            if observed is not None and k==max(records):
                er=max(er,float(np.linalg.norm(v-observed[name+'_v'],axis=1).max()))
            pos[name]=v.astype(float)
            rec=audit.clearance._exact_surface_records(v,f[name]);banks[name]=rec,audit.intersections._prepare_surface_aabb(rec)
        rows=[]
        for i,j in pairs:
            c=audit.intersections._audit_pair_prepared_first(banks[i][1],banks[j][0])['count']
            lo,hi=gap(pos[i],pos[j])
            if lo>1e-9 and c:raise ValueError('Separated cortical banks reported crossings')
            rows.append(dict(first=i,second=j,triangle_pairs=c,lower_gap_mm=max(0.,lo*1000-1e-6),upper_gap_mm=hi*1000+1e-6))
        results.append(dict(step=k,crossing_pairs=sum(x['triangle_pairs'] for x in rows),lower_gap_mm=min(x['lower_gap_mm'] for x in rows),upper_gap_mm=min(x['upper_gap_mm'] for x in rows),banks=rows))
    if observed is not None:
        direct_result=json.loads((case/'terminal-cortical.json').read_text())
        if rt['pack_file_sha256']!=direct_result['pack_sha256'] or er>2e-6:raise ValueError(('Rigid reconstruction differs from native endpoint',er))
        if [r['triangle_pairs'] for r in results[-1]['banks']]!=[r['count'] for r in direct_result['pairs']]:raise ValueError('Reconstructed and emitted cortical tests disagree')
    out={'direct_terminal_compared':observed is not None,'maximum_terminal_vertex_error_m':er if observed is not None else None,'snapshots':results,
        'scope':'Initial native coordinates followed by measured skeletal poses sampled every 32 accepted steps and at the endpoint. Bounds describe sampled triangle surfaces. Unmeasured intervals and interiors are excluded.'}
    output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))

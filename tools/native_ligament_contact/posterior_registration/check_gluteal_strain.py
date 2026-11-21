"""Test the registered pose field against the existing passive muscle specimen's energy and volume bounds."""
import json,re,sys
from pathlib import Path
import numpy as np
from pose_glutes import parts,states,model
from material_context import sha
def run(config,trace,rc,cell_dir,material,out,native_surfaces=None):
    cfg=json.loads(config.read_text())['argv'];asset=Path(cfg[cfg.index('--soft-tissue-payload')+1]);poses=states(trace,rc)
    spec=material.read_text()
    def val(key):return float(re.search(r'\b'+key+r'\s*=\s*([0-9.eE+\-]+)',spec)[1])
    if 'energy = neo_hookean(mu, lambda);' not in spec:raise ValueError('Unrecognized specimen law')
    shear=float(re.search(r'parameter mu : kPa = ([\d.]+)',spec)[1])*1000;bulk=float(re.search(r'parameter lambda : kPa = ([\d.]+)',spec)[1])*1000
    if native_surfaces is not None:
        record=json.loads((native_surfaces/'motion.json').read_text())
        if not record['recorded_pose_replay_qualified'] or record['receipt_sha256']!=sha(rc) or record['source_payload_sha256']!=sha(asset):raise ValueError('Unverified native motion source')
        for k,frame in poses.items():
            if k==0:continue
            for name,offset in record['frame_correction_local_m'].items():
                from scipy.spatial.transform import Rotation
                b=int(name);frame[b][:3]+=Rotation.from_quat(frame[b][3:]).apply(offset)
    stats={}
    for sid in (29,30):
        d=parts(asset,sid);tet=np.loadtxt(cell_dir/str(sid)/'source.1.ele',skiprows=1,dtype=int)[:,1:5]
        def basis(v):
            pts=v[tet];return (pts[:,1:]-pts[:,:1]).transpose(0,2,1)
        x=model(d,poses[0],1)[0].astype('f4').astype(float);base_points=x if native_surfaces is None else np.load(native_surfaces/('entrance-%d.npz'%sid))['xyz'].astype(float)
        base=basis(base_points);inv=np.linalg.inv(base);m=np.linalg.det(base)/6;assert m.min()>0
        peak_p=0.;peak_s=0.;peak_stretch=0.;min_stretch=1.;max_f=0.;peak_j=(0,None);peak_energy=(0,None);record=[];union=set()
        for step,pose in sorted(poses.items()):
            x=model(d,pose,1)[0].astype('f4').astype(float)
            if native_surfaces is not None and step in (0,max(poses)):
                x=base_points if step==0 else np.load(native_surfaces/('accepted-%d.npz'%sid))['xyz'].astype(float)
            F=basis(x)@inv;J=np.linalg.det(F);I1=(F**2).sum((1,2));lg=np.log(J)
            energy=.5*shear*(I1-3.-2*lg)+.5*bulk*lg**2
            P=shear*F+((bulk*lg-shear)[:,None,None]*np.linalg.inv(F).transpose(0,2,1))
            sigma=(P@F.transpose(0,2,1))/J[:,None,None]
            stress=np.linalg.norm(P,axis=(1,2));peak_p=max(peak_p,float(stress.max()));peak_s=max(peak_s,float(np.max(np.linalg.norm(sigma,axis=(1,2)))))
            sigma_f=np.linalg.svd(F,compute_uv=False)
            peak_stretch=max(peak_stretch,float(sigma_f.max()));min_stretch=min(min_stretch,float(sigma_f.min()))
            k=int(np.argmax(energy));j=int(np.argmax(J))
            if J[j]>peak_j[0]:peak_j=(float(J[j]),[step,j])
            if energy[k]>peak_energy[0]:peak_energy=(float(energy[k]),[step,k])
            fail=(J<val('minimum_J'))|(J>val('maximum_J'))|(energy>val('maximum_energy_density'))|(stress>val('maximum_stress'))
            union.update(np.flatnonzero(fail).tolist());record.append([step,float(J.min()),float(J.max()),float(energy.max()),int(fail.sum()),float(energy@m),float(J@m/m.sum())])
        a=np.array(record);stats[sid]=dict(maximum_J=peak_j,peak_energy_j_m3=peak_energy,minimum_J=float(a[:,1].min()),maximum_first_piola_frobenius_pa=peak_p,maximum_cauchy_frobenius_pa=peak_s,principal_stretch_range=[min_stretch,peak_stretch],deformed_to_registered_bulk_volume_ratio_range=[float(a[:,6].min()),float(a[:,6].max())],max_integral_energy_j=float(a[:,5].max()),failed_volume_energy_or_first_piola_cells=sorted(union),cell_sample_failures=int(a[:,4].sum()),peak_table=record)
    out.write_text(json.dumps(dict(material_sha256=sha(material),source_sha256=sha(asset),reference_sha256=sha(rc),native_pack_anchored=native_surfaces is not None,scope='Kinematic deformation compared with the separate passive skeletal muscle specimen using registered entrance as reference. Material stress does not act on this run.',sides=stats),separators=(',',':'))+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))

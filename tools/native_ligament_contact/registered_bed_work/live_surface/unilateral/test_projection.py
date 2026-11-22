"""Check unilateral contact branch values and off-node Jacobian actions on the M6."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from material_context import sha


def check(root):
    case=root/"feasible-bed"
    out=case/"build"
    cmd=json.loads((root.parent/"velocity-consistent/identities.json").read_text())["commands"][0].copy()
    cmd[7:7]=["-I",str(out)]
    cmd[cmd.index("-c")+1]=str(case/"check_bed.metal")
    cmd[cmd.index("-o")+1]=str(case/"check.air")
    subprocess.run(cmd,check=True)
    subprocess.run(["xcrun","-sdk","macosx","metallib",str(case/"check.air"),"-o",str(case/"test.metallib")],check=True)
    template=(root/"plane-limit/check_bound.mm").read_text()
    for a,b in [("test_registered_plane_fraction","measure_bed_projection"),("data.length%16","data.length%80"),
                ("data.length/4","(data.length/80)*64"),("data.length/16","data.length/80")]:
        if a not in template:
            raise ValueError(a)
        template=template.replace(a,b)
    (case/"check_bed.mm").write_text(template)
    subprocess.run(["xcrun","clang++","-fobjc-arc","-std=c++17","-O2","-framework","Metal","-framework","Foundation",
                    str(case/"check_bed.mm"),"-o",str(case/"check-bed")],check=True)
    rng=np.random.default_rng(9061)
    n=8192
    m=np.exp(rng.uniform(-8,-2,n))
    dt=1/64000
    gap=rng.uniform(-.3,4,n)*1e-7
    v=rng.normal(size=(n,3))*.1
    b=rng.normal(size=(n,3))*m[:,None]
    b[:n//4,2]=m[:n//4]
    b[n//4:,2]=-m[n//4:]
    b[n//2:3*n//4,:2]=-m[n//2:3*n//4,None]*v[n//2:3*n//4,:2]
    work=rng.normal(size=(n,3))*m[:,None]
    direction=rng.normal(size=(n,3))
    mus=rng.uniform(.04,.7,n); mud=mus*rng.uniform(.3,1.,n)
    j=json.loads((root/"normal-row/result.json").read_text()); b[0]=j["node"]["raw_rhs_Ns"]; v[0]=j["node"]["velocity_m_s"];m[0]=j["node"]["mass_kg"]
    witness=j["limiter"];gap[0]=witness["gap_m"]-(1e-9+2*(witness["floor_m"]-1e-9));mus[0]=mud[0]=.05
    arr=np.zeros((n,5,4),dtype="<f4")
    arr[:,0,:3]=b;arr[:,0,3]=gap;arr[:,1,:3]=v;arr[:,1,3]=m;arr[:,2,:3]=work
    arr[:,3,:3]=direction;arr[:,3,3]=dt;arr[:,4,0]=mus;arr[:,4,1]=mud
    unpack=arr.astype(float)
    b=unpack[:,0,:3];gap=unpack[:,0,3];v=unpack[:,1,:3];m=unpack[:,1,3,None]
    work=unpack[:,2,:3];direction=unpack[:,3,:3];dt=unpack[:,3,3];mus=unpack[:,4,0];mud=unpack[:,4,1]
    def physical(e):
        rhs=b-e*work
        vel=v+e*direction
        g=gap+e*dt*direction[:,2]
        lam=np.maximum(0.,-m[:,0]*g/dt-rhs[:,2])
        u=-m*vel[:,:2]-rhs[:,:2]
        distance=np.linalg.norm(u,axis=1)
        stick=distance<=mus*lam
        tang=u*np.where(stick,1.,mud*lam/np.maximum(distance,1e-100))[:,None]
        z=np.c_[tang,lam]
        return rhs+z,z,np.where(lam==0,-2.,np.where(stick,-1.,mud))
    res,imp,branch=physical(0)
    eps=1.e-5
    hi,_,b1=physical(eps);lo,_,b2=physical(-eps)
    valid=(b1==branch)&(b2==branch)
    fd=(lo-hi)/(2*eps)
    with tempfile.TemporaryDirectory() as td:
        p=Path(td);arr.tofile(p/"in")
        gpu=subprocess.run([str(case/"check-bed"),str(case/"test.metallib"),str(p/"in"),str(p/"out")],check=True,capture_output=True,text=True)
        data=np.fromfile(p/"out","<f4").reshape(n,4,4).astype(float)
    if not np.isfinite(data).all() or not np.array_equal(branch,data[:,0,3]):
        raise ValueError("Wrong active set")
    per=np.max(abs(data[valid,2,:3]-fd[valid])/m[valid],axis=1)
    er=np.max(abs(data[:,0,:3]-imp)/m);rr=np.max(abs(data[:,1,:3]-res)/m)
    cone=np.linalg.norm(data[:,0,:2],axis=1)-mus*data[:,0,2]
    if max(er,rr,per.max())>2.e-5 or np.any(data[:,0,2]<0) or (cone/m[:,0]).max()>1.e-6 or valid.sum()<.99*n:
        raise ValueError((er,rr,per.max(),valid.sum(),(cone/m[:,0]).max()))
    z=-gap/dt
    fitted=direction.copy()
    fitted[:,2]=np.where(branch==-2.,np.maximum(direction[:,2],z),z)
    fitted[branch==-1.,:2]=-v[branch==-1.,:2]
    proposal_error=np.max(abs(fitted-data[:,3,:3]))
    if proposal_error>2.e-6 or np.min(gap+dt*data[:,3,2]) < -4.e-12:
        raise ValueError("Infeasible trial direction or incorrect constrained branch")
    report=dict(proposal_error_m_s=float(proposal_error),device=gpu.stdout.strip(),cases=n,finite_difference_cases=int(valid.sum()),inactive=int((branch==-2).sum()),
                sticking=int((branch==-1).sum()),sliding=int((branch>=0).sum()),max_impulse_error_per_mass=float(er),
                max_action_error_per_mass=float(per.max()),observed_corner_hypothetical_dual_Ns=data[0,0,:3].tolist(),
                source=sha(case/"contact.metalinc"),test_metallib=sha(case/"test.metallib"),
                scope="Projection and derivative from the full momentum row. The first input reproduces the published row. "
                      "The computed reaction at that input is a candidate, not an accepted whole-scene load.")
    (case/"derivative.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    check(Path(sys.argv[1]).resolve())

"""Run existing handoff/lineage owners' regression suites on SSH Numi."""
from pathlib import Path
import subprocess,json,hashlib,os,time,sys
F=Path(__file__).resolve().parent;P=F.parent;H=P/"paired-region-rebuild-667/tetgen-handoff-697"
names=[H/n for n in ["test_conditioned_D_typed_precedence.py","test_handoff_697_typedD.py",
                    "test_native_capacity_preflight.py","test_operational_D_targets.py",
                    "test_prepared_reduced_lineage.py"]]
names += [F/n for n in ["test_registered_surface_handoff.py","test_readiness_dependencies.py",
                       "test_actual_handoff_cli.py","test_reduced_surface_lineage.py",
                       "test_actual_reduced_lineage.py","test_actual_D071_lineage.py",
                       "test_operational_D_point_domains.py"]]
owners=[H/"registered_surface_handoff.py",H/"prepare_tetgen_handoff_697_typedD.py",
        H/"run_native_domain_217_typedD_697.py",H/"canonical_d_operational_targets.py",
        F/"prepare_reduced_surface_lineage_current.py"]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):sha(p) for p in owners}
env=dict(os.environ,NUMI_REGISTERED_SURFACE_OWNER=str(H/"registered_surface_handoff.py"),
         NUMI_REDUCED_LINEAGE_OWNER=str(F/"prepare_reduced_surface_lineage_current.py"))
runs=[]
for p in names:
    t=time.monotonic()
    r=subprocess.run([sys.executable,str(p)],capture_output=True,text=True,env=env)
    runs.append(dict(script=str(p),sha256=sha(p),returncode=r.returncode,
                     wall_seconds=time.monotonic()-t,stdout=r.stdout,stderr=r.stderr))
    print(p.name,r.returncode,flush=True)
    if r.returncode:
        print((r.stdout+r.stderr)[-4000:],flush=True)
        break
unchanged=all(sha(p)==value for p,value in pins.items())
receipt=dict(status="PASS" if len(runs)==len(names) and all(r["returncode"]==0 for r in runs) and unchanged else "FAIL",
             owners=pins,owners_unchanged=unchanged,runs=runs,
             physical_admission=False,runtime=0,accepted_steps=0)
out=F/"combined-handoff-lineage-regression-run007.receipt.json"
out.write_text(json.dumps(receipt,indent=2)+"\n")
print(receipt["status"],out,flush=True)
if receipt["status"]!="PASS":raise SystemExit(1)

from pathlib import Path
import hashlib,json,sys,struct,tempfile
import numpy as np
from source_carrier_packet_188_root import load_packet,relative_close
H=Path(__file__).resolve().parent
P=H.parent
out=H/'recorded-carrier-verifier-188-tests-root'
out.mkdir(exist_ok=False)
cases=[]
def run(name,fn,reject=False):
    try:fn()
    except (ValueError,AssertionError) as e:
        if not reject:raise
        cases.append({'name':name,'status':'EXPECTED_REJECTION','reason':str(e)})
    else:
        if reject:raise AssertionError(name+' silently passed')
        cases.append({'name':name,'status':'PASS'})
for role,qdir,lo in [('Skin','skin-743497-run-004-source-batch/source-quadrature-007',0),('Fat','fat-853406-run-003-root/source-quadrature-162-root',14316)]:
    q=P/'fractional-semantic-support-090-sparse-graph'/qdir
    globals_=np.arange(lo,lo+14316,dtype=np.uint32)
    run(role+' actual packet binary/NPZ/source moments',lambda:load_packet(q/'source-quadrature.bin',q/'source-quadrature.npz',globals_))
    a,sha=load_packet(q/'source-quadrature.bin',q/'source-quadrature.npz',globals_)
    raw=(q/'source-quadrature.bin').read_bytes()
    work=out/role;work.mkdir()
    b=work/'packet.bin';z=work/'packet.npz'
    b.write_bytes(raw+b'x')
    z.write_bytes((q/'source-quadrature.npz').read_bytes())
    run(role+' trailing packet byte',lambda:load_packet(b,z,globals_),True)
    b.write_bytes(raw)
    aa={k:v.copy() for k,v in a.items()};aa['carrier_weights'][0]*=.9
    np.savez(z,**aa)
    run(role+' NPZ/binary mismatch',lambda:load_packet(b,z,globals_),True)
    # Consistently tamper binary and NPZ: parity alone cannot establish moments.
    vals=[];payload=bytearray(b'FSSCARR01'+struct.pack('<QQ',14316,len(aa['carrier_weights'])))
    for key in ['source_parent_tet_rows','global_source330_tet_rows','parent_offsets','carrier_weights','carrier_barycentric','carrier_target_child_rows','carrier_target_distance_m']:
        payload.extend(aa[key].tobytes())
    b.write_bytes(payload)
    run(role+' jointly corrupted mass/first moment',lambda:load_packet(b,z,globals_),True)
run('tiny positive source preserved',lambda:relative_close([1e-22],[1e-22],'tiny'))
run('tiny source omitted',lambda:relative_close([0.],[1e-22],'tiny'),True)
run('tiny source doubled',lambda:relative_close([2e-22],[1e-22],'tiny'),True)
report={'status':'PASS_PACKET_AND_SCALE_AWARE_VERIFIER_CONTROLS','cases':cases,'case_count':len(cases),'physical_admission':False,'native_steps':0,'full_allocation_checked':False,'helper_sha256':hashlib.sha256((H/'source_carrier_packet_188_root.py').read_bytes()).hexdigest(),'verifier_sha256':hashlib.sha256((H/'verify_recorded_carrier_allocation_188_root.py').read_bytes()).hexdigest()}
(out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

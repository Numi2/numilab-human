"""Recheck source-compliant preparation and bounded physical traces."""
import base64
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct

BASE=Path(__file__).resolve().parent
COUNTS={'qv':257,'motor':416,'activation':416,'tendon_force':416}
COUNTS_V2={'qv':257, **{name:416 for name in ['motor','activation','fiber_length','fiber_velocity','path_length','path_velocity','applied_force','tendon_tension','fiber_residual']}}
SCENARIOS={'recruited','replay','zero','unavailable'}


def require(value, reason):
    if not value:raise ValueError(reason)


def read(path):
    data=path.read_bytes()
    return (gzip.decompress(data) if path.suffix=='.gz' else data).decode()


def record(text, prefix):
    values=[json.loads(line.split('=',1)[1]) for line in text.splitlines() if line.startswith(prefix+'=')]
    require(len(values)==1,'missing or duplicate '+prefix)
    return values[0]


def audit_trace(text, roots, require_equivalence=True):
    records={};counts=None;schema=None
    for line in text.splitlines():
        if not line.startswith('prepared_recruitment={'):continue
        r=json.loads(line.split('=',1)[1]);key=(r['scenario'],r['root'],r['kind'])
        if schema is None:
            schema=r['schema'];counts=COUNTS if schema=='numi.human.prepared-recruitment-trace.v1' else COUNTS_V2
        require(r['schema']==schema and schema in {'numi.human.prepared-recruitment-trace.v1','numi.human.prepared-recruitment-trace.v2'},'trace schema')
        require(key[0] in SCENARIOS and type(key[1]) is int and 1<=key[1]<=roots and key[2] in counts,'trace identity')
        require(r['elapsed_microseconds']==100*key[1] and key not in records,'trace clock or duplicate')
        raw=base64.b64decode(r['fp32_le_base64'],validate=True)
        require(len(raw)==4*counts[key[2]],'trace extent')
        values=struct.unpack('<'+str(counts[key[2]])+'f',raw)
        require(all(math.isfinite(x) for x in values),'nonfinite trace')
        if key[2] in {'motor','activation'}:require(all(0<=x<=1 for x in values),'actuator bounds')
        if key[2]=='qv':require(abs(sum(x*x for x in values[3:7])-1)<=16*2**-23,'quaternion')
        records[key]=(raw,values)
    require(counts is not None and len(records)==4*len(counts)*roots,'incomplete scenario or root evidence')
    first_dropout_difference=None;replay_equal=True
    for i in range(1,roots+1):
        for kind in counts:
            replay_equal &= records['recruited',i,kind][0]==records['replay',i,kind][0]
            if records['zero',i,kind][0]!=records['unavailable',i,kind][0] and first_dropout_difference is None:
                delta=[abs(a-b) for a,b in zip(records['zero',i,kind][1],records['unavailable',i,kind][1])]
                first_dropout_difference={'root':i,'kind':kind,'index':delta.index(max(delta)),'maximum_absolute_difference':max(delta)}
        for scenario in SCENARIOS:
            motor=records[scenario,i,'motor'][1]
            require(all(x==0 for x in motor) if i==1 or scenario in {'zero','unavailable'} else max(motor)>0,'motor availability')
    if require_equivalence:
        require(replay_equal,'replay drift');require(first_dropout_difference is None,'dropout drift')
    def last(scenario,kind):return records[scenario,roots,kind][1]
    metrics={'scenarios':4,'accepted_roots_per_scenario':roots,'seconds_per_scenario':roots/10000,
        'replay':'bitwise' if replay_equal else 'failed','dropout_matches_zero':first_dropout_difference is None,
        'first_dropout_difference':first_dropout_difference,
        'maximum_delivered_excitation':max(max(records['recruited',i,'motor'][1]) for i in range(1,roots+1))}
    for kind in ['activation','tendon_force' if counts is COUNTS else 'applied_force']:
        delta=max(abs(x-y) for x,y in zip(last('recruited',kind),last('zero',kind)))
        require(delta>0,'missing physical response');metrics['terminal_'+kind+'_difference']=delta
    for scenario in ['recruited','zero']:
        final=last(scenario,'qv')
        metrics[scenario]={'right_ankle_velocity_rad_s':final[237],
            'root_linear_speed_m_s':math.sqrt(sum(x*x for x in final[129:132])),
            'maximum_root_linear_speed_m_s':max(math.sqrt(sum(x*x for x in records[scenario,i,'qv'][1][129:132])) for i in range(1,roots+1))}
    return metrics


def audit_static(text):
    r=record(text,'source_compliant_equilibrium');q=record(text,'compiled_equilibrium_q');m=record(text,'compiled_equilibrium_muscles')
    require(r['schema']=='numi.human.source-compliant-equilibrium.v1' and r['balanced'] is True,'static certificate failed')
    require(len(q)==129 and all(math.isfinite(x) for x in q),'pose extent')
    for name in ['acceleration','force_residual','equality_force','limit_force','muscle_force','support_force','gravity_target']:
        require(len(r[name])==128 and all(math.isfinite(x) for x in r[name]),'force extent or finiteness')
    require(max(abs(x) for x in r['acceleration'])<=0.05,'static acceleration bound')
    for i in range(128):
        force=sum(r[k][i] for k in ['equality_force','limit_force','muscle_force','support_force'])-r['gravity_target'][i]
        require(abs(force-r['force_residual'][i])<=1e-9*(1+abs(force)),'force decomposition')
    require(len(r['support_normal_force'])==len(r['support_gap'])==18,'support extent')
    require(all(0<=f<=5000 and g>=-1e-6 and (f<=1e-6 or abs(g)<=1e-6) for f,g in zip(r['support_normal_force'],r['support_gap'])),'unilateral support')
    history=r['objective_history'];require(len(history)==r['iterations']+1 and all(math.isfinite(x) and x>=0 for x in history),'search trace')
    require(all(b<a for a,b in zip(history,history[1:])),'nondecreasing accepted search')
    for name in ['activation_fp64','activation_fp32','reference_fiber_length_m','actuator_force_n']:
        require(len(m[name])==416 and all(math.isfinite(x) for x in m[name]),'muscle extent')
    require(all(0<=x<=1 for x in m['activation_fp64']) and all(x>0 for x in m['reference_fiber_length_m']),'muscle state bounds')
    return r,q,m


def verify(base=BASE):
    receipt=json.loads((base/'receipt.json').read_text())
    require(receipt['schema']=='numi.human.source-compliant-preparation-receipt.v1','receipt schema')
    expected={'offline_source_compliant_equilibrium':True,'fp32_pose_recruitment_balance':True,
        'bounded_native_transactions':True,'six_millisecond_dropout_equivalence':False,'registered_anatomical_tissue':False,'standing':False,'walking':False,
        'experimental_calibration':False,'performance':False,'full_release':False}
    require(receipt['qualification']==expected,'unsupported qualification')
    names=[row['path'] for row in receipt['artifacts']]
    actual={str(p.relative_to(base)) for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='receipt.json'}
    require(len(names)==len(set(names)) and set(names)==actual,'incomplete or duplicate artifact inventory')
    for row in receipt['artifacts']:
        path=base/row['path'];require(path.resolve().is_relative_to(base.resolve()),'artifact escape')
        data=path.read_bytes();require(len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],'artifact drift: '+row['path'])
    for label in ['qualified','fp32-qualified']:
        run=json.loads((base/(label+'-completion.json')).read_text())
        require(run['native_head']==receipt['native_commit'] and run['native_status']=='' and run['returncode']==0,'native revision or completion')
    fp64,q,m=audit_static(read(base/'qualified.log'))
    require(read(base/'qualified.log')==read(base/'recruit32.log'),'independent compile replay drift')
    fp32,q32,m32=audit_static(read(base/'fp32-qualified.log'))
    state=(base/'fixture/prepared.nhinit').read_bytes()
    require(state[:8]==b'NHINIT1\0','initial state schema')
    stored_q=struct.unpack_from('<129f',state,96)
    stored_a=[struct.unpack_from('<4f',state,96+4*257+16*i)[1] for i in range(416)]
    require(list(stored_q)==q32 and stored_a==m32['activation_fp64'],'support-only check changed admitted q or activation')
    require(list(stored_q)==[struct.unpack('<f',struct.pack('<f',x))[0] for x in q],'FP32 pose conversion drift')
    source=json.loads((base/'source-audit.json').read_text())
    require(source['passed'] is True and source['mujoco_version']=='3.12.0' and source['maximum_relative_mass_action_error']<=1e-6 and
        source['maximum_relative_gravity_error']<=2e-6 and source['maximum_support_lowering_error_m']<=2e-7 and source['minimum_support_gap_m']>=-1e-6,'independent source audit')
    require('100% tests passed out of 3' in read(base/'source-tests-qualified.log'),'native regressions')
    text=read(base/'published4.log')
    require('Executed 1 test, with 0 failures' in text,'native cohort completion')
    run=json.loads((base/'published4-launch.json').read_text())
    require(run['returncode']==0 and run['source_state']['native']=={'revision':receipt['runtime_commit'],'status':''} and
        run['source_state']['brain']=={'revision':receipt['brain_commit'],'status':''},'runtime source drift')
    metrics=audit_trace(text,4)
    require(metrics==receipt['measurements']['four_root_horizon'],'runtime measurements drift')
    long_run=audit_trace(read(base/'horizon-streamed.log.gz'),64,require_equivalence=False)
    require(long_run==receipt['measurements']['rejected_six_millisecond_comparison'] and not long_run['dropout_matches_zero'],'retained physical comparison failure missing')
    timeout=json.loads((base/'horizon-qualified-launch.json').read_text())
    require(timeout['returncode']==124,'retained short-deadline failure missing')
    return {'status':'source_compliant_preparation_verified','native_horizon':metrics,'standing':False,'walking':False}

if __name__=='__main__':print(json.dumps(verify(),indent=2))

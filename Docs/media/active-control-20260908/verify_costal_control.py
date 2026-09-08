"""Recompute bounded source-bound controller evidence; no behavior promotion."""
from pathlib import Path
import hashlib,json,struct
out=Path(__file__).resolve().parent

def load(directory, sha, *, raw=False):
 data=(directory/(sha+'.artifact')).read_bytes()
 assert hashlib.sha256(data).hexdigest()==sha
 return data if raw else json.loads(data)

runs={}
for mode in ['active','replay','unavailable','emergency']:
 directory=out/('costal-'+mode)
 summary=json.loads((directory/'muscle-locomotor-research.json').read_text())
 assert summary['accepted_roots']==4 and summary['rejected_roots']==0 and not summary['promotable']
 rows=[]
 for root in summary['roots']:
  action=load(directory,root['motor_action_sha256'])
  execution=load(directory,root['execution_sha256'])
  assert execution['outcome']=='accepted' and execution['controlStep']==root['control_step']
  sample=load(directory,execution['sampleSHA256'])
  sensor_hashes={};proprio=None;proprio_validity=None
  for channel in sample['channels']:
   values=load(directory,channel['valuesSHA256'],raw=True)
   validity=load(directory,channel['validitySHA256'],raw=True)
   assert len(values)==channel['valuesByteCount'] and len(validity)==channel['validityByteCount']
   sensor_hashes[channel['modality']]=(channel['valuesSHA256'],channel['validitySHA256'])
   if channel['modality']==4:
    assert channel['featureDimension']==10 and channel['receptorCount']==416
    proprio=struct.unpack('<4160f',values)
    proprio_validity=struct.unpack('<416I',validity)
  assert proprio is not None
  rows.append({'step':action['controlStep'],'action_sha256':root['motor_action_sha256'],
    'excitation_max':max(action['actuatorCommands']),
    'descending_max':max(action['learnedDescendingCommands']),
    'motor_inhibition':action['motorInhibition'],'protective_interrupt_mask':action['protectiveInterruptMask'],
    'settled_activation_max':max(proprio[1::10]) if all(v & 2 for v in proprio_validity) else None,
    'settled_tendon_tension_max_N':max(proprio[7::10]) if all(v & (1<<7) for v in proprio_validity) else None,
    'proprioception':proprio,'sensor_hashes':sensor_hashes})
 runs[mode]=rows
assert [x['action_sha256'] for x in runs['active']]==[x['action_sha256'] for x in runs['replay']]
assert [x['sensor_hashes'] for x in runs['active']]==[x['sensor_hashes'] for x in runs['replay']]
assert runs['active'][0]['excitation_max']==0
assert all(x['excitation_max']>0 for x in runs['active'][1:])
assert all(x['settled_activation_max']>0 for x in runs['active'][2:])
for mode in ['unavailable','emergency']:
 assert all(x['excitation_max']==0 for x in runs[mode])
assert all(x['settled_activation_max']==0 for x in runs['emergency'][1:])
assert all(x['motor_inhibition']==1 and x['protective_interrupt_mask']!=0 for x in runs['emergency'])
force_response=any(a['proprioception'][6::10]!=b['proprioception'][6::10] for a,b in zip(runs['active'][1:],runs['emergency'][1:]))
kinesthetic_response=any(a['sensor_hashes'][9][0]!=b['sensor_hashes'][9][0] for a,b in zip(runs['active'][1:],runs['emergency'][1:]))
# Retain unresolved response at this horizon as negative physical evidence.
# Emergency-stop roots provide valid native sensors; ablated observations do not.
for rows in runs.values():
 for row in rows:del row['proprioception']
result={'format':'numi-costal-muscle-control-verification-v1','promotable':False,
 'checks':{'four_accepted_roots_per_run':True,'exact_motor_and_sensor_replay':True,
 'unavailable_proprioception_zero_output':True,'emergency_stop_zero_output':True,
 'initialized_activation_advances':True,'source_force_response_resolved_at_this_horizon':force_response,
 'kinesthetic_values_differ':kinesthetic_response},
 'timestep_microseconds':10,'physical_time_microseconds_per_run':40,
 'physical_force_response_qualified':force_response,
 'boundary':'bounded activation recruitment; force/kinematic response is reported separately; no sustained standing, recovery or walking','runs':runs}
(out/'costal-control-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result['checks'],indent=2))

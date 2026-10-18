from pathlib import Path
import json,hashlib,sys,numpy as np
sys.path.insert(0,'/Users/n/numi-human-closed-outer-skin-20261017/src')
from numilab_human.resting_pleura_proxy import _parse_payload
from numilab_human.resting_intercostal_mesh_arrangement import encode_payload_with_row_replacements
c=Path('/Users/n/numi-human-separation-20261018/C');base=c/'improve-wall23-map-retired-004';d6=c/'improve-wall23-cluster-006'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
proof=json.loads((d6/'volume-perturbation.json').read_text());assert proof['passed'];r=json.loads((base/'resting-anatomy-receipt.json').read_text());p=d6/'resting-thorax.nhanatomy';new=sha(p);r['functional_bindings']['anatomy_payload_sha256']=new
for rib in (48,49,50):r['functional_bindings'][f'intercostal_313_rib{rib}_motion_attachment']['anatomy_payload_sha256']=new
r['passive_tissue_parent_bindings']['anatomy_payload_sha256']=new;r['payload'].update(path=str(p),sha256=new,bytes=p.stat().st_size,index_count=r['payload']['index_count']-12);r['provenance']['cardiac_geometry_binding']['common_field']['anatomy_payload_sha256']=new
r['provenance']['complete_wall23_cluster_repair']={'base_payload_sha256':sha(base/'resting-thorax.nhanatomy'),'candidate_payload_sha256':new,'preparation_sha256':[sha(c/'improve-wall23-critical-005/preparation.json'),sha(d6/'preparation.json')],'volume_proof_sha256':sha(d6/'volume-perturbation.json'),'uniform_relative_volume_bound':proof['aggregate_relative_volume_delta_uniform_upper_bound_max'],'retire_unused_source_slots':[227812,227814],'scope':'Two bounded directed collapses on one238nm source cluster; original attributes preserved. No broad failed flip batch included.'}
for name in ('common-cardiac-map-f32.bin','common-cardiac-domains-f32.bin','common-cardiac-volumes-f32.bin'):(d6/name).symlink_to((base/name).resolve())
(d6/'resting-anatomy-receipt.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
source=p
for name,drop in [('improve-wall23-retired-007',227814),('improve-wall23-final-008',227812)]:
 out=c/name;out.mkdir(exist_ok=False);raw=source.read_bytes();rows=_parse_payload(raw)[-1];row=rows[23];assert drop not in row['faces'];v=np.delete(row['vertices6'],drop,axis=0);f=row['faces'].copy();f[f>drop]-=1;assert np.array_equal(v[f],row['vertices6'][row['faces']]);payload,abi=encode_payload_with_row_replacements(raw,{23:{**row,'vertices6':v,'faces':f}});target=out/'resting-thorax.nhanatomy';target.write_bytes(payload);(out/'source-slot-compaction.json').write_text(json.dumps(dict(base=str(source),base_sha256=sha(source),candidate=str(target),candidate_sha256=sha(target),retired_source_vertex=drop,emitted_triangle_attributes_exact=True,abi=abi),indent=2)+'\n');source=target
print(json.dumps({'final':str(source),'sha256':sha(source),'uniform_cluster_volume_bound':proof['aggregate_relative_volume_delta_uniform_upper_bound_max']}))

from pathlib import Path
import json
from numilab_human import right_choroid_laterality_repair as repair
from numilab_human import model as human
root=Path('/Users/home/numilab-human')
views=root/'Build/right-choroid-laterality-repair-20260929/native/source-rest/views'
pack=next(views.glob('*.mrvpack'))
pose=next(views.glob('*.torso-anatomy-poses.json'))
report=repair.audit_native_selected(
 root/'Build/whole-visceral-coverage-20260929/payload.verified/source-organ-family-anatomy.nhanatomy',
 root/'Docs/media/whole-body-embeddedness-20260929/indexed-census.tar.gz',
 root/'Docs/media/whole-body-embeddedness-20260929/quotient-census.tar.gz',
 root/'Docs/media/source-topology-repair-20260929/payload/source-topology-repair-candidates.nhanatomy',
 root/'Docs/media/source-topology-repair-20260929/source-audit.json',
 root/'Docs/media/source-topology-repair-20260929/inspection-selection.final.json',
 root/'Build/source-topology-repair-20260929/captures.final',
 root/'Docs/media/source-topology-repair-20260929/captures/source-rest/pose-snapshot.json',
 root/'Build/right-choroid-laterality-repair-20260929/payload/right-choroid-laterality-candidate.nhanatomy',
 pack,pose)
out=root/'Build/right-choroid-laterality-repair-20260929/native-selected-audit.json'
human.write_json(out,report)
print(json.dumps({k:report[k] for k in ['passed','bilateral_source_surface_count','full_support_pass_before','full_support_pass_after','remaining_wrong_side_source_ids','visible_source_412_stable_id']}),flush=True)

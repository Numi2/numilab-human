from pathlib import Path
import json
from numilab_human import right_eye_laterality_cleanup as repair
from numilab_human import model as human
root=Path('/Users/home/numilab-human')
views599=root/'Build/right-choroid-laterality-repair-20260929/native/source-rest/views'
views602=root/'Build/right-eye-laterality-cleanup-20260929/native/source-rest/views'
report=repair.audit_native_selected(
 root/'Build/whole-visceral-coverage-20260929/payload.verified/source-organ-family-anatomy.nhanatomy',
 root/'Docs/media/whole-body-embeddedness-20260929/indexed-census.tar.gz',
 root/'Docs/media/whole-body-embeddedness-20260929/quotient-census.tar.gz',
 root/'Docs/media/source-topology-repair-20260929/payload/source-topology-repair-candidates.nhanatomy',
 root/'Docs/media/source-topology-repair-20260929/source-audit.json',
 root/'Docs/media/source-topology-repair-20260929/inspection-selection.final.json',
 root/'Build/source-topology-repair-20260929/captures.final',
 root/'Docs/media/source-topology-repair-20260929/captures/source-rest/pose-snapshot.json',
 root/'Docs/media/right-choroid-laterality-20260929/candidate/right-choroid-laterality-candidate.nhanatomy',
 next(views599.glob('*.mrvpack')),
 next(views599.glob('*.torso-anatomy-poses.json')),
 root/'Build/right-eye-laterality-cleanup-20260929/payload/right-eye-laterality-candidates.nhanatomy',
 next(views602.glob('*.mrvpack')),
 next(views602.glob('*.torso-anatomy-poses.json')))
out=root/'Build/right-eye-laterality-cleanup-20260929/native-selected-audit.json'
human.write_json(out,report)
print(json.dumps({k:report[k] for k in ['passed','bilateral_source_surface_count','full_support_pass_before','full_support_pass_after','remaining_geometry_failed_right_eye_source_ids']}),flush=True)

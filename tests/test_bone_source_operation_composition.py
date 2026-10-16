import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
from numilab_human import composed_bone_body_frame as owner, model
from test_bone_opposed_source_face_retirement import fixture
from test_thumb_reference_surface_patch import patch_fixture


def three_bone_fixture():
    raw=fixture();h,rows,vo,io=owner._decode_bones(raw);vertices=[];indices=[];outrows=[]
    for old in rows:
        src=rows[92] if old[5] in (106,113) else old
        points=np.frombuffer(raw,'<f4',count=src[2]*6,offset=vo+src[1]*24).reshape(-1,6)
        faces=np.frombuffer(raw,'<u4',count=src[4],offset=io+src[3]*4)-src[1]
        row=list(old);row[1]=len(vertices);row[2]=len(points);row[3]=len(indices);row[4]=len(faces)
        vertices.extend(points);indices.extend(faces+row[1]);outrows.append(row)
    h=list(h);h[3]=len(vertices);h[4]=len(indices)
    return (owner._BONE_HEADER.pack(*h)+b''.join(owner._BONE_RECORD.pack(*r) for r in outrows)
            +np.asarray(vertices,dtype='<f4').tobytes()+np.asarray(indices,dtype='<u4').tobytes())


def persist(folder,raw,manifest):
    folder.mkdir();p=folder/'bodyparts3d-myosim-major-bones.nhbones';p.write_bytes(raw)
    manifest=json.loads(json.dumps(manifest));h=owner._BONE_HEADER.unpack_from(raw)
    manifest['payload']={'file':p.name,'sha256':model.sha256(p),'bytes':len(raw),'vertex_count':h[3],'index_count':h[4]}
    (folder/'bodyparts3d-myosim-major-bones.manifest.json').write_text(json.dumps(manifest))
    return folder,raw,manifest


def retire(folder,prior,sid):
    source,raw,manifest=prior
    out,proof=owner.retire_exact_opposed_bone_source_faces(raw,stable_id=sid,source_face_pair=(0,4))
    manifest=json.loads(json.dumps(manifest));manifest['source_face_retirement']={'input_artifact':str(source),'input_manifest_sha256':model.sha256(source/'bodyparts3d-myosim-major-bones.manifest.json'),'input_payload_sha256':model.sha256(source/'bodyparts3d-myosim-major-bones.nhbones'),'stable_id':sid,'source_face_pair':[0,4],'proof':proof}
    return persist(folder,out,manifest)


def thumb(folder,prior):
    source,raw,manifest=prior;_,vertices,faces,ancestry=patch_fixture()
    out,proof=owner.replace_thumb_reference_surface(raw,vertices6=vertices,faces=faces,face_ancestry=ancestry,protected_source_slots=(0,1))
    folder.mkdir();np.savez(folder/'patch.npz',vertices6=vertices,faces=faces);(folder/'ancestry.json').write_text(json.dumps({'face_ancestry':ancestry}))
    manifest=json.loads(json.dumps(manifest));manifest['thumb_reference_surface']={'input_artifact':str(source),'input_manifest_sha256':model.sha256(source/'bodyparts3d-myosim-major-bones.manifest.json'),'input_payload_sha256':model.sha256(source/'bodyparts3d-myosim-major-bones.nhbones'),'patch_path':'patch.npz','patch_sha256':model.sha256(folder/'patch.npz'),'ancestry_path':'ancestry.json','ancestry_sha256':model.sha256(folder/'ancestry.json'),'protected_source_slots':[0,1],'proof':proof}
    # persist expects a new directory; retain the pinned patch files.
    p=folder/'bodyparts3d-myosim-major-bones.nhbones';p.write_bytes(out);h=owner._BONE_HEADER.unpack_from(out);manifest['payload']={'file':p.name,'sha256':model.sha256(p),'bytes':len(out),'vertex_count':h[3],'index_count':h[4]};(folder/'bodyparts3d-myosim-major-bones.manifest.json').write_text(json.dumps(manifest));return folder,out,manifest


class CompositionTests(unittest.TestCase):
    def verify_order(self,order):
        with TemporaryDirectory() as d:
            root=Path(d);base=persist(root/'base',three_bone_fixture(),{'source':{'identity':'unchanged'},'body_local_bakes':{'proof':'unchanged'}});current=base
            for i,op in enumerate(order):current=thumb(root/str(i),current) if op=='thumb' else retire(root/str(i),current,op)
            path,raw,manifest=current
            with patch.object(owner,'_validate_composed_rib_reframe_source003',return_value={'status':'BASE_SOURCE_REPLAY_ONLY','validation_sha256':'0'*64}) as check:
                result=owner.validate_composed_rib_reframe(path,manifest,raw)
                check.assert_called_once_with(base[0].resolve(),base[2],base[1])
            self.assertEqual([r['typed_identity'] for r in result['source_face_retirements']],[[51004,106],[51004,113]])
            self.assertEqual(result['thumb_reference_surface']['typed_identity'],[51004,93])
            self.assertEqual(manifest['source'],base[2]['source'])
            self.assertFalse(result['source_face_retirement']['physical_mass_or_state_changed'])
            changed=json.loads(json.dumps(manifest));changed['source']['identity']='drifted'
            with self.assertRaises(model.ImportError):owner.validate_composed_rib_reframe(path,changed,raw)

    def test_two_retirements_after_thumb(self):self.verify_order(['thumb',106,113])
    def test_thumb_between_two_retirements(self):self.verify_order([106,'thumb',113])
    def test_thumb_after_two_retirements(self):self.verify_order([106,113,'thumb'])
    def test_lineage_cycle_and_depth_fail_before_reading(self):
        for ancestors in ((Path('/tmp/bone').resolve(),),tuple(Path('/tmp')/str(i) for i in range(16))):
            with self.assertRaisesRegex(model.ImportError,'lineage'):owner.validate_composed_rib_reframe(Path('/tmp/bone'),{},b'',_lineage=ancestors)

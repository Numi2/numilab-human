"""An inferred thumb patch must not change any rigid or unrelated bone owner."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from numilab_human import composed_bone_body_frame as owner, model
from test_bone_opposed_source_face_retirement import fixture


def patch_fixture():
    raw = fixture()
    header, rows, vo, io = owner._decode_bones(raw)
    row = rows[92]
    vertices = np.frombuffer(raw, '<f4', count=header[3]*6, offset=vo).reshape(-1,6)[row[1]:row[1]+row[2]].copy()
    old_faces = np.frombuffer(raw, '<u4', count=row[4], offset=io+row[3]*4).reshape(-1,3)-row[1]
    kept = [1,2,3,5,6,7]
    face = old_faces[1]
    vertices = np.r_[vertices, vertices[face].mean(axis=0,keepdims=True)]
    faces = np.array([[face[0],face[1],8],[face[1],face[2],8],[face[2],face[0],8], *old_faces[kept[1:]]],dtype='<u4')
    ancestry = [dict(source_stable_id=93,source_face_index=i,orientation=1) for i in [1,1,1,*kept[1:]]]
    return raw, vertices, faces, ancestry


class ThumbPatchTests(unittest.TestCase):
    def apply(self, raw, vertices, faces, ancestry):
        return owner.replace_thumb_reference_surface(raw,vertices6=vertices,faces=faces,
              face_ancestry=ancestry,protected_source_slots=(0,1))

    def test_appended_slot_keeps_all_other_local_bytes_and_rigid_owners(self):
        raw, vertices, faces, ancestry = patch_fixture()
        out, proof = self.apply(raw,vertices,faces,ancestry)
        bh,br,bv,bi = owner._decode_bones(raw)
        ah,ar,av,ai = owner._decode_bones(out)
        self.assertEqual(ah[3],bh[3]+1)
        for old,new in zip(br,ar,strict=True):
            self.assertEqual(old[0],new[0]);self.assertEqual(old[5:],new[5:])
            if old[5] == 93:continue
            self.assertEqual(raw[bv+old[1]*24:bv+(old[1]+old[2])*24],out[av+new[1]*24:av+(new[1]+new[2])*24])
            before=np.frombuffer(raw,'<u4',count=old[4],offset=bi+old[3]*4)-old[1]
            after=np.frombuffer(out,'<u4',count=new[4],offset=ai+new[3]*4)-new[1]
            np.testing.assert_array_equal(before,after)
        self.assertFalse(proof['native_admission']);self.assertFalse(proof['anatomical_admission'])
        self.assertEqual(proof['original_source_slots_retained'],8)

    def test_attachment_movement_and_missing_slots_rejected(self):
        raw, vertices, faces, ancestry=patch_fixture()
        moved=vertices.copy();moved[0,0]=np.nextafter(np.float32(0),np.float32(1))
        with self.assertRaisesRegex(model.ImportError,'protected attachment'):self.apply(raw,moved,faces,ancestry)
        with self.assertRaises(model.ImportError):self.apply(raw,vertices[:7],faces,ancestry)

    def test_wrong_source_and_undeclared_borrowed_face_rejected(self):
        raw,vertices,faces,ancestry=patch_fixture()
        for changes in ({'source_stable_id':94}, {'source_stable_id':94,'inferred_articular_cap':True},
                        {'source_stable_id':99}, {'source_face_index':999}, {'orientation':True}):
            with self.subTest(changes=changes):
                a=[dict(x) for x in ancestry];a[0].update(changes)
                with self.assertRaises(model.ImportError):self.apply(raw,vertices,faces,a)

    def test_open_or_duplicate_output_rejected(self):
        raw,vertices,faces,ancestry=patch_fixture()
        for f,a in ((faces[:-1],ancestry[:-1]),(np.r_[faces,faces[:1]],ancestry+[ancestry[0]])):
            with self.assertRaises((model.ImportError,ValueError)):self.apply(raw,vertices,f,a)

    def make_pinned_case(self, root):
        raw,vertices,faces,ancestry=patch_fixture();out,proof=self.apply(raw,vertices,faces,ancestry)
        source=root/'source';source.mkdir();name='bodyparts3d-myosim-major-bones.nhbones';payload=source/name;payload.write_bytes(raw)
        prior={'payload':{'file':name,'sha256':model.sha256(payload),'bytes':len(raw),'vertex_count':owner._BONE_HEADER.unpack_from(raw)[3],'index_count':owner._BONE_HEADER.unpack_from(raw)[4]},'body_local_bakes':{'immutable':'rib owner'},'source':{'identity':'original'}}
        mp=source/'bodyparts3d-myosim-major-bones.manifest.json';mp.write_text(json.dumps(prior))
        output=root/'output';output.mkdir();np.savez(output/'patch.npz',vertices6=vertices,faces=faces);(output/'ancestry.json').write_text(json.dumps({'face_ancestry':ancestry,'inferred_not_measured':True}))
        manifest=json.loads(json.dumps(prior));h=owner._BONE_HEADER.unpack_from(out);manifest['payload'].update(sha256=hashlib.sha256(out).hexdigest(),bytes=len(out),vertex_count=h[3],index_count=h[4]);manifest['thumb_reference_surface']={'input_artifact':str(source),'input_manifest_sha256':model.sha256(mp),'input_payload_sha256':model.sha256(payload),'patch_path':'patch.npz','patch_sha256':model.sha256(output/'patch.npz'),'ancestry_path':'ancestry.json','ancestry_sha256':model.sha256(output/'ancestry.json'),'protected_source_slots':[0,1],'proof':proof}
        return output,manifest,out,prior

    def test_replay_rejects_manifest_output_and_asset_drift(self):
        with TemporaryDirectory() as d:
            output,m,out,prior=self.make_pinned_case(Path(d));got=owner.validate_thumb_reference_surface(output,m,out)
            self.assertEqual(got[1],prior)
            changed=json.loads(json.dumps(m));changed['source']['identity']='other'
            with self.assertRaisesRegex(model.ImportError,'unrelated manifest'):owner.validate_thumb_reference_surface(output,changed,out)
            with self.assertRaisesRegex(model.ImportError,'bytes or ancestry'):owner.validate_thumb_reference_surface(output,m,out[:-1]+b'!')
            (output/'ancestry.json').write_text('{}')
            with self.assertRaisesRegex(model.ImportError,'asset drifted'):owner.validate_thumb_reference_surface(output,m,out)


if __name__=='__main__':unittest.main()

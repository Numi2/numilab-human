import struct
import unittest
from numilab_human import composed_bone_body_frame as owner, model
from numilab_human import resting_anatomy_verify as verify
import numpy as np

H=struct.Struct("<8s5I32s")
R=struct.Struct("<6I8fI")

def fixture():
    points=[];indices=[];rows=[]
    for sid in range(1,186):
        fv=len(points);fi=len(indices)
        if sid==93:
            # Two tetrahedra glued along exact A/B/C, with a duplicated
            # oppositely oriented internal face in distinct source slots.
            p=[(0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.),
               (0.,0.,0.),(0.,1.,0.),(1.,0.,0.),(0.,0.,-1.)]
            faces=[(0,2,1),(0,1,3),(1,2,3),(2,0,3),
                   (4,6,5),(4,5,7),(5,6,7),(6,4,7)]
        else:
            p=[(0.,0.,0.),(1.,0.,0.),(0.,1.,0.)];faces=[(0,1,2)]
        points.extend(p);indices.extend(fv+i for f in faces for i in f)
        rows.append((sid%157,fv,len(p),fi,len(faces)*3,sid,
                     0.,0.,0.,0.,0.,0.,1.,1.,0))
    vertices=b"".join(struct.pack("<6f",*p,0.,0.,1.) for p in points)
    return (H.pack(b"NHBONES1",3,185,len(points),len(indices),123,b"x"*32)
            +b"".join(R.pack(*r) for r in rows)+vertices
            +struct.pack("<"+"I"*len(indices),*indices))

class RetirementTests(unittest.TestCase):
    def test_exact_source_retirement_preserves_every_other_owner(self):
        before=fixture()
        after,proof=owner.retire_exact_opposed_bone_source_faces(
            before,stable_id=93,source_face_pair=(0,4))
        bh,br,bv,bi=owner._decode_bones(before)
        ah,ar,av,ai=owner._decode_bones(after)
        self.assertEqual(after[av:ai],before[bv:bi])
        self.assertEqual(ah[4],bh[4]-6)
        self.assertEqual(proof["retained_source_face_ids"],[1,2,3,5,6,7])
        for old,new in zip(br,ar):
            self.assertEqual(old[:3],new[:3])
            self.assertEqual(old[5:],new[5:])
            if old[5]!=93:
                self.assertEqual(before[bi+old[3]*4:bi+(old[3]+old[4])*4],
                                 after[ai+new[3]*4:ai+(new[3]+new[4])*4])
        row=ar[92]
        verts=np.frombuffer(after[av:ai],'<f4').reshape(-1,6)
        faces=np.frombuffer(after[ai:],'<u4')[row[3]:row[3]+row[4]].reshape(-1,3)-row[1]
        surf=verify.prepare_surface((51004,93),verts[row[1]:row[1]+row[2],:3],faces)
        self.assertTrue(surf.closed)
        self.assertEqual(surf.self_audit()["count"],0)
        self.assertFalse(proof["anatomical_admission"])

    def test_near_coincidence_is_rejected(self):
        raw=bytearray(fixture());_,rows,vo,_=owner._decode_bones(raw)
        offset=vo+(rows[92][1]+4)*24
        struct.pack_into('<f',raw,offset,float(np.nextafter(np.float32(0.),np.float32(1.))))
        with self.assertRaisesRegex(model.ImportError,"exactly coincident"):
            owner.retire_exact_opposed_bone_source_faces(bytes(raw),stable_id=93,source_face_pair=(0,4))

    def test_same_winding_is_rejected(self):
        raw=bytearray(fixture());_,rows,_,io=owner._decode_bones(raw)
        offset=io+(rows[92][3]+12)*4
        a,b,c=struct.unpack_from('<3I',raw,offset)
        struct.pack_into('<3I',raw,offset,a,c,b)
        with self.assertRaisesRegex(model.ImportError,"oppositely wound"):
            owner.retire_exact_opposed_bone_source_faces(bytes(raw),stable_id=93,source_face_pair=(0,4))

    def test_wrong_face_or_bone_identity_is_rejected(self):
        for sid,pair in ((999,(0,4)),(93,(0,99)),(93,(4,0)),(93,(0,0))):
            with self.subTest(sid=sid,pair=pair),self.assertRaises(model.ImportError):
                owner.retire_exact_opposed_bone_source_faces(fixture(),stable_id=sid,source_face_pair=pair)

    def test_overlapping_source_index_owners_are_rejected(self):
        raw=bytearray(fixture());row=list(R.unpack_from(raw,H.size+94*R.size));row[3]-=3
        R.pack_into(raw,H.size+94*R.size,*row)
        with self.assertRaisesRegex(model.ImportError,"partition"):
            owner.retire_exact_opposed_bone_source_faces(bytes(raw),stable_id=93,source_face_pair=(0,4))

if __name__=="__main__":unittest.main()

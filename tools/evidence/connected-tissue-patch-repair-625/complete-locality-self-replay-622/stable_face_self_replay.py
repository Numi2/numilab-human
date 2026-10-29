"""Internal exact-locality replay of one pinned full surface audit.

All geometry and shared-vertex identities stay in the original global domain.
Face row reordering is mapped to persistent baseline IDs. Only byte-identical
retained triangles may reuse baseline pairs. Every added/changed triangle is
checked against the complete current surface by the existing exact owner.
"""
import collections

def stable_face_ids(old_faces, new_faces):
    old = {}
    for row, face in enumerate(old_faces):
        key = tuple(map(int, face))
        if key in old:
            raise ValueError("ambiguous duplicate baseline face")
        old[key] = row
    seen=set(); ids=[]; next_id=len(old_faces)
    for face in new_faces:
        key=tuple(map(int,face))
        if key in seen:
            raise ValueError("duplicate current face")
        seen.add(key)
        if key in old:
            ids.append(old[key])
        else:
            ids.append(next_id);next_id+=1
    return ids

def persistent_records(records, stable_ids):
    if len(records)!=len(stable_ids) or len(set(stable_ids))!=len(stable_ids):
        raise ValueError("invalid persistent face map")
    out=[]
    for row,record in enumerate(records):
        if record[3]!=row:
            raise ValueError("record input must use complete current face rows")
        out.append((*record[:3],int(stable_ids[row]),record[4]))
    return out

def report_pairs_to_rows(pairs, stable_ids):
    mapping={v:i for i,v in enumerate(stable_ids)}
    return sorted([sorted((mapping[int(a)],mapping[int(b)])) for a,b in pairs])

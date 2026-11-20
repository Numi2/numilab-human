"""Extract oriented O/I/D/wall faces and compare the full volume boundary chain."""
import csv
import json
import struct
import subprocess

import numpy as np

from candidate import COMPILER,load

old,x,c,mapping,_,_,_,old_c,old_m = load()
def surfaces(t,m):
    outward=np.array(((1,2,3),(0,3,2),(0,1,3),(0,2,1)))
    domains=[]
    for mat in (15,16):
        faces=t[m==mat][:,outward].reshape(-1,3)
        keys=np.sort(faces,axis=1)
        order=np.lexsort(keys.T[::-1])
        f=faces[order]
        k=keys[order]
        eq=np.all(k[1:]==k[:-1],axis=1)
        if np.any(eq[:-1]&eq[1:]):
            raise ValueError(f"nonmanifold facet in material {mat}")
        is_surface=~(np.r_[eq,False]|np.r_[False,eq])
        domains.append((f[is_surface],k[is_surface]))
    return domains

s0=surfaces(old_c,old_m)
s1=surfaces(c,mapping)
for (old_f,k0),(new_f,k1) in zip(s0,s1):
    assert np.array_equal(k0,k1), "a local topology repair changed the boundary"
    i=np.argmin(old_f,axis=1)
    j=np.argmin(new_f,axis=1)
    assert np.array_equal(old_f[np.arange(len(i))[:,None],(i[:,None]+np.arange(3))%3],
                          new_f[np.arange(len(j))[:,None],(j[:,None]+np.arange(3))%3])

def key(face):
    return tuple(sorted(map(int,face)))
proc=subprocess.Popen(["/opt/homebrew/bin/zstd","-qcd",str(COMPILER/"volume-export/volume/facet-ancestry.csv.zst")],
                      stdout=subprocess.PIPE,text=True)
records=csv.DictReader(proc.stdout)
by_role={}
for row in records:
    domain=row["face_source_layer"]
    face=key([int(row[f"joint_node{i}"])-1 for i in range(3)])
    by_role.setdefault(domain,set()).add(face)
assert proc.wait()==0
terms={}
surfaces=[]
both=set(map(key,s1[0][1]))&set(map(key,s1[1][1]))
for f,k in s1:
    terms[len(surfaces)]=len(f)
    used=[]
    for face,identity in zip(f,k):
        id=tuple(map(int,identity))
        if id in both and len(surfaces)==1:
            continue
        labels=[domain for domain, banks in by_role.items() if domain in ("O","I","D") and id in banks]
        label=labels[0] if len(labels)==1 else "wall"
        if id in both:
            assert label=="I"
        used.append([*map(int,face),["O","I","D","wall"].index(label)])
    surfaces.append(np.asarray(used,dtype="<i4"))
s=np.vstack(surfaces)
report={"boundary_faces_by_material":terms,"shared_skin_fat_faces":len(both),
        "audit_faces_by_role":np.bincount(s[:,3],minlength=4).tolist(),"exact_oriented_boundary_unchanged":True,
        "source_roles_present":{k:len(v) for k,v in by_role.items()}}
with open("fine-boundary-f32.bin","wb") as stream:
    stream.write(struct.pack("<II",len(x),len(s)))
    for p in (old,x):
        p.astype("<f4",copy=False).tofile(stream)
    s.tofile(stream)
with open("fine-boundary-preparation.json","w") as stream:
    json.dump(report,stream,indent=2)
print(report)

import struct
import numpy as np
import pytest
from numilab_human.passive_attachment_composition import _read_nhtiss4
from numilab_human.passive_reference_receipt_composition import _validate_parent_vertex_fields

def payload(path, *, faces=None, field=None, binding_shift=0., extra=False, layer=1):
    points=[(0.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)]
    if extra: points.append((.2,.2,.2))
    v=bytearray(b"".join(struct.pack("<6f4I4f",*p,0.,0.,1.,0,0,0,0,1.,0.,0.,0.) for p in points))
    if field is not None:
        offset,value,fmt=field
        struct.pack_into(fmt,v,offset,value)
    f=np.array(faces if faces is not None else [[0,2,1],[0,1,3],[0,3,2],[1,2,3]],dtype="<u4")
    raw=(struct.pack("<8s6I32s",b"NHTISS4\0",5,1,1,len(points),f.size,42,b"s"*32)
         +struct.pack("<8I",0,1,0,len(points),0,f.size,7,layer)
         +struct.pack("<I8f",20,binding_shift,0.,0.,0.,0.,0.,1.,1.)+v+f.tobytes())
    path.write_bytes(raw)
    return _read_nhtiss4(path)

def test_parent_index_only_edit_preserves_every_material_slot(tmp_path):
    a=payload(tmp_path/"a")
    b=payload(tmp_path/"b",faces=[[0,1,3],[0,3,2],[1,2,3],[0,2,1]])
    _validate_parent_vertex_fields(a,b,(7,))

@pytest.mark.parametrize("change",[
    {"field":(0,1e-8,"<f")}, {"field":(12,.01,"<f")},
    {"field":(24,1,"<I")}, {"field":(40,.99999,"<f")},
    {"binding_shift":1e-8}, {"extra":True}, {"layer":2},
])
def test_parent_field_or_allocation_edit_requires_reconstruction(tmp_path,change):
    a=payload(tmp_path/"a"); b=payload(tmp_path/"b",**change)
    with pytest.raises(ValueError,match="material-parent reconstruction"):
        _validate_parent_vertex_fields(a,b,(7,))

"""Native asset policy regression tests; no simulated physics on the CPU."""
import math
import struct

import pytest

from numilab_human.projected_joint_envelope import (
    polynomial_envelope, prepare_projected_joint_envelope,
)


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def native_horner(a, q, master_ref, dep_ref):
    x = f32(q-master_ref)
    result = a[-1]
    for value in reversed(a[:-1]):
        result = f32(value + f32(x*result))
    return f32(dep_ref+result)


def fixture():
    digest = bytes.fromhex("ab"*32)
    nq, nv, bodies, joints, sources = 10, 9, 1, 3, 1
    rigid = bytearray(struct.pack("<8s10I32s", b"NHRIGID2", 1, 5, sources,
                                 bodies, joints, nq, nv, 0, 0, 0, digest))
    rigid += struct.pack("<8I", 5,bodies,1,joints,0,0,nq,nv)+bytes(64)
    rigid += bytes(48+bodies*160+joints*144)
    for v in range(nv):
        q = v if v < 3 else (0xffffffff if v < 6 else v+1)
        limits = ([0.,2.] if v == 6 else [-2.,f32(.01)] if v == 7
                  else [-.5,.5] if v == 8 else [0.,0.])
        rigid += struct.pack("<8I8f", 0, 0xffffffff if v<6 else v-6, q, v,
                             v if v<6 else 0, 1 if v<6 else 4, 0,0,
                             *limits, 0,0, 0, .5, .01, 0)
    q0 = [0.]*nq
    q0[6], q0[8] = 1., f32(.01)
    rigid += struct.pack(f"<{nq+nv}f", *q0, *([0.]*nv))
    rigid += bytes(sources*32)
    a = list(map(f32, [.01,.02,-1.,.5,-.1]))
    row = struct.pack("<4I20f", 8,7,7,6, 0.,0.,*a[:2], *a[2:],0.,
                      .02,1.,0.,0., .9,.95,.001,.5, 2.,0.,0.,0.)
    eq = struct.pack("<8s10I32s", b"NHEQ1\0\0\0",1,nq,nv,1,96,1,0,0,0,0,digest)+row
    return bytes(rigid), eq


def test_whole_domain_covers_hidden_interior_maximum_and_native_rounding():
    # Both endpoint values miss the small maximum next to the straight knee.
    a = list(map(f32, [.010506, .0247615, -1.31647, .716337, -.138302]))
    domain = [0.,f32(2.0944)]
    result = polynomial_envelope(a,domain)
    lo,hi = result["range"]
    assert hi > a[0] and hi-a[0] < .0002
    assert lo < -1.79241
    # Actual single-precision execution, including cancellation, against a
    # certified continuous-domain bound; not the helper's Bernstein evaluation.
    samples = [f32(domain[1]*i/10000) for i in range(10001)]
    samples += [f32(.00947764)]
    for q in samples:
        assert lo <= native_horner(a,q,0.,0.) <= hi


@pytest.mark.parametrize("a,domain,ref,origin", [
    ([0.,4.,-4.,0.,0.],[0.,1.],0.,0.),
    ([1.,-4.,6.,-4.,1.],[0.,2.],-.5,.5),
    ([10000.,-10000.,0.,0.,0.],[0.,2.],-10000.,0.),
    ([.125,0.,0.,0.,0.],[-1.,1.],.25,.5),
    ([0.,0.,0.,0.,0.],[0.,0.],0.,0.),
])
def test_cancellation_shifted_domain_constant_and_zero(a,domain,ref,origin):
    bounds = polynomial_envelope(a,domain,ref,origin)["range"]
    for i in range(513):
        q = f32(domain[0]+(domain[1]-domain[0])*i/512)
        assert bounds[0] <= native_horner(a,q,origin,ref) <= bounds[1]


def test_only_authorized_auxiliary_limit_bytes_change():
    rigid, eq = fixture()
    candidate, receipt = prepare_projected_joint_envelope(rigid,eq,[7])
    assert receipt["changed_dof_count"] == 1
    assert receipt["changed_dofs"][0]["dependent_q_index"] == 8
    offset = 224+160+3*144+7*64+32
    assert candidate[:offset] == rigid[:offset]
    assert candidate[offset+8:] == rigid[offset+8:]
    assert struct.unpack_from("<2f",candidate,offset)[1] > f32(.01)
    # Reapplying preparation cannot keep inflating the envelope.
    repeated, again = prepare_projected_joint_envelope(candidate,eq,[7])
    assert repeated == candidate and again["changed_dof_count"] == 0


@pytest.mark.parametrize("selection", [[],[7,7],[8],[9],[True],[6]])
def test_invalid_or_unrepresented_master_fails_closed(selection):
    with pytest.raises(ValueError):
        prepare_projected_joint_envelope(*fixture(),selection)


def test_mismatched_digest_truncation_and_chained_owner_fail_closed():
    rigid,eq = fixture()
    for changed in [eq[:-1],eq[:48]+bytes(32)+eq[80:]]:
        with pytest.raises(ValueError):
            prepare_projected_joint_envelope(rigid,changed,[7])
    # Authoring dependent q8 as a master introduces a chain; no inferred domain.
    header=list(struct.unpack_from("<8s10I32s",eq));header[4]=header[6]=2
    extra=bytearray(eq[80:]);struct.pack_into("<4I",extra,0,9,8,8,7)
    changed=struct.pack("<8s10I32s",*header)+eq[80:]+extra
    with pytest.raises(ValueError,match="chained"):
        prepare_projected_joint_envelope(rigid,changed,[7])


@pytest.mark.parametrize("a,domain", [
    ([math.nan,0.,0.,0.,0.],[0.,1.]),
    ([0.,0.,0.,0.,0.],[1.,0.]),
    ([0.,0.,0.,0.,0.],[0.,math.inf]),
    ([.1,0.,0.,0.,0.],[0.,1.]),  # not an emitted Float32 coefficient
])
def test_bad_polynomial_inputs_are_not_silently_changed(a,domain):
    with pytest.raises(ValueError):
        polynomial_envelope(a,domain)

"""Stable face identities used by the material P1 registration compiler."""


def primary_source_face(group, faces, previous=None):
    """Return the supported prior face or the author's identity for this group.

    The compiler records its sorted distinct contributing faces in the support
    table. When more than one face contributes, its derived identity is the
    group number with bit 31 set.
    """
    if not len(faces):
        raise ValueError("A material source group must contain a face")
    if not 0 <= int(group) < 1 << 31:
        raise ValueError("Material source group exceeds the face identity domain")
    if previous is not None and previous in faces:
        return int(previous)
    if faces[0] == faces[-1]:
        return int(faces[0])
    return (1 << 31) + int(group)

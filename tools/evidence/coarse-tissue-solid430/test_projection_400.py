import importlib.util
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("builder", ROOT / "build_reduced_parent_boundary_400.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

A = np.array([0.2, -0.3, 0.7])
B = np.array([2.2, 0.1, 0.9])
C = np.array([0.8, 1.6, 1.2])

def project(q):
    tri = np.asarray([[A, B, C]], dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).reshape(1, 3)
    closest, bary, dist = builder.closest_batch(q, tri[:, 0][None, :, :], tri[:, 1][None, :, :], tri[:, 2][None, :, :])
    return closest[0, 0], bary[0, 0], float(dist[0, 0])

# An asymmetric interior barycentric point catches swapping the B/C coordinates.
known = np.array([0.2, 0.3, 0.5])
surface = known[0] * A + known[1] * B + known[2] * C
normal = np.cross(B - A, C - A)
normal = normal / np.linalg.norm(normal)
q, bary, distance = project(surface + 0.4 * normal)
assert np.allclose(q, surface, atol=1e-12), (q, surface)
assert np.allclose(bary, known, atol=1e-12), (bary, known)
assert abs(distance - 0.4) < 1e-12

# Edge-region point projects to the interior of AB, retaining the exact point.
edge = B - A
normal = np.cross(edge, C - A); normal = normal / np.linalg.norm(normal)
toward_c = C - A - np.dot(C - A, edge) / np.dot(edge, edge) * edge
outward = -toward_c / np.linalg.norm(toward_c)
q, bary, distance = project(A + 0.37 * edge + 0.1 * outward + 0.2 * normal)
expected = A + 0.37 * edge
assert np.allclose(q, expected, atol=1e-12), (q, expected)
assert np.allclose(bary, [0.63, 0.37, 0.0], atol=1e-12), bary

# Vertex-region point projects to C.
outward_c = (C - A) + (C - B)
outward_c = outward_c / np.linalg.norm(outward_c)
q, bary, distance = project(C + 0.4 * outward_c + 0.2 * normal)
assert np.allclose(q, C, atol=1e-12), (q, C)
assert np.allclose(bary, [0.0, 0.0, 1.0], atol=1e-12), bary

print("PASS: skewed triangle interior barycentric, edge, and vertex regions")

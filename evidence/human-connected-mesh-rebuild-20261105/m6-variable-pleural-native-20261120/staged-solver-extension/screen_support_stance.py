"""Screen three-point support stances using the native registered witnesses."""
from itertools import combinations
import json
from math import atan, hypot
from pathlib import Path
import re

log = Path("strict-equilibrium/stdout.log").read_text()
points = {}
for match in re.finditer(
    r"^resting_supported_source_witness row=(\d+) body=(\d+) world_m=\[([^]]+)\] gap_m=([^\n]+)$",
    log,
    re.M,
):
    row, body = (int(match[i]) for i in (1, 2))
    x, y, z = (float(v) for v in match[3].split(","))
    gap = float(match[4])
    assert gap == z
    points[row] = (x, y, z, body)
assert set(points) == set(range(32))
assert points[4][2] < 0.00005

# First accepted release COM. This is a geometric load-centrality screen.
motion = Path("staged-accuracy/stdout.log").read_text()
match = re.search(r"^human_standing_progress=accepted step=1 .*?center_of_mass_xyz_m=\[([^]]+)\]", motion, re.M)
assert match
cx, cy, _ = (float(v) for v in match[1].split(","))

def combine(rows):
    a, b, c = (points[i] for i in rows)
    ux, uy, uz = (b[i] - a[i] for i in range(3))
    vx, vy, vz = (c[i] - a[i] for i in range(3))
    signed_area_twice = ux * vy - vx * uy
    if abs(signed_area_twice) < 1e-8:
        return None
    sx = (uz * vy - vz * uy) / signed_area_twice
    sy = (ux * vz - vx * uz) / signed_area_twice
    normal_tilt = atan(hypot(sx, sy))
    if normal_tilt > 0.02:
        return None
    b_weight = ((cx - a[0]) * vy - vx * (cy - a[1])) / signed_area_twice
    c_weight = (ux * (cy - a[1]) - (cx - a[0]) * uy) / signed_area_twice
    a_weight = 1.0 - b_weight - c_weight
    minimum_remaining_gap = min(
        z - a[2] - sx * (x - a[0]) - sy * (y - a[1])
        for row, (x, y, z, _) in points.items() if row not in rows
    )
    return {
        "rows": rows,
        "body_indices": [points[i][3] for i in rows],
        "upward_normal_tilt_rad": normal_tilt,
        "minimum_other_gap_m": minimum_remaining_gap,
        "projected_xy_area_m2": 0.5 * abs(signed_area_twice),
        "center_of_mass_barycentric": [a_weight, b_weight, c_weight],
    }

feasible = [
    candidate for rows in combinations(points, 3)
    if (candidate := combine(rows)) is not None and candidate["minimum_other_gap_m"] >= -1e-6
]
feasible.sort(key=lambda row: (
    min(row["center_of_mass_barycentric"]),
    row["projected_xy_area_m2"],
), reverse=True)
result = {"points": points, "initial_rows": combine((4, 5, 6)), "qualified_lower_hull_stances": feasible}
Path("support-stance-screen.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: result[key] for key in ("initial_rows", "qualified_lower_hull_stances")}))

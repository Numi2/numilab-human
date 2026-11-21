"""Check exact local ray parity for distal sites, with a facial-opening exclusion proof."""
import json
from pathlib import Path

import numpy as np

from screen import clearance, exact, skin, skin_faces

edges=np.concatenate((skin_faces[:,[0,1]],skin_faces[:,[1,2]],skin_faces[:,[2,0]]))
edges.sort(axis=1)
unique,counts=np.unique(edges,axis=0,return_counts=True)
assert counts.max()<=2
boundary_vertices=np.unique(unique[counts==1])
assert len(boundary_vertices)>0
boundary_min_y=float(skin[boundary_vertices,1].min())
roof_max_z=float(skin[:,2].max())
records=clearance._exact_surface_records(skin,skin_faces)
locator=exact.prepare_point_location(records)
source=json.loads(Path("site-clearance.json").read_text())
proof=[]
for bank in source["surfaces"]:
    for route in bank["routes"]:
        for site in route["sites"][3:]:
            x=np.asarray(site["world_m"],dtype="<f4")
            lattice=exact.float32_point_lattice_key(x)
            # All possible indexed rays (1,k,k²) move no more up in y than
            # in z. If y + roof_z - z < min boundary y then no such ray can
            # hit an opening before leaving the mesh vertically.
            upper_y=float(x[1])+roof_max_z-float(x[2])
            assert upper_y<boundary_min_y,(route["name"],site["site_order"],upper_y,boundary_min_y)
            result=exact.point_location_prepared_indexed(lattice,locator)
            assert result["location"] in ("inside","outside"),result
            assert result["location"]==("inside" if site["outer_skin"]["winding"]>.5 else "outside")
            proof.append({"route":route["name"],"site_order":site["site_order"],
                          "world_float32_m":x.tolist(),"max_ray_y_in_surface_bounds_m":upper_y,
                          "is_distal_attachment":site["is_distal_attachment"],
                          "outer_surface_distance_mm":site["outer_skin"]["surface_distance_mm"],**result})
print(json.dumps({"skin_boundary_vertices":len(boundary_vertices),
                  "min_boundary_y_m":boundary_min_y,"max_skin_z_m":roof_max_z,
                  "exact_local_ray_parity":proof},indent=2))

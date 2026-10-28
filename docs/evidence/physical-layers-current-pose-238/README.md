# Physical layer geometry in current native poses

The complete directly placed Float32 outer skin has zero exact surface crossings against all 185 typed bone banks (177 active broad-phase banks, 63,959 skin faces, 26.72 seconds). This supplements its separate zero-self-intersection result. Every bank retains its pinned local triangles and COM offset and uses the actual native body pose. Surface crossing counts do not establish exhaustive solid containment, tissue volume, mass or motion qualification.

The old physical parent volume still has genuine defects. Local exact audits in closed banks 59/95 and 92/15 find 615 and 536 parent-tet-face/bone crossings, including 41 skin/fat-interface crossings in each bank. Retained winding samples show approximately 3.48/3.32 mm I/D penetration in 59/95. The files preserve the pre-correction frame audit as context and the actual-current-pose rerun as the authoritative result. Existing parent cells and source identities remain unchanged.

These results support preserving the passing outer surface while repairing connected inner skin/fat geometry. They admit no replacement volume and report zero physical steps. All execution was on SSH Numi; no visual verification was performed.

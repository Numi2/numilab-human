# Physical tissue boundary and conservative preparation checks

The current/reference coordinate distinction is explicit. Candidate002 contains 7,233 nodes and 28,632 positive tetrahedra. The canonical native410 input is reference Float32, not the captured current coordinate field; their maximum separation is about 0.487 mm. The displayed detailed source remains a separately pinned captured pose.

A reusable audit now extracts closed material boundaries from actual physical tetrahedra, cancelling shared faces, then intersects those solids with each of the 185 separately hashed bone operands. Both skin and fat boundaries pass manifold construction and signed-volume closure. The complete reference audit takes 0.50 s and records 185 positive material/bone pairs. This includes every nonzero Boolean volume without treating a small result as zero. These counts are neither triangle counts nor 185 independently certified macroscopic defects. Separate bank volumes overlap and must not be summed as a union.

An independent winding witness confirms a genuine fat/bone23/29 overlap: the point is 0.100 mm inside the fat boundary and 0.926 mm inside bone. The Boolean intersection is approximately 70.636 mL. Node-only inventories miss intersections through face and cell interiors. This gate must be rerun after every physical boundary repair, before another expensive native physical trial.

The reusable positive/empty intersection fixture, outward orientation, shared-face cancellation and nonmanifold incidence rejection pass. The corrected asymmetric triangle projection regression also passes; its frozen authoring source is included. Candidate002 remains unadmitted: previous physical trial419 accepted zero steps, and physical motion verification remains pending.

Identity correction: the reconstructed002 preparation report listed a later D run020 among its input files. The actual frozen builder and registration399 use D375 SHA4291970cf747616cad0a9cc9ae07cfc287a9f6515f80c11b2e7c89e51123f434. The old report is retained locally as erroneous evidence and is not used as a provenance certificate. registration399 below is the actual producer manifest. Native409 conservative mass and 417 source-only results remain independent native receipts, not motion qualification.

The authoring scripts retain the SSH Numi workspace paths in their frozen configuration; CLI audit inputs can be provided explicitly. No runtime material law, determinant limit, penetration limit, force owner, or mass tolerance was changed by this increment.

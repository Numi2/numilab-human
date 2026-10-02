# Whole-body source-overlap census

The [receipt](receipt-v4.json) compares all 198 compiled crossings against raw
BodyParts3D meshes in their declared source and rigid-owner frames. Every
crossing has matching mapped source witnesses; 187 match directly, 10 match
after exact source-face mapping for topology repairs, and the remaining pair's
two removed crossings are traced to opposite duplicate source faces removed by
the repair. The source and compiled counts are 34,806 and 34,804 triangle-pair
intersections, respectively.

See the [engineering note](../../WHOLE_BODY_SOURCE_OVERLAP_CENSUS_20261002.md)
for scope, interpretation and reproduction. This attributes geometry only; it
does not qualify anatomical seams or mechanics.

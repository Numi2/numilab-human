"""Apply each pectoral sheet layer's actual source point through the same donor map."""


def register_source_sheet(source):
    # The triangle, weights, donor identities and source anchor geometry are
    # common to both layers. The paired source coordinates carry the sheet
    # thickness and orientation, including the original outer boundary.
    replacements = {
        "                mr_float4 normal{0.0f, 0.0f, 0.0f, 0.0f};\n"
        "                mr_float4 remapResidualWorld": "                mr_float4 remapResidualWorld",
        """                const mr_float4 remapResidualLocal{
                    projection.remapResidualMeters[0],
                    projection.remapResidualMeters[1],
                    projection.remapResidualMeters[2],0.0f};""":
        """                const auto& sourceOuter =
                    fascia.nodes[region.firstNode + local % layerWidth];
                const mr_float4 remapResidualLocal{
                    projection.remapResidualMeters[0] +
                        (node.sourcePosition[0] - sourceOuter.sourcePosition[0]),
                    projection.remapResidualMeters[1] +
                        (node.sourcePosition[1] - sourceOuter.sourcePosition[1]),
                    projection.remapResidualMeters[2] +
                        (node.sourcePosition[2] - sourceOuter.sourcePosition[2]),
                    0.0f};""",
        """                    const mr_float4 vertexNormal =
                        softTissueVertexBlendedNormalWorld(*tissue, vertex, bodies);
                    normal.x += weight * vertexNormal.x;
                    normal.y += weight * vertexNormal.y;
                    normal.z += weight * vertexNormal.z;
""": "",
        """                // The registered muscle envelope is the deep fascia face.
                // Build the exterior face one sheet thickness outward.
                if (local < layerWidth) {
                    normal = femNormalized(normal,
                        "resting fascia projected source normal");
                    point.x += fascia.header.thicknessMeters * normal.x;
                    point.y += fascia.header.thicknessMeters * normal.y;
                    point.z += fascia.header.thicknessMeters * normal.z;
                }
""": "",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError(f"Unrecognized fascia source around {old[:80]!r}")
        source = source.replace(old, new)
    return source

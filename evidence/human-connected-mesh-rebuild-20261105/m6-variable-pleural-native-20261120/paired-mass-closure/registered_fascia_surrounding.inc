    if (const char* path = std::getenv("NUMI_HUMAN_REGISTERED_FASCIA_NEIGHBORS")) {
        std::ofstream neighbor(path, std::ios::binary);
        require(static_cast<bool>(neighbor), "could not open fascia surrounding geometry");
        std::array<float,3> lower{INFINITY,INFINITY,INFINITY};
        std::array<float,3> upper{-INFINITY,-INFINITY,-INFINITY};
        const auto pectoralEnd = fascia.regions[kPectoralisFasciaRegionCount-1u].firstNode +
                                 fascia.regions[kPectoralisFasciaRegionCount-1u].nodeCount;
        for (std::uint32_t i = 0u; i < pectoralEnd; ++i) {
            const std::array<float,3> p{points[i].x,points[i].y,points[i].z};
            for (std::size_t axis = 0; axis < p.size(); ++axis) {
                lower[axis] = std::min(lower[axis],p[axis]-0.030f);
                upper[axis] = std::max(upper[axis],p[axis]+0.030f);
            }
        }
        const auto record = [&](std::uint32_t kind, std::uint32_t stableID,
                                std::uint32_t face, std::array<mr_float4,3> triangle) {
            std::array<float,9> xyz{
                triangle[0].x,triangle[0].y,triangle[0].z,
                triangle[1].x,triangle[1].y,triangle[1].z,
                triangle[2].x,triangle[2].y,triangle[2].z};
            for (std::size_t i = 0; i < 3; ++i)
                if (std::max({xyz[i],xyz[i+3],xyz[i+6]}) < lower[i] ||
                    std::min({xyz[i],xyz[i+3],xyz[i+6]}) > upper[i]) return;
            const std::array<std::uint32_t,3> ids{kind,stableID,face};
            neighbor.write(reinterpret_cast<const char*>(ids.data()),sizeof(ids));
            neighbor.write(reinterpret_cast<const char*>(xyz.data()),sizeof(xyz));
        };
        for (const auto& tissue : tissues.records) {
            std::vector<mr_float4> posed(tissue.vertexCount);
            for (std::uint32_t row = 0; row < tissue.vertexCount; ++row)
                posed[row] = softTissueVertexBlendedWorld(
                    tissue,tissues.vertices[tissue.firstVertex+row],bodies);
            for (std::uint32_t f = 0; f < tissue.indexCount/3u; ++f) {
                const auto i = tissue.firstIndex+3u*f;
                record(1u,tissue.stableId,f,{
                    posed[tissues.indices[i]-tissue.firstVertex],
                    posed[tissues.indices[i+1]-tissue.firstVertex],
                    posed[tissues.indices[i+2]-tissue.firstVertex]});
            }
        }
        if (bones) for (const auto& bone : bones->records) {
            std::vector<mr_float4> posed(bone.vertexCount);
            for (std::uint32_t row = 0; row < bone.vertexCount; ++row)
                posed[row] = boneVertexWorld(
                    bone,bones->vertices[bone.firstVertex+row],bodies[bone.bodyIndex]);
            for (std::uint32_t f = 0; f < bone.indexCount/3u; ++f) {
                const auto i = bone.firstIndex+3u*f;
                record(2u,bone.stableId,f,{
                    posed[bones->indices[i]-bone.firstVertex],
                    posed[bones->indices[i+1]-bone.firstVertex],
                    posed[bones->indices[i+2]-bone.firstVertex]});
            }
        }
        if (weightedSkin) {
            std::vector<mr_float4> posed(weightedSkin->vertices.size());
            for (std::uint32_t row = 0; row < posed.size(); ++row)
                posed[row] = skinVertexWorld(
                    *weightedSkin,weightedSkin->vertices[row],row,bodies);
            for (std::uint32_t i = 0; i < weightedSkin->indices.size(); i += 3u)
                record(3u,0u,i/3u,{
                    posed[weightedSkin->indices[i]],
                    posed[weightedSkin->indices[i+1]],
                    posed[weightedSkin->indices[i+2]]});
        }
        require(static_cast<bool>(neighbor), "could not write fascia surrounding geometry");
    }

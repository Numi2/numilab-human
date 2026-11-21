from pathlib import Path
import re

def file(name):
    return Path("candidate", name)

def replace(text, old, new, count=1):
    assert text.count(old) == count, old
    return text.replace(old, new)

grouping = file("matter/include/numi/matter/fem_body_contact_sample_grouping.hpp")
text = grouping.read_text()
lo = text.index("// Detect an already grouped layout")
hi = text.index("// Deterministic sample-domain", lo)
text = text[:lo] + """// Reuse an already grouped layout without copying the immutable contact rows.
// Output one range per slave, in first-appearance order.
inline bool contiguousSlaveGroupRanges(
    std::span<const NMNumiHumanFEMBodyContactSampleGPU> samples,
    std::vector<NMIncidenceRangeGPU>& ranges
) {
    if (!ranges.empty() || samples.size() > std::numeric_limits<std::uint32_t>::max())
        return false;
    std::unordered_set<std::uint32_t> seen;
    for (std::size_t first = 0; first < samples.size();) {
        const auto slave = samples[first].slaveNode;
        if (!seen.insert(slave).second) {
            ranges.clear();
            return false;
        }
        std::size_t end = first + 1;
        while (end < samples.size() && samples[end].slaveNode == slave) ++end;
        ranges.push_back({static_cast<std::uint32_t>(first),
                          static_cast<std::uint32_t>(end - first), 0u, 0u});
        first = end;
    }
    return true;
}

""" + text[hi:]
text = replace(text, "#include <unordered_map>\n", "#include <unordered_map>\n#include <unordered_set>\n")
text = text.replace("""// the shared unchanged triangle buffer. Only each group head has a nonzero
// span count; non-head rows are zero.""", """// the shared unchanged triangle buffer. Each range indexes one complete group.""")
text = text.replace("std::vector<std::uint32_t>& groupCounts", "std::vector<NMIncidenceRangeGPU>& groupRanges")
text = text.replace("!groupCounts.empty()", "!groupRanges.empty()")
text = text.replace("    groups.reserve(samples.size());\n", "")
text = text.replace("    groupCounts.reserve(samples.size());", "    groupRanges.reserve(groups.size());")
text = text.replace("        const auto span = static_cast<std::uint32_t>(group.size());", """        groupRanges.push_back({static_cast<std::uint32_t>(groupedSamples.size()),
                               static_cast<std::uint32_t>(group.size()), 0u, 0u});""")
text = text.replace("            groupCounts.push_back(local == 0u ? span : 0u);\n", "")
text = text.replace("        groupCounts.size() == samples.size() &&\n", "")
grouping.write_text(text)

names = {
    "nm_numi_human_assemble_fem_body_contact_loads_grouped_cached": "nm_numi_human_assemble_fem_body_contact_loads_slave_ranges_cached",
    "nm_numi_human_assemble_fem_body_contact_loads_grouped": "nm_numi_human_assemble_fem_body_contact_loads_slave_ranges",
}

shader = file("matter/src/metal/numi_human.metalinc")
text = shader.read_text()
for before, after in names.items():
    text = text.replace(before, after)
text = replace(text, "    device const uint* groupCounts [[buffer(9)]],",
               "    device const NMIncidenceRangeGPU* slaveRanges [[buffer(9)]],", 2)
for name in names.values():
    first = text.index("kernel void "+name+"(")
    position = text.index("    uint global [[thread_position_in_grid]]", first)
    text = text[:position] + "    constant uint& rangeCount [[buffer(11)]],\n" + text[position:]
old = """    const uint total = dispatch.environmentCount *
        dispatch.femBodyContactSampleCount;
    if (global >= total || dispatch.femBodyContactSampleCount == 0u) return;
    const uint environment = global / dispatch.femBodyContactSampleCount;
    const uint head = global - environment * dispatch.femBodyContactSampleCount;
    const uint span = groupCounts[head];
    if (span == 0u || span > dispatch.femBodyContactSampleCount-head) return;
"""
new = """    if (rangeCount == 0u || global / rangeCount >= dispatch.environmentCount)
        return;
    const uint environment = global / rangeCount;
    const NMIncidenceRangeGPU range = slaveRanges[global % rangeCount];
    const uint head = range.first;
    const uint span = range.count;
    if (span == 0u || head >= dispatch.femBodyContactSampleCount ||
        span > dispatch.femBodyContactSampleCount - head) return;
"""
shader.write_text(replace(text, old, new, 2))

runtime = file("matter/src/numi_human.mm")
text = runtime.read_text()
text = text.replace("femBodyContactSolverGroupCounts", "femBodyContactSolverGroupRanges")
text = text.replace("femBodyContactSolverGroupCountBuffer", "femBodyContactSolverGroupRangeBuffer")
text = text.replace("contiguousSlaveGroupCounts", "contiguousSlaveGroupRanges")
text = text.replace("std::vector<std::uint32_t> femBodyContactSolverGroupRanges;",
                    "std::vector<NMIncidenceRangeGPU> femBodyContactSolverGroupRanges;")
for before, after in names.items():
    text = text.replace(before, after)
text = replace(text, """                    state_->femBodyContactSolverGroupRangeBuffer =
                        [device newBufferWithBytes:
                            state_->femBodyContactSolverGroupRanges.data()
                        length:state_->femBodyContactSolverGroupRanges.size() *
                            sizeof(std::uint32_t)
                        options:MTLResourceStorageModeShared];""", """                    const auto& ranges = state_->femBodyContactSolverGroupRanges;
                    const NMIncidenceRangeGPU empty{};
                    state_->femBodyContactSolverGroupRangeBuffer =
                        [device newBufferWithBytes:ranges.empty() ? &empty : ranges.data()
                        length:std::max<std::size_t>(1u, ranges.size()) *
                            sizeof(NMIncidenceRangeGPU)
                        options:MTLResourceStorageModeShared];""")
text = replace(text, """        if (!refreshGlides && state_->femBodyContactSolverSamplesCount() != 0u &&
            !encodeKernel(""", """        const auto bodyContactGroupCount = static_cast<std::uint32_t>(
            state_->femBodyContactSolverGroupRanges.size());
        if (!refreshGlides && state_->femBodyContactSolverSamplesCount() != 0u &&
            !encodeKernel(""")
text = replace(text, """                state_->environmentCount * state_->femBodyContactSolverSamplesCount(),
                [&](id<MTLComputeCommandEncoder> encoder) {""", """                std::size_t(state_->environmentCount) * (state_->hasMultiMasterTriangleContact
                    ? bodyContactGroupCount : state_->femBodyContactSolverSamplesCount()),
                [&](id<MTLComputeCommandEncoder> encoder) {""")
text = replace(text, """                        [encoder setBuffer:state_->femBodyContactSolverGroupRangeBuffer
                                    offset:0u atIndex:9u];""", """                        [encoder setBuffer:state_->femBodyContactSolverGroupRangeBuffer
                                    offset:0u atIndex:9u];
                        [encoder setBytes:&bodyContactGroupCount
                                    length:sizeof(bodyContactGroupCount) atIndex:11u];""")
runtime.write_text(text)

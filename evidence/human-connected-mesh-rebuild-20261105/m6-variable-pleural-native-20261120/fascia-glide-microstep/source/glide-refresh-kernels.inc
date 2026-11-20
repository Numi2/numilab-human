
// Fixed master pose within the standalone split transaction, as in the
// original regional glides. Sample the live slave each due FEM microstep and
// use that same force and point for the opposing rigid impulse.
kernel void nm_human_fascia_refresh_plane_load(
    constant NMMatterDispatchGPU& dispatch [[buffer(0)]],
    constant NMMicrostepGPU& micro [[buffer(1)]],
    constant NMNumiHumanTendonFEMLoadDispatchGPU& contact [[buffer(2)]],
    device const NMNumiHumanFEMBodyContactSampleGPU* samples [[buffer(3)]],
    device const NMFEMNodeStateGPU* accepted [[buffer(4)]],
    device const MRArticulatedBodyPoseGPU* poses [[buffer(5)]],
    device float4* externalForces [[buffer(6)]],
    device NMNumiHumanBodyWrenchGPU* rowImpulses [[buffer(7)]],
    device const NMIncidenceRangeGPU* ranges [[buffer(8)]],
    device const NMContinuumObjectGPU* objects [[buffer(9)]],
    device const NMSchedulerStateGPU* schedulers [[buffer(10)]],
    device const NMAdaptiveStateGPU* adaptive [[buffer(11)]],
    device const NMMatterStatusGPU* statuses [[buffer(12)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= contact.femBodyContactSampleCount) return;
    if (micro.microtick == 0u) rowImpulses[row] = {};
    if (!environmentHealthy(statuses,0u)) return;
    const auto sample = samples[row];
    const uint node = sample.slaveNode;
    const uint object = ranges[node].objectIndex;
    if (object >= dispatch.objectCount ||
        !objectDue(objects[object],schedulers[object],adaptive[object],micro)) return;
    const auto pose = poses[sample.bodyIndex-contact.articulationFirstBody];
    float3 slavePoint, point, normal;
    float closure, area, slip;
    float3x3 tangent;
    if (!humanFEMBodyContactState(
            sample,accepted[node],pose,slavePoint,point,normal,closure,
            area,tangent,slip)) {
        externalForces[node] = float4(NAN);
        return;
    }
    // Admission covers an exact one-row-per-node domain and all other nodal
    // loads are absent for these rows. A direct assignment avoids atomics.
    const float3 force = normal*(sample.stiffnessAndNormalStrainPerPressure.x*closure*area);
    externalForces[node] = float4(force,0.0f);
    const float3 impulse = -force*objectTimestep(dispatch,schedulers[object]);
    rowImpulses[row].force.xyz += impulse;
    rowImpulses[row].torque.xyz += cross(point-pose.position.xyz,impulse);
}

kernel void nm_human_fascia_finish_plane_wrench(
    constant NMNumiHumanTendonFEMLoadDispatchGPU& contact [[buffer(0)]],
    constant float& reciprocalFrameDt [[buffer(1)]],
    device const NMNumiHumanFEMBodyContactSampleGPU* samples [[buffer(2)]],
    device const NMNumiHumanBodyWrenchGPU* rowImpulses [[buffer(3)]],
    device NMNumiHumanBodyWrenchGPU* wrenches [[buffer(4)]],
    device const NMMatterStatusGPU* statuses [[buffer(5)]],
    uint body [[thread_position_in_grid]]
) {
    if (body >= contact.bodyPoseStride) return;
    float3 force(0),torque(0);
    if (environmentHealthy(statuses,0u)) {
        for (uint row = 0u; row < contact.femBodyContactSampleCount; ++row) {
            if (samples[row].bodyIndex != contact.articulationFirstBody+body) continue;
            force += rowImpulses[row].force.xyz;
            torque += rowImpulses[row].torque.xyz;
        }
    }
    wrenches[body].force = float4(force*reciprocalFrameDt,0);
    wrenches[body].torque = float4(torque*reciprocalFrameDt,0);
}

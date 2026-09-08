#include "metalrobo/engine_types.h"
#include "numi/matter/shared.h"

// Qualification-only scalar momentum equation. The contact response and its
// derivative are produced by the unmodified production support kernel.
kernel void diagnostic_newton_update(
    constant float4& parameters [[buffer(0)]],
    constant uint& contactCount [[buffer(1)]],
    device MRBodyStateGPU* bodies [[buffer(2)]],
    device const NMContactSampleGPU* samples [[buffer(3)]],
    device float* generalized [[buffer(4)]]) {
    const float mass = parameters.x;
    const float dt = parameters.y;
    const float freeVelocity = parameters.z;
    float impulse = 0.0f, derivative = 0.0f;
    for (uint i = 0; i < contactCount; ++i) {
        impulse += samples[i].impulseAndNormal.y;
        derivative += samples[i].barrierHessianRow1.y;
    }
    const float previous = bodies[0].linearVelocityAndInverseMass.y;
    const float residual = mass * (freeVelocity - previous) + impulse;
    const float velocity = previous + residual / (mass + derivative);
    bodies[0].linearVelocityAndInverseMass.y = velocity;
    bodies[0].position.y = dt * velocity;
    generalized[0] = velocity - freeVelocity;
}

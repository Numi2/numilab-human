// Solve both edge parameters before projecting either parameter or position.
// A rounded parameter can leave a tangential component in the shortest
// vector, reversing the energy slope of an almost tangential correction.
inline MRCompensatedScalar contactClampUnitParameter(MRCompensatedScalar value) {
    if (contactCompensatedSign(value) <= 0) return {0.0f, 0.0f};
    if (contactCompensatedSign(contactCompensatedSubtract(value, {1.0f, 0.0f})) >= 0)
        return {1.0f, 0.0f};
    return value;
}

inline bool contactCompensatedSegmentWitness(
    const MRCompensatedPositionGPU p0,
    const MRCompensatedPositionGPU p1,
    const MRCompensatedPositionGPU q0,
    const MRCompensatedPositionGPU q1,
    thread float& firstParameter,
    thread float& secondParameter,
    thread float3& firstPoint,
    thread float3& secondPoint,
    thread MRCompensatedPositionGPU& relative,
    thread float& separation
) {
    const auto initial = mrCompensatedVectorAdd(p0, mrCompensatedVectorNegate(q0));
    const auto firstEdge = mrCompensatedVectorAdd(p1, mrCompensatedVectorNegate(p0));
    const auto secondEdge = mrCompensatedVectorAdd(q1, mrCompensatedVectorNegate(q0));
    const float3 dimensions = max(abs(contactCompensatedPositionProjection(initial)),
        max(abs(contactCompensatedPositionProjection(firstEdge)),
            abs(contactCompensatedPositionProjection(secondEdge))));
    const float scale = max(dimensions.x, max(dimensions.y, dimensions.z));
    if (!isfinite(scale)) {
        relative = mrCompensatedVector(float4(NAN));
        separation = NAN;
        return false;
    }

    MRCompensatedScalar s = {0.0f, 0.0f};
    MRCompensatedScalar t = {0.0f, 0.0f};
    if (scale > 0.0f) {
        const auto inverse = mrCompensatedDivide({1.0f, 0.0f}, {scale, 0.0f});
        const auto u = mrCompensatedVectorScale(firstEdge, inverse);
        const auto v = mrCompensatedVectorScale(secondEdge, inverse);
        const auto r = mrCompensatedVectorScale(initial, inverse);
        const auto a = mrCompensatedVectorDot(u, u);
        const auto e = mrCompensatedVectorDot(v, v);
        const auto c = mrCompensatedVectorDot(u, r);
        const auto f = mrCompensatedVectorDot(v, r);
        if (contactCompensatedSign(a) <= 0) {
            if (contactCompensatedSign(e) > 0)
                t = contactClampUnitParameter(mrCompensatedDivide(f, e));
        } else if (contactCompensatedSign(e) <= 0) {
            s = contactClampUnitParameter(mrCompensatedDivide(mrCompensatedNegate(c), a));
        } else {
            const auto b = mrCompensatedVectorDot(u, v);
            const auto perpendicular = mrCompensatedVectorCross(u, v);
            const auto denominator = mrCompensatedVectorDot(perpendicular, perpendicular);
            if (contactCompensatedSign(denominator) > 0) {
                const auto numerator = mrCompensatedVectorDot(
                    mrCompensatedVectorCross(v, r), perpendicular);
                s = contactClampUnitParameter(mrCompensatedDivide(numerator, denominator));
            }
            t = mrCompensatedDivide(mrCompensatedAdd(mrCompensatedMultiply(b, s), f), e);
            if (contactCompensatedSign(t) < 0) {
                t = {0.0f, 0.0f};
                s = contactClampUnitParameter(mrCompensatedDivide(mrCompensatedNegate(c), a));
            } else if (contactCompensatedSign(contactCompensatedSubtract(t, {1.0f, 0.0f})) > 0) {
                t = {1.0f, 0.0f};
                s = contactClampUnitParameter(mrCompensatedDivide(contactCompensatedSubtract(b, c), a));
            }
        }
    }
    if (!isfinite(s.high) || !isfinite(s.low) || !isfinite(t.high) || !isfinite(t.low))
        return false;
    firstParameter = s.high + s.low;
    secondParameter = t.high + t.low;
    const auto firstOffset = mrCompensatedVectorScale(firstEdge, s);
    const auto secondOffset = mrCompensatedVectorScale(secondEdge, t);
    firstPoint = contactCompensatedPositionProjection(mrCompensatedVectorAdd(p0, firstOffset));
    secondPoint = contactCompensatedPositionProjection(mrCompensatedVectorAdd(q0, secondOffset));
    relative = mrCompensatedVectorAdd(mrCompensatedVectorAdd(initial, firstOffset),
                                      mrCompensatedVectorNegate(secondOffset));
    const auto squared = mrCompensatedVectorDot(relative, relative);
    if (contactCompensatedSign(squared) < 0 || !isfinite(squared.high) || !isfinite(squared.low)) {
        separation = NAN;
        return false;
    }
    const auto distance = squared.high == 0.0f && squared.low == 0.0f
        ? MRCompensatedScalar{0.0f, 0.0f}
        : mrCompensatedSqrt(squared);
    separation = distance.high + distance.low;
    return isfinite(separation) && separation >= 0.0f;
}


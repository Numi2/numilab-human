    if (const char* audit = std::getenv("NUMI_HUMAN_INITIAL_GRAVITY_LEDGER");
        audit && std::string_view(audit) == "1") {
        // Use the same rebased articulated FK as the dynamics. A donor frame
        // used for the skin registration is not the residual rigid COM.
        metalrobo::MetalArticulatedOperatorResult gravityPose;
        const auto fk = metalrobo::runMetalArticulatedOperator(rebased, poseInput, gravityPose, poseConfig);
        require(fk.succeeded() && fk.dispatched && gravityPose.bodyPoses.size() == rebased.bodies.size(),
                "initial gravitational wrench requires the articulated COM poses");
        using GravityRow = std::array<double, 4>;
        GravityRow total{};
        auto add=[](GravityRow& r, double m, const auto& p) {
            require(m >= 0 && std::isfinite(m) && std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z),
                    "nonfinite mass or coordinate in initial gravitational wrench");
            r[0] += m; r[1] += m*p.x; r[2] += m*p.y; r[3] += m*p.z;
        };
        auto emit=[&](int object, const GravityRow& r) {
            for (unsigned axis=0; axis<4; ++axis) total[axis] += r[axis];
            std::cout << std::setprecision(17) << "initial_gravity_component={\"object\":" << object
                      << ",\"mass_kg\":" << r[0] << ",\"first_moment_kg_m\":["
                      << r[1] << ',' << r[2] << ',' << r[3] << "]}" << std::endl;
        };
        GravityRow skeleton{};
        for (size_t i=0; i<rebased.bodies.size(); ++i)
            add(skeleton,rebased.bodies[i].massAndInverseMass.x,gravityPose.bodyPoses[i].position);
        emit(-1,skeleton);
        std::size_t covered = 0;
        for (size_t i=0; i<compiled.world.objects.size(); ++i) {
            const auto& object = compiled.world.objects.at(i);
            require(object.representation == NM_REPRESENTATION_FEM && object.stateOffset == covered,
                    "initial gravity inventory has a missing or duplicated FEM range");
            GravityRow r{};
            for (size_t n=0; n<object.stateCount; ++n) {
                const auto p=compiled.world.fem.nodes.at(covered+n).positionAndMass;
                add(r,p.w,p);
            }
            covered += object.stateCount;
            emit(int(i),r);
        }
        require(covered == compiled.world.fem.nodes.size() && std::abs(total[0]-executedPhysicalMass)<1.e-5,
                "initial gravitational wrench disagrees with the source mass ledger");
        std::cout << std::setprecision(17) << "initial_gravity_total={\"mass_kg\":" << total[0]
                  << ",\"center_m\":[" << total[1]/total[0] << ',' << total[2]/total[0] << ','
                  << total[3]/total[0] << "],\"fem_nodes\":" << covered << "}" << std::endl;
    }

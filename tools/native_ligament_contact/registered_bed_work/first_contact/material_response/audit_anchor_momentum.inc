        if (std::getenv("NUMI_HUMAN_FIRST_ROOT_ANCHOR_MOMENTUM") != nullptr) {
            require(stepCount==1 && a.femNodes.size()==z.femNodes.size() &&
                    a.femNodes.size()==continuumTransaction->restingFEMAuditAnchorBodies.size(),
                    "First-root anchored momentum requires matching native start and endpoint states");
            for (const auto& range : continuumTransaction->restingFEMAuditRanges) {
                std::array<double,3> impulse{};
                double sumMagnitude=0.0, peakSpeedChange=0.0, peakTravel=0.0;
                std::uint32_t count=0, worst=NM_INVALID_INDEX;
                for (std::uint32_t i=0; i<range.nodeCount; ++i) {
                    const auto node=range.firstNode+i;
                    if (continuumTransaction->restingFEMAuditAnchorBodies.at(node)==NM_INVALID_INDEX) continue;
                    const auto& current=a.femNodes.at(node);
                    const auto& prior=z.femNodes.at(node);
                    const double m=prior.positionAndMass.w;
                    const auto p=current.positionAndMass, p0=prior.positionAndMass;
                    const auto v=current.velocityAndInverseMass, v0=prior.velocityAndInverseMass;
                    const std::array<double,3> dv={double(v.x)-v0.x,double(v.y)-v0.y,double(v.z)-v0.z};
                    const double speed=std::hypot(dv[0],dv[1],dv[2]);
                    require(std::isfinite(speed) && m>0.0 && current.positionAndMass.w==prior.positionAndMass.w,
                            "Invalid registered anchored node");
                    if (speed>peakSpeedChange) { peakSpeedChange=speed; worst=i; }
                    // Use the compensated material coordinate where it is available.
                    double dp[3]={double(p.x)-p0.x,double(p.y)-p0.y,double(p.z)-p0.z};
                    if(current.referenceDisplacementAndMode.w==1 && prior.referenceDisplacementAndMode.w==1) {
                        const auto h=current.referenceDisplacementAndMode, h0=prior.referenceDisplacementAndMode;
                        const auto l=current.referenceDisplacementLow, l0=prior.referenceDisplacementLow;
                        dp[0]=(double(h.x)-h0.x)+(double(l.x)-l0.x);
                        dp[1]=(double(h.y)-h0.y)+(double(l.y)-l0.y);
                        dp[2]=(double(h.z)-h0.z)+(double(l.z)-l0.z);
                    }
                    peakTravel=std::max(peakTravel,std::hypot(dp[0],dp[1],dp[2]));
                    for(int k=0;k<3;k++) impulse[k]-=m*dv[k];
                    if (std::getenv("NUMI_HUMAN_FIRST_ROOT_RELEASE_TARGETS")!=nullptr) {
                        const auto originalPrecision=std::cout.precision();
                        std::cout<<std::setprecision(17)<<"first_root_fixed_displacement="<<node
                            <<" "<<dp[0]<<" "<<dp[1]<<" "<<dp[2]<<" "<<dv[0]<<" "<<dv[1]<<" "<<dv[2]<<"\n";
                        std::cout.precision(originalPrecision);
                    }
                    sumMagnitude+=m*speed;
                    ++count;
                }
                if(count) std::cout<<std::setprecision(17)<<"first_root_anchor_momentum="
                    <<"{\"object\":\""<<range.objectName<<"\",\"nodes\":"<<count<<",\"peak_node\":"<<worst
                    <<",\"peak_speed_change_m_s\":"<<peakSpeedChange<<",\"peak_travel_m\":"<<peakTravel
                    <<",\"opposite_nodal_momentum_change_Ns\":["<<impulse[0]<<","<<impulse[1]<<","<<impulse[2]
                    <<"],\"sum_momentum_change_magnitudes_Ns\":"<<sumMagnitude<<"}"<<std::endl;
            }
        }

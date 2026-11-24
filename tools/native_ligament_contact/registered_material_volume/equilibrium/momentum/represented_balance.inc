            bool usedTranslationBalance=false;
            if(!(actual>lo && actual<hi)){
                if(privateCount!=4u)return 8u;
                float3 impulse=0.f, originalError=0.f, sumAbs=0.f;
                float totalMass=0.f;
                for(uint k=0;k<4;++k){
                    const float mass=candidate[t.nodes[k]].positionAndMass.w;
                    const float3 oldDirection=direction[t.nodes[k]].xyz;
                    const float3 delta=beforeBalance[k]-oldDirection;
                    totalMass+=mass;
                    impulse=fma(mass,delta,impulse);
                    originalError=fma(mass,next[k]-oldDirection,originalError);
                    sumAbs+=mass*abs(delta);
                }
                // A common velocity translation preserves all six relative
                // edge velocities. Recheck the actual Float32 emission below.
                const float3 common=impulse/totalMass;
                const float3 tolerance=max(abs(originalError),8.f*0x1p-23f*sumAbs);
                float3 shifted[4],best[4];
                float3 remainder=0.f;
                for(uint k=0;k<4;++k){
                    shifted[k]=beforeBalance[k]-common;
                    const float mass=candidate[t.nodes[k]].positionAndMass.w;
                    remainder=fma(mass,shifted[k]-direction[t.nodes[k]].xyz,remainder);
                }
                float bestError=INFINITY;
                // Try the uniform update first. A single representable last
                // adjustment may close its remaining roundoff more accurately.
                for(uint last=0;last<=4u;++last){
                    float3 balanceError=0.f,values[4];
                    for(uint k=0;k<4;++k){
                        values[k]=shifted[k]-(last==k+1u?remainder*w[k]:float3(0.f));
                        const float mass=candidate[t.nodes[k]].positionAndMass.w;
                        balanceError=fma(mass,values[k]-direction[t.nodes[k]].xyz,balanceError);
                        p[k]=femTrialNodeAtAlpha(accepted[t.nodes[k]],candidate[t.nodes[k]],values[k],1.f,dt,true);
                    }
                    if(!all(isfinite(balanceError)) || any(abs(balanceError)>tolerance))continue;
                    const float Jtrial=determinant(registeredTrialF(p,inv,reference));
                    if(!(Jtrial>lo && Jtrial<hi))continue;
                    const float error=length(balanceError);
                    if(error>=bestError)continue;
                    bestError=error;
                    for(uint k=0;k<4;++k)best[k]=values[k];
                }
                if(!isfinite(bestError))return 8u;
                for(uint k=0;k<4;++k)next[k]=best[k];
                usedTranslationBalance=true;
            }

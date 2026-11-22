            // Adding a small correction to a large velocity can lose enough
            // bits to break the zero-resultant property of the search update.
            // Balance the emitted values on the most representable component.
            for(uint axis=0;axis<3;++axis){
                float momentum=0.f,finest=INFINITY;uint pick=0;
                for(uint k=0;k<4;++k){
                    const float mass=candidate[t.nodes[k]].positionAndMass.w;
                    momentum=fma(mass,next[k][axis]-direction[t.nodes[k]][axis],momentum);
                    const float resolution=max(abs(next[k][axis]),1.e-30f)*mass;
                    if(resolution<finest){finest=resolution;pick=k;}
                }
                next[pick][axis]-=momentum*w[pick];
            }
            for(uint k=0;k<4;++k)
                p[k]=femTrialNodeAtAlpha(accepted[t.nodes[k]],candidate[t.nodes[k]],next[k],1.f,dt,true);
            const float actual=determinant(registeredTrialF(p,inv,reference));
            if(!(actual>lo && actual<hi))return;

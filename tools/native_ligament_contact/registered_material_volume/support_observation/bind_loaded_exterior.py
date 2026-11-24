"""Initialize the presentation from precisely the material coordinates installed into Matter."""
def bind(source, once):
    needle='            if(numiLoadedSkin::enabled() && !wholeBodySkinRegisteredOuterTetrahedral &&'
    source=once(source,needle,'''            if(numiLoadedSkin::enabled()) {
                for(std::size_t row=0;row<wholeBodySkinRegisteredOuterVolumeNodeRows.size();++row)
                    wholeBodySkinCurrentBaseline.at(row)=installedObject.femNodes.at(
                        wholeBodySkinRegisteredOuterVolumeNodeRows[row]);
                if(wholeBodySkinRegisteredOuterTetrahedral) {
                    require(wholeBodySkinRegisteredVolumeAcceptedPositions.size()==installedObject.femNodes.size(),
                        "loaded tetrahedral presentation does not cover the installed volume");
                    for(std::size_t i=0;i<installedObject.femNodes.size();++i) {
                        const auto& p=installedObject.femNodes[i];
                        wholeBodySkinRegisteredVolumeAcceptedPositions[i]={float(p[0]),float(p[1]),float(p[2]),1.f};
                    }
                }
            }
'''+needle)
    source=once(source,'restingPassiveFasciaOwner->wholeBodySkinCoarseRest,true);','restingPassiveFasciaOwner->wholeBodySkinCurrentBaseline,true);')
    anchor='''                                restingPassiveFasciaOwner->wholeBodySkinRegisteredVolumeReferencePoints);
                        else if'''
    source=once(source,'restingPassiveFasciaOwner->wholeBodySkinRegisteredOuterTetrahedral)\n                            liveVisual->configureAcceptedFEMSkinTetrahedralTransfer','restingPassiveFasciaOwner->wholeBodySkinRegisteredOuterTetrahedral) {\n                            liveVisual->configureAcceptedFEMSkinTetrahedralTransfer')
    return once(source,anchor,'''                                restingPassiveFasciaOwner->wholeBodySkinRegisteredVolumeReferencePoints);
                            if(numiLoadedSkin::enabled())
                                liveVisual->updateAcceptedFEMSkinTetrahedralPositions(
                                    restingPassiveFasciaOwner->wholeBodySkinRegisteredVolumeAcceptedPositions);
                        } else if''')

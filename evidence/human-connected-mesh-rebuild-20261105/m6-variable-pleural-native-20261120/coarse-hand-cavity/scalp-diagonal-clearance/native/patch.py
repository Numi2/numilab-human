"""Exact rewrites of the pinned previous cavity overlay."""

def patch(source, once):
    anchor = '        #include "PosteriorBedReference.inc"'
    source = once(source, anchor, anchor + r'''
        if(posteriorBedReference) {
            require(region.faces.size()==kWholeBodySkinFaces.size() &&
                    std::equal(region.faces.begin(),region.faces.end(),kWholeBodySkinFaces.begin()),
                    "derived scalp reference does not match the original registered topology");
            for(const auto& edit:kScalpFaceRows) {
                require(edit[0]<region.faces.size() &&
                        region.faces[edit[0]]==Face{edit[1],edit[2],edit[3]},
                        "derived scalp edit does not match its pinned source facet");
                region.faces[edit[0]]=Face{edit[4],edit[5],edit[6]};
            }
        }''')
    a = '                const auto& cell=kElbowRefinementChildTetrahedra[t];'
    source = once(source, a, r'''                auto cell=kElbowRefinementChildTetrahedra[t];
                const auto scalp=std::lower_bound(kScalpRows.begin(),kScalpRows.end(),t);
                if(scalp!=kScalpRows.end() && *scalp==t)
                    cell=kScalpChildren[static_cast<std::size_t>(scalp-kScalpRows.begin())];''')
    a = '            require(std::equal(wholeBodySkinBoundaryFaces.begin(),wholeBodySkinBoundaryFaces.end(),'
    source = once(source, a, '            if(!posteriorBedReference)\n'+a)
    a = 'identityText+=kPosteriorMassSHA256;}'
    source = once(source,a,'identityText+=kPosteriorMassSHA256;identityText+=kScalpReferenceSHA256;}')
    a = '        std::vector<std::array<std::uint32_t,4>> cavityChildren;'
    source = once(source,a,'        std::vector<std::array<std::uint32_t,4>> cavityChildren,cavityDerivedParents;')
    a = '                std::map<std::array<std::uint32_t,3>,std::uint32_t> registeredFaces;'
    source = once(source,a,r'''                cavityDerivedParents.assign(kElbowRefinementParentTetrahedra.begin(),
                                            kElbowRefinementParentTetrahedra.end());
                for(std::size_t edit=0;edit<kScalpRows.size();++edit) {
                    const auto child=kScalpRows[edit];
                    auto original=cavityChildren[child];
                    std::sort(original.begin(),original.end());
                    require(original==kScalpOriginal[edit],
                            "derived scalp cavity does not match its physical child");
                    const auto parent=cavityParents[child];
                    const auto face=kScalpChildFaces[edit];
                    const auto begin=cavityOffsets[child],end=cavityOffsets[child+1u];
                    require(end==begin+1u && cavityAncestry[begin]==parent &&
                            cavityFractions[begin]==1.0 && face<region.faces.size(),
                            "derived scalp cavity spans a refined or invalid material parent");
                    cavityDerivedParents[parent]=kScalpParents[edit];
                    cavityChildren[child]=kScalpChildren[edit];
                    cavityFaces[child]=face;
                    cavityStableFaces[child]=region.sourceFaceIndices[face];
                    cavityParentStable[parent]=cavityStableFaces[child];
                }
                explicitTopology->parentTetrahedra=cavityDerivedParents;
''' + a)
    return source

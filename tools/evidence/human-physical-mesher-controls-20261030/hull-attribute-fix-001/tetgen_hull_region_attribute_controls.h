#ifndef TETGEN_HULL_REGION_ATTRIBUTE_CONTROLS_H
#define TETGEN_HULL_REGION_ATTRIBUTE_CONTROLS_H

#include <cstddef>
#include <vector>

namespace tetgen_hull_region_attribute_controls {

enum RestoreResult {
  kNoHullToRealTransition = 0,
  kInheritedFromRealNeighbor = 1,
  kMissingOrMalformedRealOwner = 2
};

template <typename Value>
inline RestoreResult inherit_real_neighbor_tuple(
    bool source_was_hull, bool output_is_hull, bool source_snapshot_valid,
    std::size_t expected_attribute_count,
    const std::vector<Value>& source_attributes, const Value& source_volume,
    std::vector<Value>& output_attributes, Value& output_volume)
{
  if (!source_was_hull || output_is_hull) {
    return kNoHullToRealTransition;
  }
  if (!source_snapshot_valid ||
      source_attributes.size() != expected_attribute_count) {
    return kMissingOrMalformedRealOwner;
  }
  output_attributes = source_attributes;
  output_volume = source_volume;
  return kInheritedFromRealNeighbor;
}

template <typename Value>
inline bool unprotected_real_region_conflict(bool face_is_protected,
                                              const Value& current_region,
                                              const Value& neighbor_region)
{
  return !face_is_protected && current_region != neighbor_region;
}

}  // namespace tetgen_hull_region_attribute_controls

#endif

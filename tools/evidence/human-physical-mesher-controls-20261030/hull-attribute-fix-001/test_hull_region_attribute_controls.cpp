#include "tetgen_hull_region_attribute_controls.h"
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace tetgen_hull_region_attribute_controls;

static void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
  }
}

int main() {
  // A hull-to-real transition inherits every attribute lane and the real
  // neighbor's volume bound; it never manufactures a default label.
  std::vector<double> real_neighbor_attributes;
  real_neighbor_attributes.push_back(3.25);
  real_neighbor_attributes.push_back(16.0);
  std::vector<double> output_attributes(2, 0.0);
  double output_volume = -1.0;
  RestoreResult result = inherit_real_neighbor_tuple(
      true, false, true, 2, real_neighbor_attributes, 0.125,
      output_attributes, output_volume);
  require(result == kInheritedFromRealNeighbor, "real owner copied");
  require(output_attributes == real_neighbor_attributes,
          "complete attribute tuple copied");
  require(output_volume == 0.125, "real volume bound copied");

  // A missing/malformed real owner fails closed and leaves the destination
  // unchanged so production can emit its invalid marker.
  std::vector<double> untouched(1, 7.0);
  double untouched_volume = 9.0;
  result = inherit_real_neighbor_tuple(
      true, false, false, 1, std::vector<double>(), 0.0,
      untouched, untouched_volume);
  require(result == kMissingOrMalformedRealOwner,
          "missing real owner is reported invalid");
  require(untouched.size() == 1 && untouched[0] == 7.0 &&
              untouched_volume == 9.0,
          "invalid owner does not guess an attribute or volume");

  // No-op cases must not overwrite a still-hull cell or a cell whose source
  // was already real.
  output_attributes.assign(1, 4.0);
  output_volume = 6.0;
  result = inherit_real_neighbor_tuple(
      true, true, true, 2, real_neighbor_attributes, 0.125,
      output_attributes, output_volume);
  require(result == kNoHullToRealTransition && output_attributes[0] == 4.0 &&
              output_volume == 6.0,
          "still-hull output is untouched");
  result = inherit_real_neighbor_tuple(
      false, false, true, 2, real_neighbor_attributes, 0.125,
      output_attributes, output_volume);
  require(result == kNoHullToRealTransition && output_attributes[0] == 4.0 &&
              output_volume == 6.0,
          "real-to-real transition is untouched");

  // Real region disagreement across an unprotected face is a hard conflict;
  // the same differing labels across a protected interface are valid and stay
  // on their respective cells.
  require(unprotected_real_region_conflict(false, 15.0, 16.0),
          "mixed unprotected real interface fails closed");
  double protected_left = 15.0;
  double protected_right = 16.0;
  bool protected_conflict =
      unprotected_real_region_conflict(true, protected_left, protected_right);
  require(!protected_conflict && protected_left == 15.0 &&
              protected_right == 16.0,
          "protected material interface preserves both attributes");
  require(!unprotected_real_region_conflict(false, 16.0, 16.0),
          "matching unprotected region attributes remain valid");

  std::cout << "PASS hull-to-real complete tuple/volume inheritance\n"
            << "PASS missing owner fails closed without guessing\n"
            << "PASS non-transition slots remain unchanged\n"
            << "PASS mixed unprotected interface fails closed\n"
            << "PASS protected interface preserves distinct labels\n"
            << "PASS matching unprotected region attributes\n";
  return 0;
}

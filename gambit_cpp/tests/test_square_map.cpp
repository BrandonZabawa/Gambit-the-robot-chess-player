// Test outlines for square_map (same numbers as square_map.ipynb and docs/gazebo.md).
#include <gtest/gtest.h>

#include "gambit/square_map/square_map.hpp"

using namespace gambit::square_map;

TEST(SquareMap, MatchesGazeboDocExample) {
  // docs/gazebo.md teleports the e2 pawn to e4 at (0.2025, -0.0125, 0.006).
  // Assert square_to_world("e4") is that point (1e-9) and e2 is (0.1525, -0.0125).
  GTEST_SKIP() << "TODO";
}

TEST(SquareMap, Corners) {
  // Assert a1 -> (0.1275, 0.0875) and h8 -> (0.3025, -0.0875).
  GTEST_SKIP() << "TODO";
}

TEST(SquareMap, InverseRoundTrip) {
  // For all 64 squares: world_to_square(x + 0.004, y - 0.004) == the square.
  // Also: world_to_square(0, 0) is nullopt; square_to_world("z9") is nullopt.
  GTEST_SKIP() << "TODO";
}

TEST(SquareMap, RotatedBoard) {
  // BoardParams with white_near_arm = false: h8 lands at (0.1275, 0.0875) and
  // world_to_square maps it back to "h8".
  GTEST_SKIP() << "TODO";
}

TEST(SquareMap, GraveyardMatchesWorld) {
  // chess_world.sdf spare queens: slot 0 x = 0.17, slot 3 x = 0.2375, y = -0.175, z = 0.
  GTEST_SKIP() << "TODO";
}

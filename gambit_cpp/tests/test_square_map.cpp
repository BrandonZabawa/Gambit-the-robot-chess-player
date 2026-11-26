#include <gtest/gtest.h>

#include "gambit/square_map/square_map.hpp"

using namespace gambit::square_map;

TEST(SquareMap, MatchesGazeboDocExample) {
  // docs/gazebo.md teleports the e2 pawn to e4 at (0.2025, -0.0125, 0.006).
  const auto e4 = square_to_world("e4");
  ASSERT_TRUE(e4);
  EXPECT_NEAR(e4->x, 0.2025, 1e-9);
  EXPECT_NEAR(e4->y, -0.0125, 1e-9);
  EXPECT_NEAR(e4->z, 0.006, 1e-9);
  const auto e2 = square_to_world("e2");
  EXPECT_NEAR(e2->x, 0.1525, 1e-9);
  EXPECT_NEAR(e2->y, -0.0125, 1e-9);
}

TEST(SquareMap, Corners) {
  const auto a1 = square_to_world("a1");
  EXPECT_NEAR(a1->x, 0.1275, 1e-9);
  EXPECT_NEAR(a1->y, 0.0875, 1e-9);
  const auto h8 = square_to_world("h8");
  EXPECT_NEAR(h8->x, 0.3025, 1e-9);
  EXPECT_NEAR(h8->y, -0.0875, 1e-9);
}

TEST(SquareMap, InverseRoundTrip) {
  for (int f = 0; f < 8; ++f)
    for (int r = 0; r < 8; ++r) {
      const auto name = square_name(f, r);
      const auto p = square_to_world(name);
      EXPECT_EQ(world_to_square(p->x + 0.004, p->y - 0.004).value(), name);
    }
  EXPECT_FALSE(world_to_square(0.0, 0.0).has_value());
  EXPECT_FALSE(square_to_world("z9").has_value());
}

TEST(SquareMap, RotatedBoard) {
  BoardParams b;
  b.white_near_arm = false;
  const auto h8 = square_to_world("h8", b);
  EXPECT_NEAR(h8->x, 0.1275, 1e-9);
  EXPECT_NEAR(h8->y, 0.0875, 1e-9);
  EXPECT_EQ(world_to_square(h8->x, h8->y, b).value(), "h8");
}

TEST(SquareMap, GraveyardMatchesWorld) {
  // chess_world.sdf: spare queens at x = 0.17, 0.1925, 0.215, 0.2375; y = -0.175; z = 0.
  EXPECT_NEAR(graveyard_slot(0).x, 0.17, 1e-9);
  EXPECT_NEAR(graveyard_slot(3).x, 0.2375, 1e-9);
  EXPECT_NEAR(graveyard_slot(3).y, -0.175, 1e-9);
  EXPECT_DOUBLE_EQ(graveyard_slot(1).z, 0.0);
}

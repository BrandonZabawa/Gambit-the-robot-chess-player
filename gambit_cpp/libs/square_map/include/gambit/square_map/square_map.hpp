// Chess square <-> world (x, y, z), from gambit_bringup/config/board.yaml.
// Formula (docs/gazebo.md): x = cx + (rank - 3.5) * s,  y = cy + (3.5 - file) * s,
// file 0 = a, rank 0 = rank 1, z = board thickness (board top). Mirrors square_map.ipynb.
#pragma once

#include <array>
#include <optional>
#include <string>
#include <string_view>

namespace gambit::square_map {

struct BoardParams {
  // Defaults copied from gambit_bringup/config/board.yaml. TODO: load the YAML at runtime
  // (pass the values in from the daemon's config) so the real-board measurements apply.
  double square = 0.025;
  double center_x = 0.215;
  double center_y = 0.0;
  double thickness = 0.006;
  bool white_near_arm = true;
  double graveyard_x = 0.17;
  double graveyard_y = -0.175;
};

struct Point3 {
  double x, y, z;
};

// "e4" -> (file 4, rank 3). nullopt on bad input.
std::optional<std::array<int, 2>> parse_square(std::string_view name);
std::string square_name(int file, int rank);

std::optional<Point3> square_to_world(std::string_view name, const BoardParams& b = {});

// Nearest square to (x, y); nullopt if outside the playing area.
std::optional<std::string> world_to_square(double x, double y, const BoardParams& b = {});

// Graveyard slot i: a row along +x from the graveyard origin,
// 0.9 square apart (matches gambit_sim world_generator GRAVEYARD_SPACING), on the table top
// (z = 0). Matches square_map.ipynb graveyard_slot().
Point3 graveyard_slot(int index, const BoardParams& b = {});

}  // namespace gambit::square_map

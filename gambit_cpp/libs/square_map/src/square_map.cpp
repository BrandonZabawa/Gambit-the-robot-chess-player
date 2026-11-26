#include "gambit/square_map/square_map.hpp"

#include <cmath>

namespace gambit::square_map {

inline constexpr double kGraveyardSpacingFrac = 0.9;  // of the square edge

std::optional<std::array<int, 2>> parse_square(std::string_view name) {
  if (name.size() != 2) return std::nullopt;
  const int file = name[0] - 'a';
  const int rank = name[1] - '1';
  if (file < 0 || file > 7 || rank < 0 || rank > 7) return std::nullopt;
  return std::array<int, 2>{file, rank};
}

std::string square_name(int file, int rank) {
  return {static_cast<char>('a' + file), static_cast<char>('1' + rank)};
}

std::optional<Point3> square_to_world(std::string_view name, const BoardParams& b) {
  auto fr = parse_square(name);
  if (!fr) return std::nullopt;
  int f = (*fr)[0], r = (*fr)[1];
  if (!b.white_near_arm) {  // board rotated 180 degrees
    f = 7 - f;
    r = 7 - r;
  }
  return Point3{b.center_x + (r - 3.5) * b.square, b.center_y + (3.5 - f) * b.square,
                b.thickness};
}

std::optional<std::string> world_to_square(double x, double y, const BoardParams& b) {
  const double r_f = (x - b.center_x) / b.square + 3.5;
  const double f_f = 3.5 - (y - b.center_y) / b.square;
  // Inside the playing area means within half a square of the outer square centers.
  if (r_f < -0.5 || r_f >= 7.5 || f_f < -0.5 || f_f >= 7.5) return std::nullopt;
  int r = static_cast<int>(std::lround(r_f));
  int f = static_cast<int>(std::lround(f_f));
  r = r < 0 ? 0 : (r > 7 ? 7 : r);
  f = f < 0 ? 0 : (f > 7 ? 7 : f);
  if (!b.white_near_arm) {
    f = 7 - f;
    r = 7 - r;
  }
  return square_name(f, r);
}

Point3 graveyard_slot(int index, const BoardParams& b) {
  return {b.graveyard_x + index * kGraveyardSpacingFrac * b.square, b.graveyard_y, 0.0};
}

}  // namespace gambit::square_map

// Square <-> world conversions. Design notes per function; implementations are TODO.
// Must agree with square_map.ipynb (tests/test_square_map.cpp uses the same numbers).
#include "gambit/square_map/square_map.hpp"

#include <cmath>
#include <stdexcept>

namespace gambit::square_map {

inline constexpr double kGraveyardSpacingFrac = 0.9;  // of the square edge

std::optional<std::array<int, 2>> parse_square(std::string_view /*name*/) {
  // Design: exactly 2 chars; file = name[0] - 'a', rank = name[1] - '1'; both must be in
  // 0..7, else std::nullopt (no exceptions: bad input from IPC is expected, not a bug).
  throw std::logic_error("TODO: parse_square");
}

std::string square_name(int /*file*/, int /*rank*/) {
  // Design: two chars, 'a' + file and '1' + rank.
  throw std::logic_error("TODO: square_name");
}

std::optional<Point3> square_to_world(std::string_view /*name*/, const BoardParams& /*b*/) {
  // Design: parse_square (nullopt passes through); flip both indices (7 - i) if
  // !white_near_arm; apply the header formula; z = board thickness.
  throw std::logic_error("TODO: square_to_world");
}

std::optional<std::string> world_to_square(double /*x*/, double /*y*/,
                                           const BoardParams& /*b*/) {
  // Design: invert the formula to continuous (rank, file); reject outside [-0.5, 7.5)
  // BEFORE rounding; std::lround, clamp to 0..7, undo the white_near_arm flip, square_name.
  throw std::logic_error("TODO: world_to_square");
}

Point3 graveyard_slot(int /*index*/, const BoardParams& /*b*/) {
  // Design: (graveyard_x + index * kGraveyardSpacingFrac * square, graveyard_y, 0.0).
  throw std::logic_error("TODO: graveyard_slot");
}

}  // namespace gambit::square_map

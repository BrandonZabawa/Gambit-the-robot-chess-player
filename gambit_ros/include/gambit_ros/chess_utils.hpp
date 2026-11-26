// Small header-only chess helpers for the nodes (no chess library dependency).
// Square indices follow python-chess: a1 = 0, b1 = 1, ..., h8 = 63.
#pragma once

#include <array>
#include <optional>
#include <stdexcept>
#include <string>

namespace gambit_ros
{

// "e4" -> 28; nullopt if malformed.
inline std::optional<int> square_index(const std::string & /*name*/)
{
  // Design: exactly 2 chars; file = letter - 'a', rank = digit - '1', both in 0..7;
  // index = rank * 8 + file (python-chess order). Anything else -> nullopt.
  throw std::logic_error("TODO: square_index");
}

// 28 -> "e4" (inverse of square_index; index assumed in 0..63).
inline std::string square_name(int /*index*/)
{
  // Design: file letter from index % 8, rank digit from index / 8.
  throw std::logic_error("TODO: square_name");
}

// UCI move "e2e4" / "e7e8q" -> (from, to, promotion char or 0).
struct UciMove
{
  int from;
  int to;
  char promotion;
};

inline std::optional<UciMove> parse_uci(const std::string & /*uci*/)
{
  // Design: length 4 or 5; first two chars and next two chars through square_index; optional
  // 5th char must be one of q r b n (lower case, as UCI writes it). Any failure -> nullopt.
  throw std::logic_error("TODO: parse_uci");
}

// Piece placement FEN ("rnbqkbnr/pppppppp/8/...") -> 64 chars, '.' for empty, index = square.
inline std::optional<std::array<char, 64>> placement_from_fen(const std::string & /*fen*/)
{
  // Design:
  // - Only the placement field (stop at the first space). FEN starts at rank 8, file a;
  //   '/' moves down one rank and requires exactly 8 files on the finished rank.
  // - Digits 1-8 skip empty squares; piece letters PNBRQKpnbrqk fill board[rank * 8 + file].
  // - Reject anything else, a rank overflowing 8 files, or not ending on rank 1 with 8 files.
  // Test ideas: start position round trip, "8/8/8/8/8/8/8/8", 9 files in a rank, 7 ranks.
  throw std::logic_error("TODO: placement_from_fen");
}

}  // namespace gambit_ros

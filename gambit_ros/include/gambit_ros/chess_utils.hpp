// Small header-only chess helpers for the nodes (no chess library dependency).
// Square indices follow python-chess: a1 = 0, b1 = 1, ..., h8 = 63.
#pragma once

#include <array>
#include <optional>
#include <string>

namespace gambit_ros
{

// "e4" -> 28; nullopt if malformed.
inline std::optional<int> square_index(const std::string & name)
{
  if (name.size() != 2) {return std::nullopt;}
  const int file = name[0] - 'a';
  const int rank = name[1] - '1';
  if (file < 0 || file > 7 || rank < 0 || rank > 7) {return std::nullopt;}
  return rank * 8 + file;
}

inline std::string square_name(int index)
{
  return {static_cast<char>('a' + index % 8), static_cast<char>('1' + index / 8)};
}

// UCI move "e2e4" / "e7e8q" -> (from, to, promotion char or 0).
struct UciMove
{
  int from;
  int to;
  char promotion;
};

inline std::optional<UciMove> parse_uci(const std::string & uci)
{
  if (uci.size() != 4 && uci.size() != 5) {return std::nullopt;}
  auto from = square_index(uci.substr(0, 2));
  auto to = square_index(uci.substr(2, 2));
  if (!from || !to) {return std::nullopt;}
  char promo = uci.size() == 5 ? uci[4] : 0;
  if (promo && std::string("qrbn").find(promo) == std::string::npos) {return std::nullopt;}
  return UciMove{*from, *to, promo};
}

// Piece placement FEN ("rnbqkbnr/pppppppp/8/...") -> 64 chars, '.' for empty, index = square.
inline std::optional<std::array<char, 64>> placement_from_fen(const std::string & fen)
{
  std::array<char, 64> board{};
  board.fill('.');
  int rank = 7, file = 0;
  for (char c : fen) {
    if (c == ' ') {break;}
    if (c == '/') {
      if (file != 8) {return std::nullopt;}
      --rank; file = 0;
    } else if (c >= '1' && c <= '8') {
      file += c - '0';
    } else if (std::string("PNBRQKpnbrqk").find(c) != std::string::npos) {
      if (file > 7 || rank < 0) {return std::nullopt;}
      board[rank * 8 + file] = c; ++file;
    } else {
      return std::nullopt;
    }
    if (file > 8) {return std::nullopt;}
  }
  if (rank != 0 || file != 8) {return std::nullopt;}
  return board;
}

}  // namespace gambit_ros

#include "gambit/servo_bus/sts_protocol.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace gambit::servo {

std::uint8_t checksum(std::span<const std::uint8_t> body) {
  unsigned sum = 0;
  for (auto b : body) sum += b;
  return static_cast<std::uint8_t>(~sum & 0xFF);
}

std::vector<std::uint8_t> encode(std::uint8_t id, Instruction instr,
                                 std::span<const std::uint8_t> params) {
  if (params.size() > 250) throw std::invalid_argument("too many params for one packet");
  std::vector<std::uint8_t> pkt{kHeader, kHeader, id,
                                static_cast<std::uint8_t>(params.size() + 2),
                                static_cast<std::uint8_t>(instr)};
  pkt.insert(pkt.end(), params.begin(), params.end());
  pkt.push_back(checksum(std::span(pkt).subspan(2)));
  return pkt;
}

std::vector<std::uint8_t> ping(std::uint8_t id) { return encode(id, Instruction::Ping); }

std::vector<std::uint8_t> read(std::uint8_t id, std::uint8_t address, std::uint8_t length) {
  const std::uint8_t p[2] = {address, length};
  return encode(id, Instruction::Read, p);
}

std::vector<std::uint8_t> write(std::uint8_t id, std::uint8_t address,
                                std::span<const std::uint8_t> data) {
  std::vector<std::uint8_t> p{address};
  p.insert(p.end(), data.begin(), data.end());
  return encode(id, Instruction::Write, p);
}

std::vector<std::uint8_t> sync_write(std::uint8_t address, std::uint8_t data_length,
                                     std::span<const SyncEntry> entries) {
  std::vector<std::uint8_t> p{address, data_length};
  for (const auto& e : entries) {
    if (e.data.size() != data_length) throw std::invalid_argument("sync_write: data length");
    p.push_back(e.id);
    p.insert(p.end(), e.data.begin(), e.data.end());
  }
  return encode(kBroadcastId, Instruction::SyncWrite, p);
}

std::optional<StatusPacket> decode_status(std::span<const std::uint8_t> bytes) {
  std::size_t i = 0;
  while (i + 1 < bytes.size() && !(bytes[i] == kHeader && bytes[i + 1] == kHeader)) ++i;
  // Need header(2) + id + len + err + chk at minimum.
  if (i + 6 > bytes.size()) return std::nullopt;
  const std::uint8_t len = bytes[i + 3];
  if (len < 2 || i + 4 + len > bytes.size()) return std::nullopt;
  auto body = bytes.subspan(i + 2, len + 1);  // id, len, err, params
  const std::uint8_t chk = bytes[i + 3 + len];
  if (checksum(body) != chk) return std::nullopt;
  StatusPacket s;
  s.id = bytes[i + 2];
  s.error = bytes[i + 4];
  s.params.assign(bytes.begin() + i + 5, bytes.begin() + i + 3 + len);
  return s;
}

double ticks_to_rad(int ticks) {
  return (ticks - kCenterTick) * 2.0 * std::numbers::pi / kTicksPerRev;
}

int rad_to_ticks(double rad) {
  const int t = kCenterTick + static_cast<int>(std::lround(rad * kTicksPerRev / (2.0 * std::numbers::pi)));
  return std::clamp(t, 0, kTicksPerRev - 1);
}

}  // namespace gambit::servo

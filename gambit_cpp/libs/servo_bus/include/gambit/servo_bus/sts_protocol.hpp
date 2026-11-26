// Feetech STS3215 (SCS/STS serial protocol) packet encode/decode.
// Mirrors gambit_arm/notebooks/servo_bus.ipynb; keep the two in sync.
//
// Instruction packet: 0xFF 0xFF ID LEN INSTR P1..PN CHK      LEN = N + 2
// Status packet:      0xFF 0xFF ID LEN ERR   P1..PN CHK      LEN = N + 2
// CHK = ~(ID + LEN + INSTR/ERR + sum(P)) & 0xFF
// STS multi-byte registers are little-endian (low byte first). (The older SCS series is
// big-endian; this code targets STS only.)
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace gambit::servo {

inline constexpr std::uint8_t kHeader = 0xFF;
inline constexpr std::uint8_t kBroadcastId = 0xFE;

enum class Instruction : std::uint8_t {
  Ping = 0x01,
  Read = 0x02,
  Write = 0x03,
  RegWrite = 0x04,
  Action = 0x05,
  SyncRead = 0x82,
  SyncWrite = 0x83,
};

// Control table addresses. VERIFY AGAINST THE STS3215 DATASHEET / memory table before use.
namespace reg {
inline constexpr std::uint8_t kId = 0x05;
inline constexpr std::uint8_t kBaudRate = 0x06;
inline constexpr std::uint8_t kMinAngleLimit = 0x09;    // 2 bytes
inline constexpr std::uint8_t kMaxAngleLimit = 0x0B;    // 2 bytes
inline constexpr std::uint8_t kPositionOffset = 0x1F;   // 2 bytes
inline constexpr std::uint8_t kTorqueEnable = 0x28;
inline constexpr std::uint8_t kAcceleration = 0x29;
inline constexpr std::uint8_t kGoalPosition = 0x2A;     // 2 bytes
inline constexpr std::uint8_t kGoalTime = 0x2C;         // 2 bytes
inline constexpr std::uint8_t kGoalSpeed = 0x2E;        // 2 bytes
inline constexpr std::uint8_t kTorqueLimit = 0x30;      // 2 bytes, 0..1000 (0.1 %)
inline constexpr std::uint8_t kLock = 0x37;
inline constexpr std::uint8_t kPresentPosition = 0x38;  // 2 bytes
inline constexpr std::uint8_t kPresentSpeed = 0x3A;     // 2 bytes
inline constexpr std::uint8_t kPresentLoad = 0x3C;      // 2 bytes
inline constexpr std::uint8_t kPresentVoltage = 0x3E;
inline constexpr std::uint8_t kPresentTemperature = 0x3F;
inline constexpr std::uint8_t kMoving = 0x42;
inline constexpr std::uint8_t kPresentCurrent = 0x45;   // 2 bytes
}  // namespace reg

inline constexpr int kTicksPerRev = 4096;
inline constexpr int kCenterTick = 2048;  // tick that maps to 0 rad (after calibration offset)

std::uint8_t checksum(std::span<const std::uint8_t> body);  // body = ID..last param

std::vector<std::uint8_t> encode(std::uint8_t id, Instruction instr,
                                 std::span<const std::uint8_t> params = {});

std::vector<std::uint8_t> ping(std::uint8_t id);
std::vector<std::uint8_t> read(std::uint8_t id, std::uint8_t address, std::uint8_t length);
std::vector<std::uint8_t> write(std::uint8_t id, std::uint8_t address,
                                std::span<const std::uint8_t> data);
// One SYNC_WRITE to many servos: each entry is (id, data) with data.size() == data_length.
struct SyncEntry {
  std::uint8_t id;
  std::vector<std::uint8_t> data;
};
std::vector<std::uint8_t> sync_write(std::uint8_t address, std::uint8_t data_length,
                                     std::span<const SyncEntry> entries);

struct StatusPacket {
  std::uint8_t id = 0;
  std::uint8_t error = 0;
  std::vector<std::uint8_t> params;
};

// Parse one status packet from the start of `bytes` (leading garbage before 0xFF 0xFF is
// skipped). Returns nullopt on short input or bad checksum.
std::optional<StatusPacket> decode_status(std::span<const std::uint8_t> bytes);

// Little-endian helpers.
inline std::vector<std::uint8_t> u16_le(std::uint16_t v) {
  return {static_cast<std::uint8_t>(v & 0xFF), static_cast<std::uint8_t>(v >> 8)};
}
inline std::uint16_t from_u16_le(std::uint8_t lo, std::uint8_t hi) {
  return static_cast<std::uint16_t>(lo | (hi << 8));
}

double ticks_to_rad(int ticks);
int rad_to_ticks(double rad);  // clamped to 0..4095

}  // namespace gambit::servo

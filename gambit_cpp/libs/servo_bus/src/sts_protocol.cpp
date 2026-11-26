// STS3215 packet encode/decode. Design notes per function; implementations are TODO.
// Must produce byte-for-byte the same packets as gambit_arm/notebooks/servo_bus.ipynb and
// firmware/stm32/Core/Src/servo_bus.c (tests/test_sts_protocol.cpp holds the shared vectors).
#include "gambit/servo_bus/sts_protocol.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace gambit::servo {

std::uint8_t checksum(std::span<const std::uint8_t> /*body*/) {
  // Design:
  // - Sum every byte of the body (ID .. last param) in an unsigned accumulator wider than
  //   8 bits, then take the bitwise NOT and keep the low 8 bits.
  // - Wider accumulator + final mask is the whole trick: the sum overflows 8 bits for any
  //   real packet (see ChecksumWrapsAt8Bits).
  throw std::logic_error("TODO: checksum");
}

std::vector<std::uint8_t> encode(std::uint8_t /*id*/, Instruction /*instr*/,
                                 std::span<const std::uint8_t> /*params*/) {
  // Design:
  // - Layout: 0xFF 0xFF ID LEN INSTR params... CHK with LEN = params.size() + 2.
  // - LEN is one byte: reject oversized params with std::invalid_argument.
  // - Build into one std::vector (reserve the final size), then append checksum() of the
  //   span starting after the two header bytes (std::span::subspan).
  throw std::logic_error("TODO: encode");
}

std::vector<std::uint8_t> ping(std::uint8_t /*id*/) {
  // Design: encode with Instruction::Ping and no params.
  throw std::logic_error("TODO: ping");
}

std::vector<std::uint8_t> read(std::uint8_t /*id*/, std::uint8_t /*address*/,
                               std::uint8_t /*length*/) {
  // Design: encode with Instruction::Read, params = {address, length}.
  throw std::logic_error("TODO: read");
}

std::vector<std::uint8_t> write(std::uint8_t /*id*/, std::uint8_t /*address*/,
                                std::span<const std::uint8_t> /*data*/) {
  // Design: encode with Instruction::Write, params = address followed by the data bytes.
  throw std::logic_error("TODO: write");
}

std::vector<std::uint8_t> sync_write(std::uint8_t /*address*/, std::uint8_t /*data_length*/,
                                     std::span<const SyncEntry> /*entries*/) {
  // Design:
  // - Params = {address, data_length}, then per entry: id followed by its data bytes.
  // - Every entry must have data.size() == data_length, else std::invalid_argument (a short
  //   entry would shift every following servo's bytes - silent, dangerous).
  // - Sent to kBroadcastId with Instruction::SyncWrite; servos do not reply.
  throw std::logic_error("TODO: sync_write");
}

std::optional<StatusPacket> decode_status(std::span<const std::uint8_t> /*bytes*/) {
  // Design:
  // - Scan for the first 0xFF 0xFF pair; bytes before it are noise / half-duplex echo.
  // - Need at least 6 bytes from the header (FF FF ID LEN ERR CHK) to read LEN; then the
  //   packet is 4 + LEN bytes. Too short -> std::nullopt (caller reads more and retries).
  // - Verify checksum() over ID..last param against the final byte; mismatch -> nullopt.
  // - Fill StatusPacket: id, error byte, params = the LEN - 2 bytes between ERR and CHK.
  // Index arithmetic with span::subspan keeps it bounds-safe; test the off-by-ones against
  // the notebook's decode_status.
  throw std::logic_error("TODO: decode_status");
}

double ticks_to_rad(int /*ticks*/) {
  // Design: (ticks - kCenterTick) * 2*pi / kTicksPerRev, using std::numbers::pi.
  throw std::logic_error("TODO: ticks_to_rad");
}

int rad_to_ticks(double /*rad*/) {
  // Design: inverse of ticks_to_rad rounded with std::lround, then std::clamp to
  // 0..kTicksPerRev-1 (the register cannot hold anything else).
  throw std::logic_error("TODO: rad_to_ticks");
}

}  // namespace gambit::servo

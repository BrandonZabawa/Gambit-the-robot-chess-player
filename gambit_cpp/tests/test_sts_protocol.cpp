// Test outlines for the STS protocol. Same vectors as servo_bus.ipynb's self-check plan and
// the STM32 firmware. Replace each GTEST_SKIP with the assertions described.
#include <gtest/gtest.h>

#include "gambit/servo_bus/sts_protocol.hpp"

using namespace gambit::servo;

TEST(StsProtocol, ChecksumOfPing) {
  // Assert: ping(1) == {FF FF 01 02 01 FB}  (CHK = ~(1 + 2 + 1) = 0xFB).
  GTEST_SKIP() << "TODO";
}

TEST(StsProtocol, ChecksumWrapsAt8Bits) {
  // Assert: checksum({0xFE, 0xFF, 0x10}) == ~(0xFE + 0xFF + 0x10) & 0xFF (sum > 255).
  GTEST_SKIP() << "TODO";
}

TEST(StsProtocol, ReadPresentPosition) {
  // Assert on read(3, reg::kPresentPosition, 2): size 8, LEN byte == 4, instruction byte ==
  // Read, address byte == kPresentPosition, last byte == ~(3 + 4 + 2 + 0x38 + 2) & 0xFF.
  GTEST_SKIP() << "TODO";
}

TEST(StsProtocol, WriteGoalPositionIsLittleEndian) {
  // Assert on write(1, reg::kGoalPosition, u16_le(2048)): size 9, bytes 6 and 7 are 00 08
  // (low byte first).
  GTEST_SKIP() << "TODO";
}

TEST(StsProtocol, StatusRoundTrip) {
  // Build a status reply {FF FF 01 04 00 00 08 CHK} (CHK via checksum over bytes 2..),
  // prefix two garbage bytes; assert decode_status finds it: id 1, error 0, 2 params that
  // decode (from_u16_le) to 2048.
  GTEST_SKIP() << "TODO";
}

TEST(StsProtocol, BadChecksumRejected) {
  // Assert: decode_status returns nullopt for {FF FF 01 02 00 00} (wrong CHK) and for a
  // truncated {FF FF 01}.
  GTEST_SKIP() << "TODO";
}

TEST(StsProtocol, SyncWriteLayout) {
  // sync_write(kGoalPosition, 2, {{1, u16_le(1000)}, {2, u16_le(3000)}}): assert size 14,
  // id byte == kBroadcastId, LEN == 2 + 6 + 2; an entry with the wrong data length throws
  // std::invalid_argument.
  GTEST_SKIP() << "TODO";
}

TEST(StsProtocol, TicksRadiansRoundTrip) {
  // Assert: ticks_to_rad(2048) == 0, ticks_to_rad(3072) == pi/2; rad_to_ticks(ticks_to_rad(t))
  // == t for t in {0, 100, 2048, 4000, 4095}; rad_to_ticks(100.0) == 4095 (clamped).
  GTEST_SKIP() << "TODO";
}

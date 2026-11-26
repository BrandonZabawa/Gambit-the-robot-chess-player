#include <gtest/gtest.h>

#include "gambit/servo_bus/sts_protocol.hpp"

using namespace gambit::servo;

TEST(StsProtocol, ChecksumOfPing) {
  // PING id 1: FF FF 01 02 01 CHK, CHK = ~(1 + 2 + 1) = 0xFB
  const std::vector<std::uint8_t> expected{0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB};
  EXPECT_EQ(ping(1), expected);
}

TEST(StsProtocol, ChecksumWrapsAt8Bits) {
  const std::uint8_t body[] = {0xFE, 0xFF, 0x10};
  EXPECT_EQ(checksum(body), static_cast<std::uint8_t>(~(0xFE + 0xFF + 0x10) & 0xFF));
}

TEST(StsProtocol, ReadPresentPosition) {
  const auto pkt = read(3, reg::kPresentPosition, 2);
  ASSERT_EQ(pkt.size(), 8u);
  EXPECT_EQ(pkt[3], 4);  // LEN = 2 params + 2
  EXPECT_EQ(pkt[4], static_cast<std::uint8_t>(Instruction::Read));
  EXPECT_EQ(pkt[5], reg::kPresentPosition);
  EXPECT_EQ(pkt.back(), static_cast<std::uint8_t>(~(3 + 4 + 2 + 0x38 + 2) & 0xFF));
}

TEST(StsProtocol, WriteGoalPositionIsLittleEndian) {
  const auto pkt = write(1, reg::kGoalPosition, u16_le(2048));
  // FF FF 01 05 03 2A 00 08 CHK
  ASSERT_EQ(pkt.size(), 9u);
  EXPECT_EQ(pkt[6], 0x00);
  EXPECT_EQ(pkt[7], 0x08);
}

TEST(StsProtocol, StatusRoundTrip) {
  // Build a fake status reply: id 1, no error, present position 0x0800 (LE: 00 08).
  std::vector<std::uint8_t> reply{0xFF, 0xFF, 0x01, 0x04, 0x00, 0x00, 0x08};
  reply.push_back(checksum(std::span(reply).subspan(2)));
  std::vector<std::uint8_t> noisy{0x12, 0x34};  // garbage before the header is skipped
  noisy.insert(noisy.end(), reply.begin(), reply.end());
  const auto s = decode_status(noisy);
  ASSERT_TRUE(s.has_value());
  EXPECT_EQ(s->id, 1);
  EXPECT_EQ(s->error, 0);
  ASSERT_EQ(s->params.size(), 2u);
  EXPECT_EQ(from_u16_le(s->params[0], s->params[1]), 2048);
}

TEST(StsProtocol, BadChecksumRejected) {
  std::vector<std::uint8_t> reply{0xFF, 0xFF, 0x01, 0x02, 0x00, 0x00};
  EXPECT_FALSE(decode_status(reply).has_value());
  std::vector<std::uint8_t> short_reply{0xFF, 0xFF, 0x01};
  EXPECT_FALSE(decode_status(short_reply).has_value());
}

TEST(StsProtocol, SyncWriteLayout) {
  std::vector<SyncEntry> e{{1, u16_le(1000)}, {2, u16_le(3000)}};
  const auto pkt = sync_write(reg::kGoalPosition, 2, e);
  // header(2) id len instr addr dlen + 2*(1+2) + chk
  ASSERT_EQ(pkt.size(), 2u + 3u + 2u + 6u + 1u);
  EXPECT_EQ(pkt[2], kBroadcastId);
  EXPECT_EQ(pkt[3], 2 + 6 + 2);
}

TEST(StsProtocol, TicksRadiansRoundTrip) {
  EXPECT_DOUBLE_EQ(ticks_to_rad(2048), 0.0);
  EXPECT_NEAR(ticks_to_rad(3072), 3.14159265 / 2, 1e-6);
  for (int t : {0, 100, 2048, 4000, 4095}) EXPECT_EQ(rad_to_ticks(ticks_to_rad(t)), t);
  EXPECT_EQ(rad_to_ticks(100.0), 4095);  // clamped
}

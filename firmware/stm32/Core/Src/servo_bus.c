#include "servo_bus.h"

#include <string.h>

#ifdef USE_HAL_DRIVER
#include "main.h"           /* CubeMX: huart for the servo bus, e.g. extern UART_HandleTypeDef huart1; */
#endif

/* ------------------------------------------------------------------ pure helpers */

uint8_t sb_checksum(const uint8_t *body, size_t len) {
  unsigned sum = 0;
  for (size_t i = 0; i < len; ++i) sum += body[i];
  return (uint8_t)(~sum & 0xFFu);
}

size_t sb_encode(uint8_t id, uint8_t instr, const uint8_t *params, size_t nparams,
                 uint8_t *out, size_t out_cap) {
  const size_t total = nparams + 6u;
  if (total > out_cap || nparams > 250u) return 0;
  out[0] = 0xFF;
  out[1] = 0xFF;
  out[2] = id;
  out[3] = (uint8_t)(nparams + 2u);
  out[4] = instr;
  if (nparams) memcpy(&out[5], params, nparams);
  out[total - 1u] = sb_checksum(&out[2], nparams + 3u);
  return total;
}

size_t sb_encode_sync_write_u16(uint8_t addr, const uint8_t *ids, const uint16_t *values,
                                size_t n, uint8_t *out, size_t out_cap) {
  uint8_t params[2 + 3 * SB_NUM_SERVOS];
  if (n > SB_NUM_SERVOS) return 0;
  params[0] = addr;
  params[1] = 2;  /* data length per servo */
  for (size_t i = 0; i < n; ++i) {
    params[2 + 3 * i] = ids[i];
    params[3 + 3 * i] = (uint8_t)(values[i] & 0xFFu);  /* STS: little-endian */
    params[4 + 3 * i] = (uint8_t)(values[i] >> 8);
  }
  return sb_encode(SB_BROADCAST_ID, SB_INST_SYNC_WRITE, params, 2 + 3 * n, out, out_cap);
}

bool sb_decode_status(const uint8_t *buf, size_t len, uint8_t *id, uint8_t *error,
                      uint8_t *params, size_t *nparams) {
  size_t i = 0;
  while (i + 1 < len && !(buf[i] == 0xFF && buf[i + 1] == 0xFF)) ++i;
  if (i + 6 > len) return false;
  const uint8_t plen = buf[i + 3];
  if (plen < 2 || i + 4u + plen > len) return false;
  if (sb_checksum(&buf[i + 2], plen + 1u) != buf[i + 3 + plen]) return false;
  *id = buf[i + 2];
  *error = buf[i + 4];
  *nparams = plen - 2u;
  if (*nparams) memcpy(params, &buf[i + 5], *nparams);
  return true;
}

/* ------------------------------------------------------------------ hardware (TODO) */

void sb_init(void) {
  /* TODO: CubeMX: USARTx in Single Wire (Half-Duplex) mode, 1 000 000 baud 8N1, TX pin open
   * drain with pull-up (or the Waveshare-style direction buffer); DMA for TX and RX; enable
   * the IDLE-line interrupt to detect end of a status packet. */
}

bool sb_ping(uint8_t id) {
  /* TODO: encode PING, HAL_HalfDuplex_EnableTransmitter + HAL_UART_Transmit, then
   * HAL_HalfDuplex_EnableReceiver and wait (<= 2 ms) for a status packet from `id`. */
  (void)id;
  return false;
}

bool sb_write_goal_positions(const uint16_t goal_ticks[SB_NUM_SERVOS]) {
  static const uint8_t ids[SB_NUM_SERVOS] = {1, 2, 3, 4, 5, 6};
  uint8_t pkt[SB_MAX_PACKET];
  const size_t n = sb_encode_sync_write_u16(SB_REG_GOAL_POSITION, ids, goal_ticks,
                                            SB_NUM_SERVOS, pkt, sizeof pkt);
  if (n == 0) return false;
  /* TODO: transmit pkt (DMA), no reply expected for broadcast. Callers must have run
   * limits_check_command() first. */
  return false;
}

bool sb_set_torque(bool enable) {
  /* TODO: broadcast WRITE SB_REG_TORQUE_ENABLE = enable. */
  (void)enable;
  return false;
}

bool sb_read_state(sb_state_t *state) {
  /* TODO: SYNC_READ (or six READs) of present position / current / temperature; set
   * missing_mask for servos that time out. Budget: whole read < 2 ms at 1 Mbaud. */
  (void)state;
  return false;
}

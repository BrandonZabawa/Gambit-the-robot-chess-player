#include "servo_bus.h"

#include <string.h>

#ifdef USE_HAL_DRIVER
#include "main.h"           /* CubeMX: huart for the servo bus, e.g. extern UART_HandleTypeDef huart1; */
#endif

/* ------------------------------------------------------------------ pure helpers */

uint8_t sb_checksum(const uint8_t *body, size_t len) {
  /* Design: sum the body bytes (ID .. last param) in an unsigned int (wider than 8 bits),
   * bitwise NOT, keep the low byte. Same as checksum() in the notebook and gambit_cpp.
   * TODO: implement */
  (void)body; (void)len;
  return 0;
}

size_t sb_encode(uint8_t id, uint8_t instr, const uint8_t *params, size_t nparams,
                 uint8_t *out, size_t out_cap) {
  /* Design: FF FF ID LEN INSTR params CHK with LEN = nparams + 2 and total = nparams + 6.
   * No malloc on the MCU: the caller owns `out`; return 0 if it does not fit (or LEN would
   * overflow a byte) so callers can test the length. Checksum over out[2] .. last param.
   * TODO: implement */
  (void)id; (void)instr; (void)params; (void)nparams; (void)out; (void)out_cap;
  return 0;
}

size_t sb_encode_sync_write_u16(uint8_t addr, const uint8_t *ids, const uint16_t *values,
                                size_t n, uint8_t *out, size_t out_cap) {
  /* Design: params = addr, data length 2, then per servo: id, value low byte, value high
   * byte (STS is little-endian). Build them in a stack buffer sized for SB_NUM_SERVOS, reject
   * n > SB_NUM_SERVOS, then sb_encode to SB_BROADCAST_ID with SB_INST_SYNC_WRITE.
   * TODO: implement */
  (void)addr; (void)ids; (void)values; (void)n; (void)out; (void)out_cap;
  return 0;
}

bool sb_decode_status(const uint8_t *buf, size_t len, uint8_t *id, uint8_t *error,
                      uint8_t *params, size_t *nparams) {
  /* Design: skip to the first FF FF; need 6 bytes to read LEN, then 4 + LEN bytes in total;
   * verify the checksum; fill id, error and the LEN - 2 params. Return false (outputs
   * untouched) on short input or a bad checksum. Mirror decode_status in the notebook and
   * check the same vectors (gambit_cpp/tests/test_sts_protocol.cpp).
   * TODO: implement */
  (void)buf; (void)len; (void)id; (void)error; (void)params; (void)nparams;
  return false;
}

/* ------------------------------------------------------------------ hardware (TODO) */

void sb_init(void) {
  /* Design: CubeMX: USARTx in Single Wire (Half-Duplex) mode, 1 000 000 baud 8N1, TX pin open
   * drain with pull-up (or the Waveshare-style direction buffer); DMA for TX and RX; enable
   * the IDLE-line interrupt to detect the end of a status packet.
   * TODO: implement */
}

bool sb_ping(uint8_t id) {
  /* Design: sb_encode a PING, HAL_HalfDuplex_EnableTransmitter + HAL_UART_Transmit, then
   * HAL_HalfDuplex_EnableReceiver and wait (<= 2 ms) for a status packet from `id`
   * (sb_decode_status). Returns true only on a valid reply from that id.
   * TODO: implement */
  (void)id;
  return false;
}

bool sb_write_goal_positions(const uint16_t goal_ticks[SB_NUM_SERVOS]) {
  /* Design: one sb_encode_sync_write_u16 to SB_REG_GOAL_POSITION for ids 1..6 into a stack
   * buffer of SB_MAX_PACKET, transmit (DMA), no reply for broadcast. Callers must have run
   * limits_check_command() first: this function does not check limits.
   * TODO: implement */
  (void)goal_ticks;
  return false;
}

bool sb_set_torque(bool enable) {
  /* Design: broadcast WRITE of SB_REG_TORQUE_ENABLE = enable (no reply).
   * TODO: implement */
  (void)enable;
  return false;
}

bool sb_read_state(sb_state_t *state) {
  /* Design: SYNC_READ (or six READs to start) of present position / current / temperature;
   * set the missing_mask bit for servos that time out. Budget: the whole read < 2 ms at
   * 1 Mbaud, or it eats the 10 ms control cycle.
   * TODO: implement */
  (void)state;
  return false;
}

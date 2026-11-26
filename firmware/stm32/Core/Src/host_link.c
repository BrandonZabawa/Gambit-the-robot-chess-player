#include "host_link.h"

#include <string.h>

uint16_t hl_crc16(const uint8_t *data, size_t len) {
  /* Design: CRC-16/CCITT-FALSE, bitwise: crc = 0xFFFF; per byte XOR it into the high byte,
   * then 8 times shift left, XOR 0x1021 when the bit shifted out was 1. No reflection, no
   * final XOR. Check value: CRC("123456789") = 0x29B1. (A 256-entry table is faster; bitwise
   * is plenty for 70-byte frames at 100 Hz.)
   * TODO: implement */
  (void)data; (void)len;
  return 0;
}

size_t hl_encode(const hl_frame_t *f, uint8_t *out, size_t out_cap) {
  /* Design: SOF0 SOF1 TYPE SEQ LEN PAYLOAD CRC_LO CRC_HI (HL_OVERHEAD + len bytes). Return 0
   * if len > HL_MAX_PAYLOAD or the frame does not fit out_cap. CRC over TYPE..PAYLOAD,
   * stored little-endian.
   * TODO: implement */
  (void)f; (void)out; (void)out_cap;
  return 0;
}

bool hl_feed_byte(uint8_t byte, hl_frame_t *out) {
  /* Design: static state machine SOF0 -> SOF1 -> TYPE -> SEQ -> LEN -> PAYLOAD -> CRC_LO ->
   * CRC_HI. Byte-wise so it works with any transport (DMA ring buffer, CDC callback).
   * LEN > HL_MAX_PAYLOAD or a bad CRC -> back to SOF0 (resync) and count the error; on a
   * good CRC copy into *out and return true.
   * TODO: implement */
  (void)byte; (void)out;
  return false;
}

void hl_init(void) {
  /* Design: USB CDC (CubeMX USB_DEVICE middleware, CDC class) is simplest on boards with USB
   * FS; otherwise UART + DMA circular RX buffer at 921600 baud via the ST-LINK virtual COM port.
   * TODO: implement */
}

bool hl_send(const hl_frame_t *f) {
  /* Design: hl_encode into a stack buffer (HL_MAX_PAYLOAD + HL_OVERHEAD), then
   * CDC_Transmit_FS or HAL_UART_Transmit_DMA; queue (or drop and count) if the transport is
   * busy. Return false on encode failure or when not sent.
   * TODO: implement */
  (void)f;
  return false;
}

void hl_poll(void) {
  /* Design: drain received bytes; for each, if hl_feed_byte(b, &frame) dispatch to
   * app_handle_command(&frame) in main.c. Called once per control cycle.
   * TODO: implement */
}

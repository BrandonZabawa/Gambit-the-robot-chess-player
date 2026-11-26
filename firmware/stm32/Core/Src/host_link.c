#include "host_link.h"

#include <string.h>

uint16_t hl_crc16(const uint8_t *data, size_t len) {
  /* CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final xor. */
  uint16_t crc = 0xFFFFu;
  for (size_t i = 0; i < len; ++i) {
    crc ^= (uint16_t)data[i] << 8;
    for (int b = 0; b < 8; ++b)
      crc = (crc & 0x8000u) ? (uint16_t)((crc << 1) ^ 0x1021u) : (uint16_t)(crc << 1);
  }
  return crc;
}

size_t hl_encode(const hl_frame_t *f, uint8_t *out, size_t out_cap) {
  const size_t total = (size_t)f->len + HL_OVERHEAD;
  if (f->len > HL_MAX_PAYLOAD || total > out_cap) return 0;
  out[0] = HL_SOF0;
  out[1] = HL_SOF1;
  out[2] = f->type;
  out[3] = f->seq;
  out[4] = f->len;
  if (f->len) memcpy(&out[5], f->payload, f->len);
  const uint16_t crc = hl_crc16(&out[2], (size_t)f->len + 3u);
  out[5 + f->len] = (uint8_t)(crc & 0xFFu);
  out[6 + f->len] = (uint8_t)(crc >> 8);
  return total;
}

bool hl_feed_byte(uint8_t byte, hl_frame_t *out) {
  /* TODO: receive state machine described in host_link.h. */
  (void)byte; (void)out;
  return false;
}

void hl_init(void) {
  /* TODO: USB CDC (CubeMX USB_DEVICE middleware, CDC class) is simplest on boards with USB FS;
   * otherwise UART + DMA circular RX buffer at 921600 baud via the ST-LINK virtual COM port. */
}

bool hl_send(const hl_frame_t *f) {
  uint8_t buf[HL_MAX_PAYLOAD + HL_OVERHEAD];
  const size_t n = hl_encode(f, buf, sizeof buf);
  if (n == 0) return false;
  /* TODO: CDC_Transmit_FS(buf, n) or HAL_UART_Transmit_DMA; queue if busy. */
  return false;
}

void hl_poll(void) {
  /* TODO: for each received byte: if (hl_feed_byte(b, &frame)) dispatch to
   * app_handle_command(&frame) in main.c. */
}

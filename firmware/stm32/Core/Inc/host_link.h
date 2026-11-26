/* Host <-> MCU framed protocol over UART or USB CDC. Full description: firmware/stm32/protocol.md.
 * Frame: SOF0 SOF1 TYPE SEQ LEN PAYLOAD[LEN] CRC16_LO CRC16_HI
 *   SOF = 0xAA 0x55, LEN <= HL_MAX_PAYLOAD, CRC16-CCITT (poly 0x1021, init 0xFFFF) over
 *   TYPE..PAYLOAD. Multi-byte fields little-endian. */
#ifndef GAMBIT_HOST_LINK_H
#define GAMBIT_HOST_LINK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define HL_SOF0 0xAAu
#define HL_SOF1 0x55u
#define HL_MAX_PAYLOAD 64u
#define HL_OVERHEAD 7u  /* SOF0 SOF1 TYPE SEQ LEN CRC_LO CRC_HI */

typedef enum {
  /* host -> MCU */
  HL_CMD_PING = 0x01,              /* payload: none                  -> HL_RSP_ACK            */
  HL_CMD_HEARTBEAT = 0x02,         /* payload: none                  -> (no reply)            */
  HL_CMD_SET_TORQUE = 0x03,        /* payload: u8 enable             -> ACK / NACK            */
  HL_CMD_SET_JOINT_TARGETS = 0x04, /* payload: 6 x u16 ticks         -> ACK / NACK(fault)     */
  HL_CMD_GET_STATE = 0x05,         /* payload: none                  -> HL_RSP_STATE          */
  HL_CMD_SET_LIMITS = 0x06,        /* payload: limits (protocol.md)  -> ACK / NACK            */
  HL_CMD_ESTOP = 0x07,             /* payload: none                  -> ACK                   */
  HL_CMD_CLEAR_ESTOP = 0x08,       /* payload: none                  -> ACK / NACK            */
  HL_CMD_GET_VERSION = 0x09,       /* payload: none                  -> HL_RSP_VERSION        */
  /* MCU -> host */
  HL_RSP_ACK = 0x81,               /* payload: u8 acked type                                  */
  HL_RSP_NACK = 0x82,              /* payload: u8 type, u32 fault bits (limit_fault_t)        */
  HL_RSP_STATE = 0x83,             /* payload: see protocol.md                                */
  HL_RSP_FAULT = 0x84,             /* unsolicited: u32 fault bits, u8 estop latched           */
  HL_RSP_VERSION = 0x85,           /* payload: u8 major, u8 minor, u8 patch                   */
} hl_type_t;

typedef struct {
  uint8_t type;
  uint8_t seq;
  uint8_t len;
  uint8_t payload[HL_MAX_PAYLOAD];
} hl_frame_t;

/* CRC and encoding (pure functions, TODO bodies; testable on the host). */
uint16_t hl_crc16(const uint8_t *data, size_t len);
size_t hl_encode(const hl_frame_t *f, uint8_t *out, size_t out_cap);

/* Byte-wise receive state machine. Returns true when `out` holds a complete, CRC-valid frame.
 * TODO: implement (states: SOF0, SOF1, TYPE, SEQ, LEN, PAYLOAD, CRC_LO, CRC_HI; resync on
 * bad CRC; count errors for diagnostics). */
bool hl_feed_byte(uint8_t byte, hl_frame_t *out);

/* Transport (TODO): USB CDC (CDC_Transmit_FS) or UART DMA. */
void hl_init(void);
bool hl_send(const hl_frame_t *f);
void hl_poll(void);   /* drain RX buffer, dispatch complete frames to the command handler */

#endif /* GAMBIT_HOST_LINK_H */

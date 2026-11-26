/* Feetech STS3215 servo bus over half-duplex UART (STM32 USART in half-duplex mode, TX pin
 * shared with the bus data line). Same packet format and checksum as
 * gambit_arm/notebooks/servo_bus.ipynb and gambit_cpp/libs/servo_bus (keep the three in sync).
 *   instruction: FF FF ID LEN INSTR P1..PN CHK   LEN = N + 2
 *   CHK = ~(ID + LEN + INSTR + P1 + ... + PN) & 0xFF
 * Register addresses: VERIFY against the STS3215 datasheet. */
#ifndef GAMBIT_SERVO_BUS_H
#define GAMBIT_SERVO_BUS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define SB_NUM_SERVOS        6u     /* ids 1..6: pan, lift, elbow, wrist_flex, wrist_roll, gripper */
#define SB_BROADCAST_ID      0xFEu
#define SB_MAX_PACKET        64u

#define SB_INST_PING         0x01u
#define SB_INST_READ         0x02u
#define SB_INST_WRITE        0x03u
#define SB_INST_SYNC_READ    0x82u
#define SB_INST_SYNC_WRITE   0x83u

#define SB_REG_TORQUE_ENABLE    0x28u  /* VERIFY */
#define SB_REG_GOAL_POSITION    0x2Au  /* VERIFY, 2 bytes little-endian */
#define SB_REG_TORQUE_LIMIT     0x30u  /* VERIFY, 2 bytes, 0..1000 */
#define SB_REG_PRESENT_POSITION 0x38u  /* VERIFY, 2 bytes */
#define SB_REG_PRESENT_TEMP     0x3Fu  /* VERIFY */
#define SB_REG_PRESENT_CURRENT  0x45u  /* VERIFY, 2 bytes */

typedef struct {
  uint16_t position[SB_NUM_SERVOS];   /* ticks 0..4095 */
  int16_t current_ma[SB_NUM_SERVOS];
  uint8_t temperature_c[SB_NUM_SERVOS];
  uint8_t error[SB_NUM_SERVOS];       /* status packet error byte */
  uint32_t missing_mask;              /* bit i set: servo id i+1 did not answer */
} sb_state_t;

/* Pure packet helpers (no hardware; TODO bodies, testable on the host). Return the packet
 * length or 0 on error. */
uint8_t sb_checksum(const uint8_t *body, size_t len);
size_t sb_encode(uint8_t id, uint8_t instr, const uint8_t *params, size_t nparams,
                 uint8_t *out, size_t out_cap);
size_t sb_encode_sync_write_u16(uint8_t addr, const uint8_t *ids, const uint16_t *values,
                                size_t n, uint8_t *out, size_t out_cap);
/* Parse one status packet. Returns true and fills id/error/params on a valid checksum. */
bool sb_decode_status(const uint8_t *buf, size_t len, uint8_t *id, uint8_t *error,
                      uint8_t *params, size_t *nparams);

/* Hardware layer (TODO bodies). */
void sb_init(void);                                   /* UART half-duplex @ 1 Mbaud, DMA */
bool sb_ping(uint8_t id);
bool sb_write_goal_positions(const uint16_t goal_ticks[SB_NUM_SERVOS]);  /* one SYNC_WRITE */
bool sb_set_torque(bool enable);                      /* broadcast torque enable/disable */
bool sb_read_state(sb_state_t *state);                /* positions, currents, temperatures */

#endif /* GAMBIT_SERVO_BUS_H */

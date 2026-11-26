/* Joint, velocity, current and temperature limits enforced on the MCU, independent of the
 * host. Defaults mirror gambit_bringup/config/safety.yaml; the host may tighten (never loosen)
 * them with HL_CMD_SET_LIMITS. Units: servo ticks (4096 per rev), mA, deg C, ms. */
#ifndef GAMBIT_LIMITS_H
#define GAMBIT_LIMITS_H

#include <stdbool.h>
#include <stdint.h>

#include "servo_bus.h"

typedef enum {
  LIM_OK = 0,
  LIM_JOINT_RANGE = 1 << 0,
  LIM_STEP = 1 << 1,
  LIM_OVERCURRENT = 1 << 2,
  LIM_OVERTEMP = 1 << 3,
  LIM_SERVO_MISSING = 1 << 4,
  LIM_HEARTBEAT = 1 << 5,
} limit_fault_t;

typedef struct {
  uint16_t min_ticks[SB_NUM_SERVOS];
  uint16_t max_ticks[SB_NUM_SERVOS];
  uint16_t max_step_ticks;          /* per control cycle */
  int16_t max_current_ma;
  uint16_t overcurrent_hold_ms;
  uint8_t max_temperature_c;
  uint16_t heartbeat_timeout_ms;
} limits_t;

void limits_init_defaults(limits_t *lim);
/* 0 = ok, else OR of limit_fault_t. Does not modify the command. */
uint32_t limits_check_command(const limits_t *lim, const uint16_t prev[SB_NUM_SERVOS],
                              const uint16_t cmd[SB_NUM_SERVOS]);
/* Call every cycle with fresh telemetry; tracks over-current hold time. */
uint32_t limits_check_state(const limits_t *lim, const sb_state_t *st, uint32_t now_ms);
/* Accept new limits from the host only if every value is at least as strict. */
bool limits_tighten(limits_t *lim, const limits_t *requested);

#endif /* GAMBIT_LIMITS_H */

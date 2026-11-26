#include "limits.h"

/* Defaults from safety.yaml (URDF joint limits minus joint_margin, converted with
 * ticks = 2048 + rad * 4096 / (2 pi)). TODO: regenerate after calibration: the tick values
 * depend on each servo's calibration offset. measure/verify. */
void limits_init_defaults(limits_t *lim) {
  static const uint16_t lo[SB_NUM_SERVOS] = {829, 943, 979, 1000, 292, 1967};   /* TODO: verify */
  static const uint16_t hi[SB_NUM_SERVOS] = {3267, 3153, 3117, 3096, 3868, 3153}; /* TODO: verify */
  for (unsigned i = 0; i < SB_NUM_SERVOS; ++i) {
    lim->min_ticks[i] = lo[i];
    lim->max_ticks[i] = hi[i];
  }
  lim->max_step_ticks = 130;        /* ~0.2 rad (velocity.max_step_rad) */
  lim->max_current_ma = 1500;
  lim->overcurrent_hold_ms = 300;
  lim->max_temperature_c = 60;
  lim->heartbeat_timeout_ms = 500;
}

uint32_t limits_check_command(const limits_t *lim, const uint16_t prev[SB_NUM_SERVOS],
                              const uint16_t cmd[SB_NUM_SERVOS]) {
  /* TODO: for each joint: range check against min/max_ticks (LIM_JOINT_RANGE) and
   * |cmd - prev| <= max_step_ticks (LIM_STEP). Keep it branch-light; runs every cycle. */
  (void)lim; (void)prev; (void)cmd;
  return LIM_OK;
}

uint32_t limits_check_state(const limits_t *lim, const sb_state_t *st, uint32_t now_ms) {
  /* TODO: missing_mask -> LIM_SERVO_MISSING; temperature -> LIM_OVERTEMP; current above
   * max_current_ma continuously for overcurrent_hold_ms (static per-joint timestamps) ->
   * LIM_OVERCURRENT. */
  (void)lim; (void)st; (void)now_ms;
  return LIM_OK;
}

bool limits_tighten(limits_t *lim, const limits_t *requested) {
  /* TODO: accept only if requested ranges are inside the current ones and all scalar limits
   * are <= current; copy and return true, else return false (reply NACK). */
  (void)lim; (void)requested;
  return false;
}

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
  /* Design: for each joint: cmd outside [min_ticks, max_ticks] -> LIM_JOINT_RANGE;
   * |cmd - prev| > max_step_ticks -> LIM_STEP (compute the difference in a signed int).
   * OR the bits together; never modify the command (reject, do not "fix"). Runs every cycle,
   * so keep it simple and branch-light.
   * TODO: implement (returning LIM_OK until then means NO limits are enforced) */
  (void)lim; (void)prev; (void)cmd;
  return LIM_OK;
}

uint32_t limits_check_state(const limits_t *lim, const sb_state_t *st, uint32_t now_ms) {
  /* Design: missing_mask != 0 -> LIM_SERVO_MISSING; any temperature > max_temperature_c ->
   * LIM_OVERTEMP; |current| > max_current_ma continuously for overcurrent_hold_ms ->
   * LIM_OVERCURRENT (static per-joint "over since" timestamps, cleared when the current
   * drops; short grasp spikes must not trip). Same logic as EStop.update_telemetry in
   * safety.ipynb. Use unsigned subtraction for now_ms differences (wraps correctly).
   * TODO: implement */
  (void)lim; (void)st; (void)now_ms;
  return LIM_OK;
}

bool limits_tighten(limits_t *lim, const limits_t *requested) {
  /* Design: accept only if every requested range lies inside the current one and every
   * scalar limit is <= the current one (the host may tighten, never loosen); then copy and
   * return true, else leave *lim unchanged and return false (caller replies NACK).
   * TODO: implement */
  (void)lim; (void)requested;
  return false;
}

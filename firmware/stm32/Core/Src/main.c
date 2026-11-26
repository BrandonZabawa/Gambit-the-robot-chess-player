/* Gambit STM32 real-time layer: owns the servo bus, enforces limits, talks to the host.
 *
 * Main loop at 100 Hz (matches controllers.yaml update_rate):
 *   1. hl_poll(): handle host commands (targets, torque, e-stop, heartbeat)
 *   2. sb_read_state(): positions / currents / temperatures
 *   3. limits_check_state() + heartbeat timeout -> latch e-stop on any fault
 *   4. if not e-stopped and a new target is pending and limits_check_command() passes:
 *        sb_write_goal_positions()
 *   5. periodically send HL_RSP_STATE
 * On e-stop: hold the current position (estop.on_trip: hold) or disable torque.
 *
 * Built with STM32CubeIDE / CubeMX-generated HAL (see README.md); without USE_HAL_DRIVER this
 * file still compiles (host syntax check) with the HAL calls stubbed out. */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "host_link.h"
#include "limits.h"
#include "servo_bus.h"

#ifdef USE_HAL_DRIVER
#include "main.h"
#define NOW_MS() HAL_GetTick()
#else
static uint32_t fake_ms;
#define NOW_MS() (fake_ms++)
#endif

#define CONTROL_PERIOD_MS 10u
#define STATE_REPORT_EVERY 5u   /* cycles -> 20 Hz state reports */

typedef struct {
  limits_t limits;
  sb_state_t state;
  uint16_t target[SB_NUM_SERVOS];
  uint16_t last_cmd[SB_NUM_SERVOS];
  bool target_pending;
  bool torque_on;
  bool estop_latched;
  uint32_t faults;
  uint32_t last_heartbeat_ms;
} app_t;

static app_t app;

static void app_estop(uint32_t faults) {
  /* Design: latch (estop_latched = true, OR the faults in), then make the arm safe: hold the
   * last measured positions as goals (estop.on_trip: hold) or sb_set_torque(false); send one
   * HL_RSP_FAULT to the host. Only CLEAR_ESTOP unlatches.
   * TODO: implement */
  (void)faults;
}

void app_handle_command(const hl_frame_t *f) {
  /* Design: switch (f->type) over hl_type_t: PING -> ACK; HEARTBEAT -> update
   * last_heartbeat_ms; SET_JOINT_TARGETS -> unpack 6 x u16 LE into app.target, set
   * target_pending (NACK while e-stopped); SET_TORQUE; GET_STATE -> send state;
   * SET_LIMITS -> limits_tighten(); ESTOP -> app_estop(0); CLEAR_ESTOP -> clear only if
   * limits_check_state() is clean; GET_VERSION. Every command resets the heartbeat timer.
   * Check f->len against the expected payload size before unpacking.
   * TODO: implement */
  (void)f;
}

static void control_step(uint32_t now) {
  /* Design: one 10 ms cycle, steps 1-4 of the header comment: hl_poll; sb_read_state ->
   * limits_check_state (fault -> app_estop); heartbeat older than heartbeat_timeout_ms ->
   * app_estop(LIM_HEARTBEAT); if not e-stopped and a target is pending: limits_check_command
   * against last_cmd, then sb_write_goal_positions and remember it as last_cmd. Decide which
   * faults latch the e-stop and which only NACK the command.
   * TODO: implement */
  (void)now;
  (void)app_estop;  /* referenced so the skeleton builds warning-free; remove when used */
}

int main(void) {
#ifdef USE_HAL_DRIVER
  HAL_Init();
  SystemClock_Config();   /* CubeMX generated */
  MX_GPIO_Init();
  /* MX_DMA_Init(); MX_USART1_UART_Init(); MX_USB_DEVICE_Init(); MX_TIM6_Init(); -- CubeMX */
#endif
  memset(&app, 0, sizeof app);
  limits_init_defaults(&app.limits);
  sb_init();
  hl_init();
  /* TODO: ping all servos (sb_ping 1..6); refuse to enable torque if any is missing; read
   * present positions into last_cmd so the first command is checked against reality. */
  app.last_heartbeat_ms = NOW_MS();

  uint32_t cycle = 0;
  for (;;) {
    const uint32_t now = NOW_MS();
    control_step(now);
    if (++cycle % STATE_REPORT_EVERY == 0) {
      /* TODO: build and hl_send() an HL_RSP_STATE frame. */
    }
    /* TODO: replace busy timing with a TIM6 100 Hz interrupt setting a flag, then __WFI(). */
#ifndef USE_HAL_DRIVER
    if (cycle > 3) break;   /* host syntax-check build: do not loop forever */
#else
    HAL_Delay(CONTROL_PERIOD_MS);
#endif
  }
  return 0;
}

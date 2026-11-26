// gambit_arm_daemon: the C++ path that owns the servo bus.
//   gambit_arm_daemon [--port /dev/ttyACM0] [--baud 1000000] [--dry-run]
// With --dry-run it never opens the port; it logs the packets it would send (useful in CI).
//
// Planned loop (TODO): read joint targets from the planner (IPC), validate them against
// gambit_bringup/config/safety.yaml limits, SYNC_WRITE goal positions, read back present
// positions/currents, publish state, and drop torque on e-stop or heartbeat loss.
#include <spdlog/spdlog.h>

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "gambit/kinematics/so101.hpp"
#include "gambit/servo_bus/serial_port.hpp"
#include "gambit/servo_bus/sts_protocol.hpp"

namespace {
std::atomic<bool> g_stop{false};

void on_signal(int) {
  // TODO: set g_stop = true (only async-signal-safe work here: a lock-free atomic store).
  // Signal handlers must not throw.
}

[[maybe_unused]] std::string hex(const std::vector<std::uint8_t>& /*v*/) {
  // Design: format each byte as "%02X " (std::snprintf or std::format) for dry-run logs, so
  // packets can be compared by eye with the notebook / firmware vectors.
  throw std::logic_error("TODO: hex");
}
}  // namespace

int main(int argc, char** argv) {
  std::string port = "/dev/ttyACM0";
  int baud = 1'000'000;
  bool dry_run = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--port" && i + 1 < argc) port = argv[++i];
    else if (a == "--baud" && i + 1 < argc) baud = std::atoi(argv[++i]);
    else if (a == "--dry-run") dry_run = true;
    else {
      spdlog::error("unknown argument {}", a);
      return 2;
    }
  }
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
  spdlog::info("gambit_arm_daemon starting (port={}, baud={}, dry_run={})", port, baud, dry_run);

  // Servo IDs 1..6 = shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
  // (LeRobot SO-101 convention; verify after run_servo_setup.ipynb).
  [[maybe_unused]] constexpr std::uint8_t kIds[] = {1, 2, 3, 4, 5, 6};

  // Design (TODO, in this order):
  // 1. Unless --dry-run: gambit::servo::SerialPort::open(port, baud); on failure log
  //    spdlog::critical with e.what() and return 1.
  // 2. Startup check: PING every id in kIds (log the packet with hex() in dry-run), read and
  //    decode_status() each reply (skip the half-duplex echo), refuse to start if any
  //    servo is missing.
  // 3. Sanity log: FK of the zero pose (gambit::kinematics::forward) at debug level.
  // 4. Control loop at a fixed rate (e.g. 100 Hz, matching controllers.yaml), using
  //    std::chrono::steady_clock deadlines (sleep_until, not sleep_for, to avoid drift):
  //    read joint targets from the planner (IPC: Unix socket or shared memory; message
  //    format shared with firmware/stm32/protocol.md), validate against safety.yaml limits,
  //    SYNC_WRITE goal positions, read back positions / currents, publish state, and record
  //    loop period / jitter for the Python vs C++ comparison.
  // 5. Heartbeat watchdog and e-stop: no planner message within the timeout -> hold or drop
  //    torque. Exit the loop on g_stop (SIGINT / SIGTERM), disable torque, close the port.
  spdlog::error("TODO: gambit_arm_daemon is a skeleton (see the design steps in main())");
  (void)g_stop;
  return 1;
}

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
#include <string>

#include "gambit/kinematics/so101.hpp"
#include "gambit/servo_bus/serial_port.hpp"
#include "gambit/servo_bus/sts_protocol.hpp"

namespace {
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

std::string hex(const std::vector<std::uint8_t>& v) {
  std::string s;
  char buf[4];
  for (auto b : v) {
    std::snprintf(buf, sizeof(buf), "%02X ", b);
    s += buf;
  }
  return s;
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
  constexpr std::uint8_t kIds[] = {1, 2, 3, 4, 5, 6};

  gambit::servo::SerialPort bus;
  if (!dry_run) {
    try {
      bus.open(port, baud);
    } catch (const std::exception& e) {
      spdlog::critical("cannot open {}: {}", port, e.what());
      return 1;
    }
  }

  for (auto id : kIds) {
    const auto pkt = gambit::servo::ping(id);
    spdlog::info("PING id {} -> {}", id, hex(pkt));
    if (!dry_run) {
      bus.write(pkt);
      // TODO: read the status packet (and skip our own echo on half-duplex adapters), decode
      // it with decode_status(), and refuse to start if any servo is missing.
    }
  }

  const auto t = gambit::kinematics::forward(gambit::kinematics::JointVector::Zero());
  spdlog::debug("FK(zero) tool = [{:.4f}, {:.4f}, {:.4f}]", t.translation().x(),
                t.translation().y(), t.translation().z());

  // TODO: IPC with the Python planner / ROS 2 node (Unix socket or shared memory; message
  // format shared with firmware/stm32/protocol.md), a fixed-rate control loop (e.g. 100 Hz,
  // matching controllers.yaml), limit checks, and a heartbeat watchdog that disables torque.
  while (!g_stop && !dry_run) {
    break;  // TODO: control loop body
  }

  spdlog::info("gambit_arm_daemon stopping");
  return 0;
}

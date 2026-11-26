# gambit_cpp: the C++ layer

C++20 libraries and the arm daemon: the second control path next to the Python notebooks in
`gambit_arm/notebooks/`, so both can be measured on the same task (README: "Measure both paths
on the same task"). The packet bytes and FK numbers in the tests are the same ones the
notebooks' self-checks use.

| Target | What | Status |
|---|---|---|
| `gambit_servo_bus` | STS3215 packet encode/decode + checksum (`sts_protocol`), POSIX termios port (`serial_port`) | protocol done; port open/configure/read/write basic, latency tuning TODO |
| `gambit_kinematics` | SO-101 FK from the URDF joint origins (Eigen), numeric Jacobian | FK done; IK TODO (port `ik_dls` from kinematics.ipynb) |
| `gambit_square_map` | square <-> world from board.yaml numbers, graveyard slots | done; YAML loading TODO |
| `gambit_inference` | `OnnxClassifier`: 64 crops -> (64, 13) scores | interface + stub; ONNX Runtime calls TODO |
| `gambit_arm_daemon` | owns the servo bus, spdlog logging, `--dry-run` | skeleton; IPC + control loop TODO |
| `gambit_cpp_py` | pybind11 module (optional) | a few functions bound |
| `gambit_tests` | GoogleTest: checksum, packet round trip, square map, FK | passing |

## Build

```bash
cmake -S gambit_cpp -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
./build/gambit_arm_daemon --dry-run
```

Options:

| Option | Default | Effect |
|---|---|---|
| `GAMBIT_BUILD_TESTS` | ON | GoogleTest (FetchContent v1.15.2) |
| `GAMBIT_BUILD_PYTHON` | OFF | pybind11 module `gambit_cpp_py` (FetchContent v2.13.6) |
| `GAMBIT_WITH_ONNX` | OFF | link ONNX Runtime from `-DONNXRUNTIME_ROOT=/path` (or env `ONNXRUNTIME_ROOT`) |
| `GAMBIT_BUILD_DAEMON` | ON | `gambit_arm_daemon` |

Dependencies: Eigen 3.4 (`apt install libeigen3-dev`, else fetched from gitlab), spdlog
(system package if found, else fetched v1.14.1), GoogleTest and pybind11 always fetched.
FetchContent uses shallow git clones (GitHub/GitLab archive downloads may be blocked behind
some proxies; git works). Needs CMake >= 3.24, Ninja, a C++20 compiler (tested GCC 13.3).

Python module:

```bash
cmake -S gambit_cpp -B build -G Ninja -DGAMBIT_BUILD_PYTHON=ON && cmake --build build
PYTHONPATH=build python3 -c "import gambit_cpp_py as g; print(g.ping_packet(1), g.square_to_world('e4'))"
```

## Layout

```
CMakeLists.txt
libs/servo_bus/{include/gambit/servo_bus,src}/   sts_protocol, serial_port
libs/kinematics/{include/gambit/kinematics,src}/ so101
libs/square_map/{include/gambit/square_map,src}/ square_map
libs/inference/{include/gambit/inference,src}/   onnx_classifier
bindings/pybind_module.cpp
app/gambit_arm_daemon.cpp
tests/
```

## TODO

- [ ] IK (`so101.cpp inverse`) and its round-trip test
- [ ] YAML config (board.yaml, safety.yaml) loading, e.g. yaml-cpp via FetchContent
- [ ] daemon: status-packet reads, limit checks from safety.yaml, 100 Hz loop, IPC with the
      planner, heartbeat; host client for the STM32 link (`firmware/stm32/protocol.md`)
- [ ] ONNX Runtime run() and a test with a tiny exported model
- [ ] latency benchmark vs the Python `ServoBus.round_trip_time`

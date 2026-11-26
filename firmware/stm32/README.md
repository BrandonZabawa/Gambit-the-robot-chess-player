# Gambit STM32 real-time layer (optional, planned)

Bare-metal firmware that sits between the computer and the SO-ARM101 servo bus. From the
README Software table: *"Optional bare-metal STM32 firmware that owns the servo bus, enforces
joint and torque limits, and talks to the computer over a serial or network link."*
Development order: get the arm working from the laptop first (Python, then `gambit_cpp`), then
move the bus here and measure both paths on the same task (see
`gambit_arm/notebooks/00_overview.ipynb`, "Python vs C++ path").

Status: **skeleton**. Packet encoding/checksums (servo bus and host link CRC) are implemented
and match the Python and C++ code; everything that touches hardware is a `TODO`.

## Role

| Responsibility | Where |
|---|---|
| Own the servo bus: half-duplex UART at 1 Mbaud, STS3215 packets, SYNC_WRITE goals, read telemetry | `Core/Src/servo_bus.c` |
| Enforce joint range, step size, over-current (with hold time), temperature, heartbeat | `Core/Src/limits.c` |
| Host link over USB CDC or UART: framed, CRC-16, sequence numbers | `Core/Src/host_link.c`, [`protocol.md`](protocol.md) |
| 100 Hz control loop, latched e-stop | `Core/Src/main.c` |

Limits default to `gambit_bringup/config/safety.yaml` converted to servo ticks; the host can
only *tighten* them. The MCU keeps enforcing them even if the host crashes (heartbeat timeout).

## Suggested board

* **NUCLEO-G474RE** (Cortex-M4F 170 MHz, several USARTs with single-wire half-duplex mode,
  ST-LINK virtual COM port for the host link, cheap). Alternatives: NUCLEO-F446RE, or a
  Black Pill (STM32F411) if native USB CDC is wanted.
* Bus wiring: USART TX pin in single-wire mode to the servo data line (3.3 V MCU vs the bus's
  logic level: check the servo's input threshold; add a level shifter / 74LVC1T45 buffer if
  needed). Servo power (7.4 V or 12 V depending on servo version) from its own supply, common
  ground with the MCU. A hardware e-stop switch cuts servo power, independent of firmware.

## CubeMX / HAL notes

1. New STM32CubeIDE project for the board; HAL drivers.
2. USARTx: *Single Wire (Half-Duplex)*, 1000000 baud, 8N1; DMA RX + TX; global interrupt
   (IDLE line detection for end of status packet).
3. Host link: USART2 (ST-LINK VCP) at 921600 baud with DMA circular RX, **or** USB_DEVICE CDC.
4. TIM6 at 100 Hz for the control tick.
5. Copy `Core/Src/{servo_bus,limits,host_link}.c` and `Core/Inc/*.h` into the generated project;
   merge `main.c` into the generated `main.c` between the `USER CODE` markers (keep CubeMX's
   init calls). `USE_HAL_DRIVER` is defined by the generated project.

Host syntax check (no HAL, the hardware functions are stubs):

```bash
gcc -std=c11 -Wall -Wextra -Ifirmware/stm32/Core/Inc firmware/stm32/Core/Src/*.c -o /tmp/fw && /tmp/fw
```

## TODO

- [ ] choose and order the board; CubeMX project
- [ ] `servo_bus.c` hardware functions (half-duplex direction switching, timeouts, SYNC_READ)
- [ ] `limits.c` checks; regenerate tick limits after calibration
- [ ] `host_link.c` RX state machine and transport; host-side client in `gambit_cpp`
- [ ] measure loop jitter (GPIO toggle + logic analyzer) and bus round-trip vs the PC paths

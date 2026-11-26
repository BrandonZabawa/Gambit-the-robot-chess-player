# Host <-> STM32 protocol

Transport: USB CDC or UART (921600 8N1). Same framing either way. Implemented in
`Core/Src/host_link.c` (MCU side); the host side belongs in `gambit_cpp` (TODO).

## Frame

```
+------+------+------+-----+-----+-----------------+--------+--------+
| 0xAA | 0x55 | TYPE | SEQ | LEN | PAYLOAD (LEN B) | CRC lo | CRC hi |
+------+------+------+-----+-----+-----------------+--------+--------+
```

* `LEN` 0..64. Multi-byte fields little-endian.
* `CRC` = CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no reflection, no xor-out) over
  `TYPE SEQ LEN PAYLOAD`. Check value: CRC("123456789") = 0x29B1.
* `SEQ`: host increments per command (wraps at 255); replies echo the command's `SEQ`.
  Unsolicited MCU frames (`FAULT`, periodic `STATE`) use their own counter.
* Bad CRC or unknown type: frame dropped (no reply); the host times out (20 ms) and retries.

## Commands (host -> MCU)

| Type | Name | Payload | Reply |
|---|---|---|---|
| 0x01 | PING | none | ACK |
| 0x02 | HEARTBEAT | none | none (any command also counts as a heartbeat) |
| 0x03 | SET_TORQUE | u8 enable (0/1) | ACK, or NACK if e-stopped |
| 0x04 | SET_JOINT_TARGETS | 6 x u16 goal ticks (ids 1..6: pan, lift, elbow, wrist_flex, wrist_roll, gripper) | ACK, or NACK + fault bits if a limit check fails |
| 0x05 | GET_STATE | none | STATE |
| 0x06 | SET_LIMITS | 6 x u16 min, 6 x u16 max, u16 max_step, i16 max_current_ma, u16 hold_ms, u8 max_temp_c, u16 heartbeat_ms (33 B) | ACK if every limit is at least as strict as the current one, else NACK |
| 0x07 | ESTOP | none | ACK (e-stop latched) |
| 0x08 | CLEAR_ESTOP | none | ACK, or NACK if a fault is still present |
| 0x09 | GET_VERSION | none | VERSION |

## Replies / events (MCU -> host)

| Type | Name | Payload |
|---|---|---|
| 0x81 | ACK | u8 acked command type |
| 0x82 | NACK | u8 command type, u32 fault bits |
| 0x83 | STATE | u32 timestamp_ms, 6 x u16 position ticks, 6 x i16 current mA, 6 x u8 temperature C, u32 fault bits, u8 flags (bit0 torque on, bit1 e-stop latched) (39 B) |
| 0x84 | FAULT | u32 fault bits, u8 e-stop latched (sent once when a fault latches) |
| 0x85 | VERSION | u8 major, u8 minor, u8 patch |

Fault bits (`limit_fault_t` in `Core/Inc/limits.h`): bit0 joint range, bit1 step too large,
bit2 over-current, bit3 over-temperature, bit4 servo missing, bit5 heartbeat lost.

## Timing

* Host sends SET_JOINT_TARGETS at up to 100 Hz (one per MCU control cycle) and a HEARTBEAT at
  least every 100 ms when idle; the MCU latches e-stop after `heartbeat_timeout_ms` (500 ms
  default, safety.yaml `estop.heartbeat_timeout_s`).
* MCU sends STATE at 20 Hz unsolicited, or on GET_STATE.

## Units

Positions are raw servo ticks (4096 per revolution) *after* calibration: the host converts
radians with the calibration offsets (`servo_bus.ipynb` `raw_from_joint_rad`). Keeping the MCU
in ticks keeps it free of floating point and calibration logic.

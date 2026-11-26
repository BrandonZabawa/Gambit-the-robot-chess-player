// Minimal POSIX termios serial port for the Waveshare Bus Servo Adapter (USB to half-duplex
// TTL). 8N1, raw mode, default 1 Mbaud (STS3215 factory default; verify).
#pragma once

#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace gambit::servo {

class SerialPort {
 public:
  SerialPort() = default;
  ~SerialPort();
  SerialPort(const SerialPort&) = delete;
  SerialPort& operator=(const SerialPort&) = delete;
  SerialPort(SerialPort&& other) noexcept;
  SerialPort& operator=(SerialPort&& other) noexcept;

  // Open and configure (raw 8N1, no flow control). Throws std::system_error on failure.
  void open(const std::string& device, int baud = 1'000'000);
  void close();
  bool is_open() const { return fd_ >= 0; }

  // Write all bytes. Returns the number written.
  std::size_t write(std::span<const std::uint8_t> data);

  // Read up to `max_bytes`, waiting at most `timeout` for the first byte.
  std::vector<std::uint8_t> read(std::size_t max_bytes, std::chrono::milliseconds timeout);

  // Discard unread input (stale status packets / echo of our own TX on half duplex).
  void flush_input();

 private:
  int fd_ = -1;
};

}  // namespace gambit::servo

// POSIX serial port for the Waveshare Bus Servo Adapter. Design notes per function;
// implementations are TODO. Key APIs: open(2), termios (tcgetattr / cfmakeraw / cfsetispeed /
// cfsetospeed / tcsetattr / tcflush / tcdrain), poll(2), read(2) / write(2).
#include "gambit/servo_bus/serial_port.hpp"

#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace gambit::servo {
namespace {

[[maybe_unused]] speed_t to_speed(int /*baud*/) {
  // Design:
  // - Map the integer baud to the termios constant (B9600 .. B1000000) with a switch;
  //   anything else -> std::invalid_argument.
  // - Non-standard rates (e.g. 128000) need termios2 / BOTHER on Linux or IOSSIOSPEED on
  //   macOS; not needed while the servos run at the 1 Mbaud factory default.
  throw std::logic_error("TODO: to_speed");
}

[[maybe_unused]] [[noreturn]] void throw_errno(const char* /*what*/) {
  // Design: throw std::system_error(errno, std::generic_category(), what) so callers get the
  // OS error text ("Permission denied" -> add yourself to the dialout group).
  throw std::logic_error("TODO: throw_errno");
}

}  // namespace

SerialPort::~SerialPort() {
  // TODO: close() the descriptor (RAII). Destructors must not throw, so close() must not
  // throw either.
}

SerialPort::SerialPort(SerialPort&& /*other*/) noexcept {
  // TODO: take other's fd_ and leave other with -1 (std::exchange), so only one object
  // ever closes the descriptor. noexcept: must not throw.
}

SerialPort& SerialPort::operator=(SerialPort&& /*other*/) noexcept {
  // TODO: guard self-assignment, close() our own fd, then take other's (std::exchange).
  // noexcept: must not throw.
  return *this;
}

void SerialPort::open(const std::string& /*device*/, int /*baud*/) {
  // Design:
  // - close() first so open() can be called again; ::open with O_RDWR | O_NOCTTY | O_NONBLOCK
  //   (no controlling terminal; non-blocking so poll() controls timing).
  // - tcgetattr, then cfmakeraw (no echo, no line discipline, 8 data bits), CLOCAL | CREAD,
  //   no parity, 1 stop bit, no RTS/CTS, no XON/XOFF; VMIN = VTIME = 0 because reads are
  //   driven by poll().
  // - cfsetispeed / cfsetospeed with to_speed(baud), tcsetattr(TCSANOW), tcflush(TCIOFLUSH).
  // - Every failing syscall -> throw_errno with the call's name.
  // Later: ASYNC_LOW_LATENCY via TIOCSSERIAL (USB-serial latency timer, default ~16 ms,
  // dominates the round trip); measure a PING round trip before and after.
  throw std::logic_error("TODO: SerialPort::open");
}

void SerialPort::close() {
  // Design: if fd_ >= 0, ::close it and set fd_ = -1. Must not throw (called from the
  // destructor), so ignore / log close errors.
  // TODO: implement
}

std::size_t SerialPort::write(std::span<const std::uint8_t> /*data*/) {
  // Design:
  // - std::logic_error if not open.
  // - Loop ::write until every byte is out (partial writes happen on non-blocking fds);
  //   retry on EAGAIN / EINTR, throw_errno otherwise.
  // - tcdrain so the bytes have actually left before we start waiting for the reply
  //   (matters on half duplex: the servo answers ~immediately).
  throw std::logic_error("TODO: SerialPort::write");
}

std::vector<std::uint8_t> SerialPort::read(std::size_t /*max_bytes*/,
                                           std::chrono::milliseconds /*timeout*/) {
  // Design:
  // - std::logic_error if not open.
  // - Loop: poll(POLLIN, timeout); 0 = timeout -> return what we have; then ::read into a
  //   stack buffer and append, until max_bytes. Retry on EINTR / EAGAIN.
  // - Decision for you: per-chunk timeout (simple) vs one overall deadline
  //   (std::chrono::steady_clock, better for a fixed-rate loop), and stopping as soon as a
  //   full status packet (LEN byte known) has arrived.
  throw std::logic_error("TODO: SerialPort::read");
}

void SerialPort::flush_input() {
  // Design: tcflush(fd_, TCIFLUSH) if open; drops stale replies and our own echo.
  throw std::logic_error("TODO: SerialPort::flush_input");
}

}  // namespace gambit::servo

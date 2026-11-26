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

speed_t to_speed(int baud) {
  switch (baud) {
    case 9600: return B9600;
    case 57600: return B57600;
    case 115200: return B115200;
    case 500000: return B500000;
    case 1000000: return B1000000;
    // TODO: non-standard rates (e.g. 128000) need termios2 / BOTHER on Linux, or IOSSIOSPEED
    // on macOS. The STS3215 factory default is 1 Mbaud, so start there.
    default: throw std::invalid_argument("unsupported baud rate");
  }
}

[[noreturn]] void throw_errno(const char* what) {
  throw std::system_error(errno, std::generic_category(), what);
}

}  // namespace

SerialPort::~SerialPort() { close(); }

SerialPort::SerialPort(SerialPort&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}

SerialPort& SerialPort::operator=(SerialPort&& other) noexcept {
  if (this != &other) {
    close();
    fd_ = std::exchange(other.fd_, -1);
  }
  return *this;
}

void SerialPort::open(const std::string& device, int baud) {
  close();
  fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd_ < 0) throw_errno("open serial port");

  termios tio{};
  if (tcgetattr(fd_, &tio) != 0) throw_errno("tcgetattr");
  cfmakeraw(&tio);                       // no echo, no canonical mode, 8 data bits
  tio.c_cflag |= (CLOCAL | CREAD);
  tio.c_cflag &= ~(PARENB | CSTOPB);     // no parity, 1 stop bit
  tio.c_cflag &= ~CRTSCTS;               // no hardware flow control
  tio.c_iflag &= ~(IXON | IXOFF | IXANY);
  tio.c_cc[VMIN] = 0;                    // reads are driven by poll() below
  tio.c_cc[VTIME] = 0;
  const speed_t s = to_speed(baud);
  cfsetispeed(&tio, s);
  cfsetospeed(&tio, s);
  if (tcsetattr(fd_, TCSANOW, &tio) != 0) throw_errno("tcsetattr");
  tcflush(fd_, TCIOFLUSH);
  // TODO: set ASYNC_LOW_LATENCY via TIOCSSERIAL (FTDI/CH340 latency timer) and measure the
  // round-trip time of a PING; the default 16 ms USB latency dominates the control loop.
}

void SerialPort::close() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

std::size_t SerialPort::write(std::span<const std::uint8_t> data) {
  if (fd_ < 0) throw std::logic_error("serial port not open");
  std::size_t done = 0;
  while (done < data.size()) {
    const ssize_t n = ::write(fd_, data.data() + done, data.size() - done);
    if (n < 0) {
      if (errno == EAGAIN || errno == EINTR) continue;
      throw_errno("write serial port");
    }
    done += static_cast<std::size_t>(n);
  }
  tcdrain(fd_);
  return done;
}

std::vector<std::uint8_t> SerialPort::read(std::size_t max_bytes,
                                           std::chrono::milliseconds timeout) {
  if (fd_ < 0) throw std::logic_error("serial port not open");
  std::vector<std::uint8_t> out;
  out.reserve(max_bytes);
  pollfd pfd{fd_, POLLIN, 0};
  // TODO: use one overall deadline instead of a per-chunk timeout, and stop as soon as a full
  // status packet (length byte known) has arrived.
  while (out.size() < max_bytes) {
    const int r = ::poll(&pfd, 1, static_cast<int>(timeout.count()));
    if (r < 0) {
      if (errno == EINTR) continue;
      throw_errno("poll serial port");
    }
    if (r == 0) break;  // timeout
    std::uint8_t buf[256];
    const std::size_t want = std::min(sizeof(buf), max_bytes - out.size());
    const ssize_t n = ::read(fd_, buf, want);
    if (n < 0) {
      if (errno == EAGAIN || errno == EINTR) continue;
      throw_errno("read serial port");
    }
    if (n == 0) break;
    out.insert(out.end(), buf, buf + n);
  }
  return out;
}

void SerialPort::flush_input() {
  if (fd_ >= 0) tcflush(fd_, TCIFLUSH);
}

}  // namespace gambit::servo

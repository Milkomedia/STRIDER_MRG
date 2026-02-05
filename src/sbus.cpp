#include "sbus.hpp"
#include "utils.hpp"
#include "params.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include <immintrin.h> // _mm_pause
#include <pthread.h>
#include <sched.h>

#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>

#include <asm/termbits.h>   // termios2, TCGETS2/TCSETS2, VMIN/VTIME
#include <asm/termbits.h>
#include <linux/serial.h>

// Stop only; caller must join the owning std::thread before destroying this object.
SBUS::~SBUS() {
  request_stop();
}

void SBUS::call_kill(const char* msg) const {
  KillFn fn = kill_fn_.load(std::memory_order_acquire);
  if (fn) fn(msg);
}

bool SBUS::read_latest(SBUSFrame& out) const {
  const uint64_t s0 = seq_.load(std::memory_order_acquire);
  if (s0 == 0) return false;

  for (int spin = 0; spin < 100; ++spin) {
    const uint64_t s1 = seq_.load(std::memory_order_acquire);
    if (s1 & 1ULL) { _mm_pause(); continue; } // writer in progress

    const SBUSFrame tmp = buf_;

    const uint64_t s2 = seq_.load(std::memory_order_acquire);
    if (s1 == s2) { out = tmp; return true; } // consistent snapshot
  }
  return false;
}

static bool configure_sbus_port(int fd, std::string& err) {
  termios2 tio{};
  if (ioctl(fd, TCGETS2, &tio) != 0) {
    err = "TCGETS2 failed: " + std::string(strerror(errno));
    return false;
  }

  // Raw mode base
  tio.c_iflag = 0;
  tio.c_oflag = 0;
  tio.c_lflag = 0;

  // 8 data bits + enable receiver + local
  tio.c_cflag &= ~CSIZE;
  tio.c_cflag |= (CS8 | CLOCAL | CREAD);

  // Even parity
  tio.c_cflag |= PARENB;
  tio.c_cflag &= ~PARODD;

  // 2 stop bits
  tio.c_cflag |= CSTOPB;

  // Disable HW flow control
  tio.c_cflag &= ~CRTSCTS;

  // Non-blocking read behavior via VMIN/VTIME
  tio.c_cc[VMIN]  = 0;
  tio.c_cc[VTIME] = 0;

  // Custom baud using BOTHER
  tio.c_cflag &= ~CBAUD;
  tio.c_cflag |= BOTHER;
  tio.c_ispeed = 100000;
  tio.c_ospeed = 100000;

  if (ioctl(fd, TCSETS2, &tio) != 0) {
    err = "TCSETS2 failed: " + std::string(strerror(errno));
    return false;
  }

  // Flush stale bytes
  constexpr int kTCIOFLUSH = 2; // same as TCIOFLUSH
  (void)::ioctl(fd, TCFLSH, kTCIOFLUSH);
  return true;
}

static bool decode_sbus_frame(const uint8_t* f, uint16_t out_ch[18], uint8_t& out_failsafe) {
  // Decode 25-byte SBUS frame.
  // out_ch[0..17], out_failsafe: 0 OK, 1 LOST, 2 FAILSAFE
  if (f[0] != 0x0F) return false;

  const uint8_t endb = f[24];
  if (!(endb == 0x00 || endb == 0x04 || endb == 0x14 || endb == 0x24)) {return false;}

  const uint8_t b1  = f[1];
  const uint8_t b2  = f[2];
  const uint8_t b3  = f[3];
  const uint8_t b4  = f[4];
  const uint8_t b5  = f[5];
  const uint8_t b6  = f[6];
  const uint8_t b7  = f[7];
  const uint8_t b8  = f[8];
  const uint8_t b9  = f[9];
  const uint8_t b10 = f[10];
  const uint8_t b11 = f[11];
  const uint8_t b12 = f[12];
  const uint8_t b13 = f[13];
  const uint8_t b14 = f[14];
  const uint8_t b15 = f[15];
  const uint8_t b16 = f[16];
  const uint8_t b17 = f[17];
  const uint8_t b18 = f[18];
  const uint8_t b19 = f[19];
  const uint8_t b20 = f[20];
  const uint8_t b21 = f[21];
  const uint8_t b22 = f[22];

  // 16 analog channels, 11-bit packed little-endian style
  out_ch[0]  = (uint16_t)(( b1        | (b2  << 8)) & 0x07FF);
  out_ch[1]  = (uint16_t)(((b2 >> 3)  | (b3  << 5)) & 0x07FF);
  out_ch[2]  = (uint16_t)(((b3 >> 6)  | (b4  << 2) | (b5  << 10)) & 0x07FF);
  out_ch[3]  = (uint16_t)(((b5 >> 1)  | (b6  << 7)) & 0x07FF);
  out_ch[4]  = (uint16_t)(((b6 >> 4)  | (b7  << 4)) & 0x07FF);
  out_ch[5]  = (uint16_t)(((b7 >> 7)  | (b8  << 1) | (b9  << 9))  & 0x07FF);
  out_ch[6]  = (uint16_t)(((b9 >> 2)  | (b10 << 6)) & 0x07FF);
  out_ch[7]  = (uint16_t)(((b10 >> 5) | (b11 << 3)) & 0x07FF);

  out_ch[8]  = (uint16_t)(( b12       | (b13 << 8)) & 0x07FF);
  out_ch[9]  = (uint16_t)(((b13 >> 3) | (b14 << 5)) & 0x07FF);
  out_ch[10] = (uint16_t)(((b14 >> 6) | (b15 << 2) | (b16 << 10)) & 0x07FF);
  out_ch[11] = (uint16_t)(((b16 >> 1) | (b17 << 7)) & 0x07FF);
  out_ch[12] = (uint16_t)(((b17 >> 4) | (b18 << 4)) & 0x07FF);
  out_ch[13] = (uint16_t)(((b18 >> 7) | (b19 << 1) | (b20 << 9))  & 0x07FF);
  out_ch[14] = (uint16_t)(((b20 >> 2) | (b21 << 6)) & 0x07FF);
  out_ch[15] = (uint16_t)(((b21 >> 5) | (b22 << 3)) & 0x07FF);

  const uint8_t flags = f[23];

  // digital channels (often used as ch17/ch18)
  out_ch[16] = (flags & 0x01) ? 2047 : 0;
  out_ch[17] = (flags & 0x02) ? 2047 : 0;

  const bool frame_lost = (flags & 0x04) != 0;
  const bool failsafe   = (flags & 0x08) != 0;

  if (failsafe) out_failsafe = 2;
  else if (frame_lost) out_failsafe = 1;
  else out_failsafe = 0;

  return true;
}

void SBUS::run() {
  stop_request_.store(false, std::memory_order_relaxed);
  is_running_.store(true, std::memory_order_release);

  try_pin_cpu(param::CPU_SBUS);
  try_set_prior(param::SBUS_PRIOR);

  const uint64_t start_ns = now_steady_ns();
  uint64_t last_good_ns = 0;

  int fd = -1;
  {
    fd = ::open(param::SBUS_PORT_NAME, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
      call_kill("[SBUS] open() failed");
      is_running_.store(false, std::memory_order_release);
      return;
    }

    std::string err;
    if (!configure_sbus_port(fd, err)) {
      std::fprintf(stderr, "[SBUS] port config error: %s\n", err.c_str());
      ::close(fd);
      call_kill("[SBUS] port config failed (100000/8E2)");
      is_running_.store(false, std::memory_order_release);
      return;
    }
  }

  // Accumulate bytes & parse frames
  std::array<uint8_t, 4096> rx{};
  size_t rx_len = 0;

  int lost_consecutive = 0;
  bool started = false;

  while (!stop_request_.load(std::memory_order_relaxed)) {
    // Stale timeout check (based on last valid frame)
    const uint64_t now_ns = now_steady_ns();
    const uint64_t ref_ns = started ? last_good_ns : start_ns;
    const uint64_t dt_ms = (now_ns - ref_ns) / 1000000ULL;
    const uint64_t timeout_ms = started ? (uint64_t)frame_timeout_ms_ : (uint64_t)first_frame_timeout_ms_;
    if (dt_ms > timeout_ms) {
      ::close(fd);
      stop_request_.store(true, std::memory_order_relaxed);
      call_kill(started ? "[SBUS] stale: no valid frame for 100ms" : "[SBUS] first frame timeout");
      break;
    }

    // poll for readable
    pollfd pfd{};
    pfd.fd = fd;
    pfd.events = POLLIN;
    const int pr = ::poll(&pfd, 1, poll_timeout_ms_);
    if (pr < 0) {
      ::close(fd);
      stop_request_.store(true, std::memory_order_relaxed);
      call_kill("[SBUS] poll() failed");
      break;
    }
    if (pr == 0) continue; // no data

    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
      ::close(fd);
      stop_request_.store(true, std::memory_order_relaxed);
      call_kill("[SBUS] serial error/hup");
      break;
    }

    // read bytes
    uint8_t tmp[512];
    const ssize_t n = ::read(fd, tmp, sizeof(tmp));
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
      ::close(fd);
      stop_request_.store(true, std::memory_order_relaxed);
      call_kill("[SBUS] read() failed");
      break;
    }
    if (n == 0) continue;

    // append to rx buffer (simple truncation if overflow)
    if (rx_len + (size_t)n > rx.size()) {
      // Keep last part to attempt resync
      const size_t keep = rx.size() / 2;
      if (rx_len > keep) {
        std::memmove(rx.data(), rx.data() + (rx_len - keep), keep);
        rx_len = keep;
      }
      else {rx_len = 0;}
    }
    std::memcpy(rx.data() + rx_len, tmp, (size_t)n);
    rx_len += (size_t)n;

    // parse frames
    size_t i = 0;
    while (i + 25 <= rx_len) {
      if (rx[i] != 0x0F) { ++i; continue; }

      uint16_t ch[18];
      uint8_t fs = 0;
      if (!decode_sbus_frame(rx.data() + i, ch, fs)) {
        ++i; // resync by shifting
        continue;
      }

      // valid frame
      const uint64_t host_ns = now_steady_ns();
      SBUSFrame s;
      s.host_time_ns = host_ns;
      for (int k = 0; k < 18; ++k) s.ch[k] = ch[k];
      s.failsafe = fs;

      {
        SeqWriteGuard g(seq_);
        buf_ = s;
      }

      frame_count_.fetch_add(1, std::memory_order_relaxed);
      last_host_ns_.store(host_ns, std::memory_order_relaxed);
      last_good_ns = host_ns;
      started = true;

      // Safety checks
      if (s.failsafe == 2) {
        ::close(fd);
        stop_request_.store(true, std::memory_order_relaxed);
        call_kill("[SBUS] FAILSAFE flag set");
        break;
      }
      if (s.failsafe == 1) {
        lost_consecutive++;
        if (lost_consecutive >= lost_consecutive_kill_) {
          ::close(fd);
          stop_request_.store(true, std::memory_order_relaxed);
          call_kill("[SBUS] LOST flag persisted");
          break;
        }
      }
      else {lost_consecutive = 0;}

      if (s.ch[kill_chn_idx_] != arm_val_) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "[SBUS] kill switch (ch[%d]=%u)", kill_chn_idx_, (unsigned)s.ch[kill_chn_idx_]);
        ::close(fd);
        stop_request_.store(true, std::memory_order_relaxed);
        call_kill(msg);
        break;
      }

      // consume this frame
      i += 25;
    }

    // drop consumed bytes
    if (i > 0) {
      if (i < rx_len) std::memmove(rx.data(), rx.data() + i, rx_len - i);
      rx_len -= i;
    }
  }

  // cleanup
  if (fd >= 0) {
    ::close(fd);
    fd = -1;
  }

  is_running_.store(false, std::memory_order_release);
}

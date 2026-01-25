#include "teensy.hpp"
#include "params.hpp"

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <thread>
#include <chrono>

Teensy::~Teensy() {
  close();
}

bool Teensy::open() {
  close();

  // Non-blocking socket to avoid stalling the control loop.
  sock_ = ::socket(PF_CAN, SOCK_RAW | SOCK_NONBLOCK, CAN_RAW);
  if (sock_ < 0) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "[TEENSY] socket(PF_CAN) failed: errno=%d (%s)", errno, std::strerror(errno));
    return false;
  }

  // Disable receiving to avoid any accidental RX overhead.
  (void)::setsockopt(sock_, SOL_CAN_RAW, CAN_RAW_FILTER, nullptr, 0);
  int recv_own = 0;
  (void)::setsockopt(sock_, SOL_CAN_RAW, CAN_RAW_RECV_OWN_MSGS, &recv_own, sizeof(recv_own));
  int loopback = 0;
  (void)::setsockopt(sock_, SOL_CAN_RAW, CAN_RAW_LOOPBACK, &loopback, sizeof(loopback));

  struct ifreq ifr;
  std::memset(&ifr, 0, sizeof(ifr));
  std::snprintf(ifr.ifr_name, IFNAMSIZ, "%s", param::CAN_PORT_NAME);

  if (::ioctl(sock_, SIOCGIFINDEX, &ifr) < 0) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "[TEENSY] ioctl(SIOCGIFINDEX, %s) failed: errno=%d (%s)", param::CAN_PORT_NAME, errno, std::strerror(errno));
    close();
    return false;
  }

  struct sockaddr_can addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.can_family = AF_CAN;
  addr.can_ifindex = ifr.ifr_ifindex;

  if (::bind(sock_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "[TEENSY] bind(%s) failed: errno=%d (%s)", param::CAN_PORT_NAME, errno, std::strerror(errno));
    close();
    return false;
  }

  return true;
}

void Teensy::close() {
  if (sock_ < 0) return;

  write_zeros();
  ::close(sock_);
  sock_ = -1;
}

bool Teensy::check() {
  if (sock_ < 0) {return false;}

  for (int i = 0; i < 10; ++i) {
    if (write_pwm(0.0, 0.0, 0.0, 0.0)) return true;
    if (sock_ < 0) return false; // fatal close() happened
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
  }
  return false;
}

void Teensy::write_zeros() {
  for (int i = 0; i < 10; ++i) {
    write_pwm(0.0, 0.0, 0.0, 0.0);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
}

bool Teensy::write_pwm(const double pwm1, const double pwm2, const double pwm3, const double pwm4) {
  if (sock_ < 0) {return false;}

  // Map [0,1] -> [16383, 32767]
  const uint16_t m1 = static_cast<uint16_t>(pwm1 * Pwm_scaler_) + Pwm_offset_;
  const uint16_t m2 = static_cast<uint16_t>(pwm2 * Pwm_scaler_) + Pwm_offset_;
  const uint16_t m3 = static_cast<uint16_t>(pwm3 * Pwm_scaler_) + Pwm_offset_;
  const uint16_t m4 = static_cast<uint16_t>(pwm4 * Pwm_scaler_) + Pwm_offset_;

  // set CAN frame
  struct can_frame frame;
  std::memset(&frame, 0, sizeof(frame));
  frame.can_id = CanId_;
  frame.can_dlc = 8;

  frame.data[0] = static_cast<uint8_t>((m1 >> 8) & 0xFF);  frame.data[1] = static_cast<uint8_t>(m1 & 0xFF);
  frame.data[2] = static_cast<uint8_t>((m2 >> 8) & 0xFF);  frame.data[3] = static_cast<uint8_t>(m2 & 0xFF);
  frame.data[4] = static_cast<uint8_t>((m3 >> 8) & 0xFF);  frame.data[5] = static_cast<uint8_t>(m3 & 0xFF);
  frame.data[6] = static_cast<uint8_t>((m4 >> 8) & 0xFF);  frame.data[7] = static_cast<uint8_t>(m4 & 0xFF);

  // CAN write
  const ssize_t n = ::write(sock_, &frame, sizeof(frame));

  if (n == static_cast<ssize_t>(sizeof(frame))) {return true;}

  return false;
}
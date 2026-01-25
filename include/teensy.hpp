#ifndef TEENSY_H
#define TEENSY_H

#include <atomic>
#include <cstdint>

// CAN protocol:
// can_id  : 0x123
// payload : 4 * uint16 (big-endian)
// mapping : pwm in [0,1] -> [16383, 32767]

class Teensy {
public:
  using KillFn = void(*)(const char* msg);

  Teensy() = default;
  ~Teensy();

  Teensy(const Teensy&) = delete;
  Teensy& operator=(const Teensy&) = delete;

  bool open(); // Open SocketCAN interface

  void close(); // Close socket

  bool write_pwm(const double pwm1, const double pwm2, const double pwm3, const double pwm4);

  void write_zeros();

  bool check(); // try zero-frame writes

private:
  int sock_ = -1;

  static constexpr uint32_t CanId_      = 0x123;
  static constexpr uint16_t Pwm_offset_ = 16383;
  static constexpr double   Pwm_scaler_ = 16384.;
};

#endif // TEENSY_H
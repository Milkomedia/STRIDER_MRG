#ifndef MMAP_LOGGER_H
#define MMAP_LOGGER_H

#include "params.hpp"

#include <cstdint>
#include <cstring>
#include <string>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace mmap_logger {

// -----------------------------
// Payload (packed, fixed layout)
// -----------------------------
#pragma pack(push, 1)
struct LogData {
  float t              = 0.0f;     // timestamp [sec]
  float pos_d[3]       = {0.0f};   // desired position [m]
  float pos[3]         = {0.0f};   // current position [m]
  float vel[3]         = {0.0f};   // opti velocity (filtered) [m/s]
  float rpy[3]         = {0.0f};   // current attitude [rad]
  float omega[3]       = {0.0f};   // imu angular rate (filtered) [rad/s]
  float rpy_raw[3]     = {0.0f};   // desired attitude from position ctrl [rad]
  float rpy_d[3]       = {0.0f};   // desired attitude reconstructed (R_d) [rad]
  float tau_d[3]       = {0.0f};   // desired torque (att ctrl) [N.m]
  float tau_off[2]     = {0.0f};   // cot-offset torque [N.m] (x,y)
  float tau_thrust[2]  = {0.0f};   // thrust-diff torque [N.m] (x,y)
  float tilt_rad[4]    = {0.0f};   // per-rotor tilt command [rad]
  float f_thrust[4]    = {0.0f};   // per-rotor thrust command [N]
  float f_total        = 0.0f;     // desired collective thrust [N]
  float r_cot[3]       = {0.0f};   // current CoT position [m] (x,y,z)
  float r_cot_cmd[3]   = {0.0f};   // optimal CoT command [m] (x,y,z_cmd)
  float q_mea[20]      = {0.0f};   // joint read [rad]
  float q_d[20]        = {0.0f};   // joint desired [rad]
  float solve_ms       = 0.0f;     // acados solve time [ms]
  int32_t solve_status = -1;       // solver status

  // idx0: ch0  x | idx1: ch1  y | idx2: ch2 z | idx3: ch3  heading
  // idx4: ch7  flight mode switch | idx5: ch8  arm/idle/rising toggle
  // idx6: ch10 L-dial | idx7: ch11 R-dial
  uint16_t sbus_used[8] = {0};
};
#pragma pack(pop)

// .bin File 
static FILE* log_fp = nullptr;

// 176 bytes with the layout above
static_assert(sizeof(LogData) == 360, "LogData size changed. Update Python reader offsets.");

// -----------------------------
// MMap header + ring buffer slot
// - Slot must be 8-byte aligned to avoid SIGBUS on atomic u64 access.
// -----------------------------
#pragma pack(push, 1)
struct MMapHeader {
  char     magic[8];      // "STRLOG2\0"
  uint32_t version;       // 2
  uint32_t header_size;   // sizeof(MMapHeader)
  uint32_t capacity;      // number of slots
  uint32_t slot_size;     // sizeof(Slot)
  uint64_t write_count;   // monotonically increasing
  uint64_t start_time_ns; // optional (0 ok)
  uint8_t  reserved[24];  // pad to 64 bytes
};
#pragma pack(pop)

static_assert(sizeof(MMapHeader) == 64, "MMapHeader must be 64 bytes.");

struct alignas(8) Slot {
  uint64_t seq;  // seqlock counter (odd=writing, even=stable)
  LogData  data; // payload
};

static_assert(alignof(Slot) == 8, "Slot alignment must be 8.");
static_assert(sizeof(Slot) == 368, "Slot size must be 368 bytes.");

static constexpr uint32_t k_Sec = 10;
static constexpr uint32_t k_Cap = static_cast<uint32_t>(1.0 / std::chrono::duration<double>(param::CTRL_DT).count()) * k_Sec;

inline uint64_t atomic_load_u64(const uint64_t* p) {return __atomic_load_n(p, __ATOMIC_ACQUIRE);}

inline void atomic_store_u64(uint64_t* p, uint64_t v) {__atomic_store_n(p, v, __ATOMIC_RELEASE);}

// -----------------------------
// Writer (SPSC producer-side)
// - No locks in control loop.
// -----------------------------
class MMapLogger {
public:
  explicit MMapLogger(const std::string& path = "/tmp/strider_log.mmap", bool reset = true)
  : path_(path), reset_(reset) {}

  ~MMapLogger() { close(); }

  MMapLogger(const MMapLogger&) = delete;
  MMapLogger& operator=(const MMapLogger&) = delete;

  void open();
  void close();

  void push(const LogData& x);

  uint64_t write_count() const;

  static constexpr uint32_t capacity() { return k_Cap; }
  static constexpr uint32_t header_size() { return sizeof(MMapHeader); }
  static constexpr uint32_t slot_size() { return sizeof(Slot); }

private:
  std::string path_;
  bool reset_ = true;

  bool opened_ = false;
  int fd_ = -1;
  size_t map_size_ = 0;

  uint8_t* base_ = nullptr;
  MMapHeader* header_ = nullptr;
  Slot* slots_ = nullptr;
};

} // namespace mmap_logger

#endif // MMAP_LOGGER_H
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

// .bin File 
static FILE* log_fp = nullptr;

// data length to store in mmap
static constexpr uint32_t k_Sec = 10; // [sec]
static constexpr uint32_t k_Cap = static_cast<uint32_t>(1.0 / std::chrono::duration<double>(param::CTRL_DT).count()) * k_Sec;

// -----------------------------
// Payload (packed, fixed layout)
// -----------------------------
#pragma pack(push, 1)
struct LogData {
  float t              =  0.0f;  // timestamp [sec]

  float pos_d[3]       = {0.0f}; // desired position [m]
  float vel_d[3]       = {0.0f}; // desired velocity [m/s]
  float acc_d[3]       = {0.0f}; // desired acceleration [m/s^2]
  float pos[3]         = {0.0f}; // current position [m]
  float vel[3]         = {0.0f}; // current velocity [m/s]
  float acc[3]         = {0.0f}; // current acceleration [m/s^2]

  float rpy_raw[3]     = {0.0f}; // desired attitude (from position ctrl) [rad]
  float rpy_d[3]       = {0.0f}; // desired attitude reconstructed (MRG applied) [rad]
  float omega_d[3]     = {0.0f}; // desired angular rate (from position ctrl) [rad/s]
  float alpha_d[3]     = {0.0f}; // desired angular acceleration (from position ctrl) [rad/s^2]
  float rpy[3]         = {0.0f}; // current attitude [rad]
  float omega[3]       = {0.0f}; // current angular rate [rad/s]
  float alpha[3]       = {0.0f}; // current angular acceleration [rad/s^2]

  float f_total        =  0.0f;  // desired collective thrust (from att ctrl) [N]
  float tau_d[3]       = {0.0f}; // desired torque (from att ctrl) [N.m]

  float tau_z_t        =  0.0f;  // thrust z torque [N.m]
  float tilt_rad[4]    = {0.0f}; // per-rotor tilt command [rad]
  float f_thrst[4]     = {0.0f}; // per-rotor thrust command [N]
  float f_thrst_con[4] = {0.0f}; // constrain-applied per-rotor thrust command [N]

  float tau_off[2]     = {0.0f}; // cot&com-offset torque [N.m] (x,y)
  float tau_thrust[3]  = {0.0f}; // thrust-diff torque [N.m] (x,y,z)

  float r_rotor1[2]   = {0.0f}; // current rotor-1 position [m] (x,y)
  float r_rotor2[2]   = {0.0f}; // current rotor-2 position [m] (x,y)
  float r_rotor3[2]   = {0.0f}; // current rotor-3 position [m] (x,y)
  float r_rotor4[2]   = {0.0f}; // current rotor-4 position [m] (x,y)
  float r_cot[2]       = {0.0f}; // current CoT position [m] (x,y)
  float r_rotor1_d[2]  = {0.0f}; // optimal rotor-1 position [m] (x,y)
  float r_rotor2_d[2]  = {0.0f}; // optimal rotor-2 position [m] (x,y)
  float r_rotor3_d[2]  = {0.0f}; // optimal rotor-3 position [m] (x,y)
  float r_rotor4_d[2]  = {0.0f}; // optimal rotor-4 position [m] (x,y)
  float r_cot_d[2]     = {0.0f}; // optimal CoT command [m] (x,y)
  
  float q[20]          = {0.0f};   // current joint angle [rad]
  float q_cmd[20]      = {0.0f};   // joint angle command [rad]

  float solve_ms       =  0.0f;  // acados solve time [ms]
  int32_t solve_status = -1;     // solver status (https://docs.acados.org/python_interface/index.html#acados_template.acados_ocp_options.AcadosOcpOptions.qp_solver)
  
  uint8_t phase        = 255;    // flight state phase
};
#pragma pack(pop)

// Packed size must match Python reader LOGDATA_SIZE.
static_assert(sizeof(LogData) == 497, "LogData size changed. Update Python reader offsets/sizes.");

static constexpr std::size_t kLogDataBytes = sizeof(LogData);
static constexpr std::size_t kSlotPadBytes = (8 - (kLogDataBytes % 8)) % 8; // ensure slot stride multiple of 8

// -----------------------------
// MMap header + ring buffer slot
// - Slot must be 8-byte aligned to avoid SIGBUS on atomic u64 access.
// -----------------------------
#pragma pack(push, 1)
struct MMapHeader {
  char     magic[8];      // "STRLOG3\0"
  uint32_t version;       // 3
  uint32_t header_size;   // sizeof(MMapHeader)
  uint32_t capacity;      // number of slots
  uint32_t slot_size;     // sizeof(Slot)
  uint64_t write_count;   // monotonically increasing
  uint64_t start_time_ns; // optional (0 ok)
  uint8_t  reserved[24];  // pad to 64 bytes
};
#pragma pack(pop)

static_assert(sizeof(MMapHeader) == 64, "MMapHeader must be 64 bytes.");

// Slot stride must be multiple of 8 so that every Slot::seq is 8-byte aligned in the ring.
struct alignas(8) Slot {
  uint64_t seq;  // seqlock counter (odd=writing, even=stable)
  LogData  data; // payload
  std::array<uint8_t, kSlotPadBytes> pad{};
};

static_assert(alignof(Slot) == 8, "Slot alignment must be 8.");
static_assert((sizeof(Slot) % 8) == 0, "Slot size must be multiple of 8.");
static_assert(sizeof(Slot) == (8 + kLogDataBytes + kSlotPadBytes), "Slot size mismatch.");

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
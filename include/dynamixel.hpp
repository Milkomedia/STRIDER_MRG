#ifndef DYNAMIXEL_H
#define DYNAMIXEL_H

#include <atomic>
#include <cstdint>
#include <cstddef>

#include <dynamixel_sdk/dynamixel_sdk.h> // Dynamixel SDK (Protocol 2.0)

struct DynamixelFrame {
  uint64_t host_time_ns = 0;     // Host timestamp (steady_clock)

  double q_mea[20] = {0.0};      // Present joint angles [rad]
  double q_cmd[20] = {0.0};      // Last commanded joint angles [rad]

  uint32_t comm_error_count = 0; // Communication error counter (best-effort)
  uint8_t hw_error_any = 0;      // OR of HW error status bytes (best-effort)
};

class Dynamixel {
public:
  using KillFn = void(*)(const char* msg);

  Dynamixel() = default;
  ~Dynamixel(); // stop only; caller must join

  Dynamixel(const Dynamixel&) = delete;
  Dynamixel& operator=(const Dynamixel&) = delete;

  void set_kill_callback(KillFn fn) { kill_fn_.store(fn, std::memory_order_release); }

  void run(); // Thread entry

  // External stop request: thread will freeze once and then exit.
  void request_stop() { stop_request_.store(true, std::memory_order_relaxed); }

  // Write desired joint goals [rad] (lock-free)
  void write_goal(const double q_rad[20]);

  // Read latest snapshot (seqlock)
  bool read_latest(DynamixelFrame& out) const;

  uint64_t get_frame_count() const { return frame_count_.load(std::memory_order_acquire); }

private:
  // ===== Hardcoded config =====
  static constexpr float protocol_ = 2.0f;
  static constexpr int baudrate_ = 4000000;

  // Loop rate target
  static constexpr double loop_hz_ = 60.0;

  // Control table addresses
  static constexpr uint16_t ADDR_OPERATING_MODE        = 11;
  static constexpr uint16_t ADDR_TORQUE_ENABLE         = 64;
  static constexpr uint16_t ADDR_GOAL_POSITION         = 116;
  static constexpr uint16_t ADDR_PRESENT_POSITION      = 132;
  static constexpr uint16_t ADDR_HARDWARE_ERROR_STATUS = 70;

  static constexpr uint16_t ADDR_POSITION_D_GAIN       = 80;
  static constexpr uint16_t ADDR_POSITION_I_GAIN       = 82;
  static constexpr uint16_t ADDR_POSITION_P_GAIN       = 84;
  static constexpr uint16_t ADDR_VELOCITY_I_GAIN       = 76;
  static constexpr uint16_t ADDR_VELOCITY_P_GAIN       = 78;

  static constexpr size_t ARM_NUM = 4;
  static constexpr size_t JOINT_NUM = 5;
  static constexpr size_t DXL_NUM = ARM_NUM * JOINT_NUM;

  // IDs: arm0(1..5), arm1(6..10), arm2(11..15), arm3(16..20)
  inline static constexpr uint8_t DXL_IDS[ARM_NUM][JOINT_NUM] = {
    {1, 2, 3, 4, 5},
    {6, 7, 8, 9, 10},
    {11, 12, 13, 14, 15},
    {16, 17, 18, 19, 20}
  };

  // Joint sign of each joint index
  inline static constexpr int JOINT_SIGN[JOINT_NUM] = {1, -1, -1, 1, 1};

  struct Gains {
    uint16_t pos_P, pos_I, pos_D;
    uint16_t vel_P, vel_I;
  };

  inline static constexpr Gains GAINS[JOINT_NUM] = {
    {  800,    0,    0,  100,  20  }, // j0
    { 2562, 1348,  809, 1079, 3843 }, // j1
    { 2500, 1341, 3843, 1314, 9102 }, // j2
    { 2700,  390,  100, 2023, 2023 }, // j3
    {  700,    0,    0,  100, 1920 }, // j4
  };

  uint8_t goal_param_[DXL_NUM][4] = {{0}};

  // ===== seqlock helper =====
  struct SeqWriteGuard {
    explicit SeqWriteGuard(std::atomic<uint64_t>& s) : seq(s) { seq.fetch_add(1, std::memory_order_acq_rel); }
    ~SeqWriteGuard() { seq.fetch_add(1, std::memory_order_release); }
    std::atomic<uint64_t>& seq;
  };

  // ===== internal helpers =====
  void call_kill_(const char* msg) const;

  bool open_and_init_();
  void close_();

  bool read_all_present_rad_(double out_q_rad[20], const double fallback_q_rad[20], uint32_t& comm_err_cnt);
  bool write_all_goal_rad_(const double q_rad[20], uint32_t& comm_err_cnt);

  // Freeze: set goal = present (best-effort) and publish final snapshot
  void freeze_now_(double q_mea_rad[20], double q_cmd_rad[20], uint32_t& comm_err_cnt, uint8_t hw_error_any);

  // Convert rad <-> ppr
  static inline int rad_2_ppr_(int joint_idx, double rad) {
    if (joint_idx == 0) {
      const double ppr = JOINT_SIGN[joint_idx] * rad * 4074.3665431525 + 2048.0; // (6.25 * 2048 / pi)
      return static_cast<int>(ppr);
    }
    const double ppr = JOINT_SIGN[joint_idx] * rad * 651.8986469044 + 2048.0; // (2048 / pi)
    return static_cast<int>(ppr);
  }

  static inline double ppr_2_rad_(int joint_idx, int ppr) {
    if (joint_idx == 0) {
      return JOINT_SIGN[joint_idx] * (static_cast<double>(ppr) - 2048.0 ) * 0.0002454369260617; // (pi / 6.25 / 2048)
    }
    return JOINT_SIGN[joint_idx] * (static_cast<double>(ppr) - 2048.0) * 0.0015339807878856; // (pi / 2048)
  }

  // ===== SDK handles =====
  dynamixel::PortHandler*    port_ = nullptr;
  dynamixel::PacketHandler*  pkt_ = nullptr;
  dynamixel::GroupSyncWrite* sync_write_ = nullptr; // Desired angle write
  dynamixel::GroupSyncRead*  sync_read_  = nullptr; // Current angle read
  dynamixel::GroupSyncRead*  sync_read_hw_ = nullptr; // HW error status

  // ===== control atomics =====
  std::atomic<bool> stop_request_{false};
  std::atomic<uint64_t> frame_count_{0};
  std::atomic<KillFn> kill_fn_{nullptr};

  // latest state buffer (seqlock)
  mutable std::atomic<uint64_t> seq_{0};
  DynamixelFrame buf_{};

  // desired goal buffer (seqlock)
  struct GoalBuf {
    double q_des[20] = {0.0};
    bool has = false;
  };
  mutable std::atomic<uint64_t> goal_seq_{0};
  GoalBuf goal_buf_{};
};

#endif // DYNAMIXEL_H
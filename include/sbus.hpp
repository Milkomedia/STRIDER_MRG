#ifndef SBUS_H
#define SBUS_H

#include <atomic>
#include <cstdint>

/* Futaba transmitter Mapping
[chn  0] -> position y command
[chn  1] -> position x command
[chn  2] -> position z command
[chn  3] -> yaw angle command (incremental)
[chn  4] -> SA *
[chn  5] -> SH *
[chn  6] -> SD *
[chn  7] -> SG MRG on(1696) / off(352) command
[chn  8] -> SE armed(352) / idle(1024) / rising(1696) command
[chn  9] -> SF kill command
[chn 10] -> LD CoT z command
[chn 11] -> RD inter-rotor distance trimming
[chn 12] -> SC *
[chn 13] -> No assigned (spits 112)
[chn 14] -> No assigned (spits 1024)
[chn 15] -> No assigned (spits 1024)
[chn 16] -> No assigned (same act as chn12)
[chn 17] -> No assigned (same act as chn8)
*/

struct SBUSFrame {
  uint64_t host_time_ns = 0;  // Host timestamp (steady_clock) [ns]
  uint16_t ch[18] = {0};      // ch0..ch15: 0..2047 (11-bit). ch16/17: digital (0 or 2047)
  uint8_t failsafe = 0;       // 0=OK, 1=LOST, 2=FAILSAFE
};

class SBUS {
public:
  using KillFn = void(*)(const char* msg);

  SBUS() = default;
  ~SBUS(); // stop only; caller must join

  SBUS(const SBUS&) = delete;
  SBUS& operator=(const SBUS&) = delete;

  void set_kill_callback(KillFn fn) { kill_fn_.store(fn, std::memory_order_release); }

  void run();          // Thread entry
  void request_stop() { stop_request_.store(true, std::memory_order_relaxed); }

  bool read_latest(SBUSFrame& out) const;
  uint64_t get_frame_count() const { return frame_count_.load(std::memory_order_relaxed); }
  bool is_running() const { return is_running_.load(std::memory_order_acquire); }

private:
  struct SeqWriteGuard {
    std::atomic<uint64_t>& seq;
    explicit SeqWriteGuard(std::atomic<uint64_t>& s) : seq(s) {seq.fetch_add(1, std::memory_order_acq_rel);} // odd
    ~SeqWriteGuard() {seq.fetch_add(1, std::memory_order_release);} // even
  };

  void call_kill(const char* msg) const;

  // Seqlock buffer
  mutable std::atomic<uint64_t> seq_{0};
  SBUSFrame buf_{};

  // Lifecycle
  std::atomic<bool> stop_request_{false};
  std::atomic<bool> is_running_{false};

  // Kill callback
  std::atomic<KillFn> kill_fn_{nullptr};

  // Stats
  std::atomic<uint64_t> frame_count_{0};
  std::atomic<uint64_t> last_host_ns_{0};

  // Safety / timing
  static constexpr int first_frame_timeout_ms_ = 800;
  static constexpr int frame_timeout_ms_      = 100;   // kill if no valid frame for 100ms
  static constexpr int poll_timeout_ms_       = 10;    // serial poll granularity

  // Kill channel rule (0-based index)
  static constexpr int kill_chn_idx_ = 9;       // ch[9]
  static constexpr uint16_t arm_val_ = 352;

  // Lost frame rule
  static constexpr int lost_consecutive_kill_ = 30; // if lost flag persists, kill
};

#endif // SBUS_H
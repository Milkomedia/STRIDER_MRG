#ifndef T265_H
#define T265_H

#include "params.hpp"

#include <atomic>
#include <cstdint>
#include <chrono>
#include <mutex>
#include <condition_variable>

#include <librealsense2/rs.hpp>

struct T265frame {
  uint64_t host_time_ns = 0;   // Host timestamp (steady_clock) [ns]
  double device_time_ms = 0.0; // Device timestamp from librealsense [ms]

  double pos[3]  = {0.0, 0.0, 0.0};      // [m]
  double vel[3]  = {0.0, 0.0, 0.0};      // [m/s]
  double quat[4] = {0.0, 0.0, 0.0, 1.0}; // [x y z w]
  double omega[3]= {0.0, 0.0, 0.0};      // [rad/s]

  uint8_t tracker_conf = 0; // 1,2,3 -> LOW,MEDIUM,HIGH
  uint8_t mapper_conf  = 0; // 1,2,3 -> LOW,MEDIUM,HIGH
};

class T265 {
public:
  using KillFn = void(*)(const char* msg);

  T265() = default;
  ~T265(); // stop only; does NOT join

  void set_kill_callback(KillFn fn);

  void run();          // Thread entry: call from std::thread
  void request_stop(); // Request stop (non-blocking)

  bool read_latest(T265frame& out) const; // Lock-free read

  bool wait_new_frame_until(std::chrono::steady_clock::time_point deadline, const uint64_t& last_frame_count) const;

  uint64_t get_frame_count() const { return frame_count_.load(std::memory_order_relaxed); }
  bool is_running() const { return is_running_.load(std::memory_order_relaxed); }

private:

  // RealSense
  rs2::context  ctx_;
  rs2::pipeline pipe_;

  struct SeqWriteGuard { // Seqlock writer guard (prevents odd-stuck on exceptions inside writer scope)
    std::atomic<uint64_t>& seq;
    // Mark write-in-progress (odd). acq_rel prevents reordering around this boundary.
    explicit SeqWriteGuard(std::atomic<uint64_t>& s) : seq(s) {seq.fetch_add(1, std::memory_order_acq_rel);}
    // Publish (even). release ensures readers see fully written data.
    ~SeqWriteGuard() {seq.fetch_add(1, std::memory_order_release);}
  };

  void call_kill(const char* msg) const;
  bool start_stream_(); // returns true if started and pose stream exists
  void on_frame_(const rs2::frame& f) noexcept;

  // Seqlock buffer
  mutable std::atomic<uint64_t> seq_{0};
  T265frame buf_{};

  // Lifecycle
  std::atomic<bool> stop_request_{false};
  std::atomic<bool> is_running_{false};

  // Kill callback
  std::atomic<KillFn> kill_fn_{nullptr};

  // Stats
  std::atomic<uint64_t> frame_count_{0};

  // Frame event (main control loop wake-up)
  mutable std::mutex frame_mtx_;
  mutable std::condition_variable frame_cv_;

  // Timeouts
  static constexpr int FirstFrameTimeoutMs = 800;
  static constexpr int MonitorSleepMs = 1;
  const double MAX_TIMEOUT_MS = 15.0;

  // For stale detection / monitoring
  std::atomic<uint64_t> last_host_ns_{0};
  std::atomic<uint64_t> last_device_ms_{0};
};

#endif // T265_H
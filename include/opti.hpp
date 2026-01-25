#ifndef OPTI_H
#define OPTI_H

#include <atomic>
#include <cstdint>

struct OptiFrame {
  uint64_t host_time_ns = 0;             // Host timestamp (steady_clock)
  double pos[3]  = {0.0, 0.0, 0.0};      // [m]
  double quat[4] = {0.0, 0.0, 0.0, 1.0}; // [x y z w]
};

namespace libmotioncapture {
class MotionCapture; // forward decl
}

class Opti {
public:
  using KillFn = void(*)(const char*);

  Opti() = default;
  ~Opti();

  Opti(const Opti&) = delete;
  Opti& operator=(const Opti&) = delete;

  void set_kill_callback(KillFn fn) { kill_fn_.store(fn, std::memory_order_release); }

  void run(); // Thread entry
  void request_stop() { stop_request_.store(true, std::memory_order_relaxed); }

  bool read_latest(OptiFrame& out) const;

  uint64_t get_frame_count() const { return frame_count_.load(std::memory_order_relaxed); }
  bool is_running() const { return is_running_.load(std::memory_order_acquire); }

private:
  struct SeqWriteGuard {
    std::atomic<uint64_t>* seq;
    explicit SeqWriteGuard(std::atomic<uint64_t>& s) : seq(&s) { seq->fetch_add(1, std::memory_order_acq_rel); } // odd
    ~SeqWriteGuard() { seq->fetch_add(1, std::memory_order_release); } // even
  };

  void call_kill(const char* msg) const;
  bool start_stream_();
  void stop_stream_();

  std::atomic<bool> stop_request_{false};
  std::atomic<bool> is_running_{false};

  std::atomic<uint64_t> frame_count_{0};
  std::atomic<uint64_t> seq_{0};
  OptiFrame buf_{};

  std::atomic<KillFn> kill_fn_{nullptr};

  std::atomic<uint64_t> last_host_ns_{0}; // "Good" update time (target rigid body found) for stale detection.

  libmotioncapture::MotionCapture* mc_ = nullptr;

  // If the target rigid body is not found for this long, kill.
  static constexpr uint64_t FirstTimeoutNs = 2 * 1e9;
  static constexpr uint64_t TimeoutNs = 10 * 1e6;
};

#endif // OPTI_H
#include "opti.hpp"
#include "utils.hpp"
#include "params.hpp"

#include <chrono>
#include <thread>
#include <string>
#include <string_view>
#include <immintrin.h> // _mm_pause
#include <map>
#include <cstdio>
#include <cctype>

#include <pthread.h>
#include <sched.h>

#include <libmotioncapture/motioncapture.h>

static inline bool is_space(unsigned char c) { return std::isspace(c) != 0; }

static inline std::string_view normalize_view(const std::string& s) {
  std::string_view v(s);

  while (!v.empty() && is_space(static_cast<unsigned char>(v.front()))) v.remove_prefix(1);
  while (!v.empty() && is_space(static_cast<unsigned char>(v.back())))  v.remove_suffix(1);
  while (!v.empty() && v.front() == '/') v.remove_prefix(1);

  return v;
}

static inline std::string normalize_copy(const char* s) {
  std::string out = (s ? s : "");
  // trim space
  while (!out.empty() && is_space(static_cast<unsigned char>(out.front()))) out.erase(out.begin());
  while (!out.empty() && is_space(static_cast<unsigned char>(out.back()))) out.pop_back();
  // strip leading '/'
  while (!out.empty() && out.front() == '/') out.erase(out.begin());
  return out;
}

// Stop only; caller must join the owning std::thread before destroying this object.
Opti::~Opti() {
  request_stop();
}

void Opti::call_kill(const char* msg) const {
  KillFn fn = kill_fn_.load(std::memory_order_acquire);
  if (fn) fn(msg);
}

bool Opti::read_latest(OptiFrame& out) const {
  const uint64_t s0 = seq_.load(std::memory_order_acquire);
  if (s0 == 0) return false;

  for (int spin = 0; spin < 100; ++spin) {
    const uint64_t s1 = seq_.load(std::memory_order_acquire);
    if (s1 & 1ULL) { _mm_pause(); continue; } // writer in progress (odd)

    const OptiFrame tmp = buf_;

    const uint64_t s2 = seq_.load(std::memory_order_acquire);
    if (s1 == s2) { out = tmp; return true; } // consistent snapshot
  }
  return false;
}

bool Opti::start_stream_() {
  if (stop_request_.load(std::memory_order_relaxed)) return false;

  try {
    std::map<std::string, std::string> cfg;
    cfg["hostname"] = param::OPTI_IP;

    libmotioncapture::MotionCapture* mc = libmotioncapture::MotionCapture::connect(param::MOCAP_TYPE, cfg);
    if (!mc) { return false; }

    // Success: take ownership in run thread only.
    mc_ = mc;
    return true;

  }
  catch (const std::exception& e) {
    std::fprintf(stderr, "[Opti] connect exception: %s\n", e.what());
    std::fflush(stderr);
    return false;
  }
  catch (...) {
    std::fprintf(stderr, "[Opti] connect unknown exception\n");
    std::fflush(stderr);
    return false;
  }
  return false;
}

void Opti::stop_stream_() {
  // Must be called only from run thread.
  try {
    delete mc_;
  } catch (...) {}
  mc_ = nullptr;
}

void Opti::run() {
  stop_request_.store(false, std::memory_order_relaxed);
  is_running_.store(true, std::memory_order_release);

  try_pin_cpu(param::CPU_OPTI);
  try_set_prior(param::OPTI_PRIOR);

  const std::string target = normalize_copy(param::RIGIDBODY_NAME);
  const std::string_view target_v = normalize_view(target);

  if (!start_stream_()) {
    call_kill("[Opti] failed to connect motioncapture");
    is_running_.store(false, std::memory_order_release);
    return;
  }

  uint64_t last_warn_ns = 0;

  try {
    while (!stop_request_.load(std::memory_order_relaxed)) {
      // Blocking until next mocap frame
      mc_->waitForNextFrame();

      const uint64_t host_ns = now_steady_ns();

      const auto& rbs = mc_->rigidBodies();
      bool found = false;

      for (const auto& kv : rbs) {
        const auto& rb = kv.second;

        const std::string_view key_v = normalize_view(kv.first);
        const std::string_view name_v = normalize_view(rb.name());

        if (key_v != target_v && name_v != target_v) continue;

        const auto p = rb.position();
        const auto q = rb.rotation();

        {
          SeqWriteGuard g(seq_);
          buf_.host_time_ns = host_ns;
          buf_.pos[0] = static_cast<double>(p.x());
          buf_.pos[1] = static_cast<double>(p.y());
          buf_.pos[2] = static_cast<double>(p.z());
          buf_.quat[0] = static_cast<double>(q.x());
          buf_.quat[1] = static_cast<double>(q.y());
          buf_.quat[2] = static_cast<double>(q.z());
          buf_.quat[3] = static_cast<double>(q.w());
        }

        frame_count_.fetch_add(1, std::memory_order_relaxed);
        last_host_ns_.store(host_ns, std::memory_order_relaxed);

        found = true;
        break;
      }

      if (!found) {
        const uint64_t last_good = last_host_ns_.load(std::memory_order_relaxed);
        if (last_good != 0) {
          const uint64_t dt_ns = host_ns - last_good;
          if (dt_ns > TimeoutNs) {
            call_kill("[Opti] stale: target rigid body not found");
            break;
          }
        }
        else { // first try
          const uint64_t dt_ns = host_ns - last_good;
          if (dt_ns > FirstTimeoutNs) {
            call_kill("[Opti] no strider detected: rigid body not found");
            break;
          }
        }
      }
    }
  }
  catch (const std::exception& e) {call_kill(e.what());}
  catch (...) {call_kill("[Opti] unknown exception");}

  stop_stream_();
  is_running_.store(false, std::memory_order_release);
}

#include "t265.hpp"
#include "utils.hpp"
#include "params.hpp"

#include <thread>
#include <iostream>
#include <string>
#include <immintrin.h> // _mm_pause
#include <pthread.h>
#include <sched.h>
#include <chrono> 

// Stop only; caller must join the owning std::thread before destroying this object.
T265::~T265() {
  request_stop();
}

void T265::set_kill_callback(KillFn fn) {
  kill_fn_.store(fn, std::memory_order_release);
}

void T265::call_kill(const char* msg) const {
  KillFn fn = kill_fn_.load(std::memory_order_acquire);
  if (fn) fn(msg);
}

void T265::request_stop() {
  stop_request_.store(true, std::memory_order_relaxed);
  try { pipe_.stop(); }
  catch (...) {} // ignore
}

bool T265::read_latest(T265frame& out) const {
  const uint64_t s0 = seq_.load(std::memory_order_acquire);
  if (s0 == 0) return false;

  for (int spin = 0; spin < 100; ++spin) {
    const uint64_t s1 = seq_.load(std::memory_order_acquire);
    if (s1 & 1ULL) { _mm_pause(); continue; } // writer in progress (odd)

    const T265frame tmp = buf_;

    const uint64_t s2 = seq_.load(std::memory_order_acquire);
    if (s1 == s2) { out = tmp; return true; } // consistent snapshot
  }
  return false;
}

void T265::on_frame_(const rs2::frame& f) noexcept {
  // Keep callback ultra-light; never throw out of this function.
  try {
    if (stop_request_.load(std::memory_order_relaxed)) return;

    rs2::frame pose_any = f;
    if (auto fs = f.as<rs2::frameset>()) {
      pose_any = fs.first_or_default(RS2_STREAM_POSE);
    }

    if (!pose_any) return;
    if (!pose_any.is<rs2::pose_frame>()) return;

    rs2::pose_frame pf(pose_any);

    const uint64_t host_ns = now_steady_ns();
    const double device_ms = pf.get_timestamp();

    // Stale detection (device timestamp jump / host-time gap)
    const uint64_t prev_host = last_host_ns_.exchange(host_ns, std::memory_order_relaxed);
    const uint64_t prev_dev_u64 =
      last_device_ms_.exchange(static_cast<uint64_t>(device_ms), std::memory_order_relaxed);

    if (prev_dev_u64 != 0 && prev_host != 0) {
      const double prev_dev = static_cast<double>(prev_dev_u64);
      const double dt_host_ms = static_cast<double>(host_ns - prev_host) * 1e-6;
      const double dt_dev_ms  = device_ms - prev_dev;

      if (dt_dev_ms <= 0.0 || dt_dev_ms > MAX_TIMEOUT_MS || dt_host_ms > MAX_TIMEOUT_MS) {
        // Mark stop request and kill (do not call pipe_.stop() inside callback).
        stop_request_.store(true, std::memory_order_relaxed);
        call_kill("[T265] stale detected");
        return;
      }
    }

    const rs2_pose p = pf.get_pose_data();

    T265frame s;
    s.host_time_ns   = host_ns;
    s.device_time_ms = device_ms;
    const auto now_tp = std::chrono::steady_clock::now(); 

    s.pos[0] = static_cast<double>(p.translation.x);
    s.pos[1] = static_cast<double>(p.translation.y);
    s.pos[2] = static_cast<double>(p.translation.z);

    s.vel[0] = static_cast<double>(p.velocity.x);
    s.vel[1] = static_cast<double>(p.velocity.y);
    s.vel[2] = static_cast<double>(p.velocity.z);

    s.quat[0] = static_cast<double>(p.rotation.x);
    s.quat[1] = static_cast<double>(p.rotation.y);
    s.quat[2] = static_cast<double>(p.rotation.z);
    s.quat[3] = static_cast<double>(p.rotation.w);

    const double wx_raw = static_cast<double>(p.angular_velocity.x);
    const double wy_raw = static_cast<double>(p.angular_velocity.y);
    const double wz_raw = static_cast<double>(p.angular_velocity.z);
    s.omega_raw[0] = wx_raw;
    s.omega_raw[1] = wy_raw;
    s.omega_raw[2] = wz_raw;

    // s.omega[0] = gyro_bf_[0].update(wx_raw, now_tp);
    // s.omega[1] = gyro_bf_[1].update(wy_raw, now_tp);
    s.omega[0] = wx_raw;
    s.omega[1] = gyro_bf_[1].update(wy_raw, now_tp);
    s.omega[2] = wz_raw;

    s.tracker_conf = static_cast<uint8_t>(p.tracker_confidence);
    s.mapper_conf  = static_cast<uint8_t>(p.mapper_confidence);

    {
      SeqWriteGuard g(seq_);
      buf_ = s;
    }

    frame_count_.fetch_add(1, std::memory_order_release);
    frame_cv_.notify_one();
  }
  catch (const rs2::error& e) {
    std::cerr << "[T265] callback rs2::error: " << e.what() << "\n";
    stop_request_.store(true, std::memory_order_relaxed);
    call_kill("[T265] callback rs2::error");
  }
  catch (const std::exception& e) {
    std::cerr << "[T265] callback exception: " << e.what() << "\n";
    stop_request_.store(true, std::memory_order_relaxed);
    call_kill("[T265] callback exception");
  }
  catch (...) {
    std::cerr << "[T265] callback unknown exception\n";
    stop_request_.store(true, std::memory_order_relaxed);
    call_kill("[T265] callback unknown exception");
  }
}

bool T265::wait_new_frame_until(std::chrono::steady_clock::time_point deadline, const uint64_t& last_frame_count) const {
  std::unique_lock<std::mutex> lk(frame_mtx_);

  const bool ok = frame_cv_.wait_until(lk, deadline, [&] {return stop_request_.load(std::memory_order_relaxed) || frame_count_.load(std::memory_order_acquire) != last_frame_count;});

  if (!ok) return false;
  if (stop_request_.load(std::memory_order_relaxed)) return false;

  return true;
}

bool T265::start_stream_() {
  std::string serial;
  std::string name;

  try {
    rs2::device_list devs = ctx_.query_devices();

    for (auto&& dev : devs) {
      std::string n = dev.get_info(RS2_CAMERA_INFO_NAME);
      if (n.find("T265") != std::string::npos || n.find("Tracking Camera") != std::string::npos) {
        name = n;
        serial = dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
        break;
      }
    }
  }
  catch (const rs2::error& e) {
    std::cerr << "[T265] query_devices rs2::error: " << e.what() << "\n";
    return false;
  }

  if (serial.empty()) {
    std::cerr << "[T265] no T265 device found\n";
    return false;
  }

  rs2::config cfg;
  cfg.disable_all_streams();
  cfg.enable_device(serial);
  cfg.enable_stream(RS2_STREAM_POSE, 0, 0, 0, RS2_FORMAT_6DOF, 200);

  // Best-effort pipeline reset
  try { pipe_.stop(); } catch (...) {}

  // Reset monitoring state
  last_host_ns_.store(0, std::memory_order_relaxed);
  last_device_ms_.store(0, std::memory_order_relaxed);
  gyro_bf_[0].reset(); gyro_bf_[1].reset(); gyro_bf_[2].reset();

  try {
    rs2::pipeline_profile prof = pipe_.start(cfg, [this](const rs2::frame& f) { on_frame_(f); });

    // Check pose stream exists after start (started check)
    bool has_pose = false;
    for (const auto& sp : prof.get_streams()) {if (sp.stream_type() == RS2_STREAM_POSE) { has_pose = true; break; }}
    if (!has_pose) {
      try { pipe_.stop(); } catch (...) {}
      std::cerr << "[T265] POSE stream missing after start\n";
      return false;
    }

    // Wait for first frame to arrive (started check)
    const uint64_t t_start = now_steady_ns();
    while (!stop_request_.load(std::memory_order_relaxed)) {
      if (frame_count_.load(std::memory_order_relaxed) > 0) {return true;}
      const uint64_t dt_ms = (now_steady_ns() - t_start) / 1000000ULL;
      if (dt_ms > static_cast<uint64_t>(FirstFrameTimeoutMs)) {break;}
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (stop_request_.load(std::memory_order_relaxed)){ std::cerr << "[T265] Stopped during first frame check.\n";}
    else { // No first frame in time
      try { pipe_.stop(); } catch (...) {}
      std::cerr << "[T265] first frame timeout\n";
    }
    return false;
  }
  catch (const rs2::error& e) {
    std::cerr << "[T265] start failed: " << e.what()
              << " | failed_function=" << e.get_failed_function()
              << " | failed_args=" << e.get_failed_args()
              << "\n";
  }
  catch (const std::exception& e) {
    std::cerr << "[T265] start exception: " << e.what() << "\n";
  }

  return false;
}

void T265::run() {
  stop_request_.store(false, std::memory_order_relaxed);
  is_running_.store(true, std::memory_order_relaxed);

  try_pin_cpu(param::CPU_T265);
  try_set_prior(param::T265_PRIOR);

  try {
    const bool started = start_stream_();
    if (!started) {
      call_kill("[T265] start failed (no device / no pose / no first frame)");
      throw std::runtime_error("[T265] start failed");
    }

    // Monitor loop
    while (!stop_request_.load(std::memory_order_relaxed)) {
      const uint64_t host_ns = now_steady_ns();
      const uint64_t last_ns = last_host_ns_.load(std::memory_order_relaxed);

      if (last_ns != 0) {
        const double gap_ms = static_cast<double>(host_ns - last_ns) * 1e-6;
        if (gap_ms > MAX_TIMEOUT_MS) {
          call_kill("[T265] monitor timeout (no callback update)");
          throw std::runtime_error("[T265] monitor timeout");
        }
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(MonitorSleepMs));
    }
  }
  catch (const rs2::error& e) {
    std::cerr << "[T265] RealSense error: " << e.what() << "\n";
    call_kill("[T265] rs2::error");
  }
  catch (const std::exception& e) {
    std::cerr << "[T265] Exception: " << e.what() << "\n";
    call_kill("[T265] exception");
  }
  catch (...) {
    std::cerr << "[T265] Unknown exception\n";
    call_kill("[T265] unknown exception");
  }

  // Best-effort stop
  try { pipe_.stop(); } catch (...) {}

  is_running_.store(false, std::memory_order_relaxed);
}
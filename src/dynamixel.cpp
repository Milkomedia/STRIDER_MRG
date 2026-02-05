#include "dynamixel.hpp"
#include "utils.hpp"
#include "params.hpp"

#include <chrono>
#include <thread>
#include <cstdio>
#include <cstring>
#include <new>

#include <pthread.h>
#include <sched.h>

// Stop only; caller must join the owning std::thread before destroying this object.
Dynamixel::~Dynamixel() {
  request_stop();
}

void Dynamixel::call_kill_(const char* msg) const {
  KillFn fn = kill_fn_.load(std::memory_order_acquire);
  if (fn) fn(msg);
}

void Dynamixel::write_goal(const double q_rad[20]) {
  SeqWriteGuard g(goal_seq_);
  std::memcpy(goal_buf_.q_des, q_rad, sizeof(goal_buf_.q_des));
  goal_buf_.has = true;
}

bool Dynamixel::read_latest(DynamixelFrame& out) const {
  const uint64_t s0 = seq_.load(std::memory_order_acquire);
  if (s0 == 0) {return false;}

  for (int i = 0; i < 100; ++i) {
    const uint64_t s1 = seq_.load(std::memory_order_acquire);
    if (s1 & 1ULL) {continue;}  // writer in progress

    DynamixelFrame tmp = buf_;
    
    const uint64_t s2 = seq_.load(std::memory_order_acquire);
    if (s1 == s2) { out = tmp; return true; }  // consistent snapshot
  }
  return false;
}

bool Dynamixel::open_and_init_() {
  port_ = dynamixel::PortHandler::getPortHandler(param::DXL_PORT_NAME);
  pkt_  = dynamixel::PacketHandler::getPacketHandler(protocol_);
  if (!port_ || !pkt_) {return false;}

  if (!port_->openPort()) {return false;}
  if (!port_->setBaudRate(baudrate_)) {return false;}

  uint8_t dxl_error = 0;

  // Configure motors (operating mode, torque enable, gains)
  for (size_t ai = 0; ai < ARM_NUM; ++ai) {
    for (size_t ji = 0; ji < JOINT_NUM; ++ji) {
      const uint8_t id = DXL_IDS[ai][ji];

      const uint8_t mode = (ji == 0) ? 4 : 3; // j0: Extended Position, others: Position
      if (pkt_->write1ByteTxRx(port_, id, ADDR_OPERATING_MODE, mode, &dxl_error) != COMM_SUCCESS) {return false;}
      if (pkt_->write1ByteTxRx(port_, id, ADDR_TORQUE_ENABLE, 1, &dxl_error) != COMM_SUCCESS) {return false;}

      const Gains& g = GAINS[ji];
      if (pkt_->write2ByteTxRx(port_, id, ADDR_POSITION_P_GAIN, g.pos_P, &dxl_error) != COMM_SUCCESS) {return false;}
      if (pkt_->write2ByteTxRx(port_, id, ADDR_POSITION_I_GAIN, g.pos_I, &dxl_error) != COMM_SUCCESS) {return false;}
      if (pkt_->write2ByteTxRx(port_, id, ADDR_POSITION_D_GAIN, g.pos_D, &dxl_error) != COMM_SUCCESS) {return false;}
      if (pkt_->write2ByteTxRx(port_, id, ADDR_VELOCITY_P_GAIN, g.vel_P, &dxl_error) != COMM_SUCCESS) {return false;}
      if (pkt_->write2ByteTxRx(port_, id, ADDR_VELOCITY_I_GAIN, g.vel_I, &dxl_error) != COMM_SUCCESS) {return false;}
    }
  }

  sync_write_ = new (std::nothrow) dynamixel::GroupSyncWrite(port_, pkt_, ADDR_GOAL_POSITION, 4);
  sync_read_  = new (std::nothrow) dynamixel::GroupSyncRead (port_, pkt_, ADDR_PRESENT_POSITION, 4);
  sync_read_hw_ = new (std::nothrow) dynamixel::GroupSyncRead(port_, pkt_, ADDR_HARDWARE_ERROR_STATUS, 1);
  if (!sync_write_ || !sync_read_ || !sync_read_hw_) {return false;}

  // Pre-register IDs once for sync read
  sync_write_->clearParam();
  sync_read_->clearParam();
  sync_read_hw_->clearParam();

  size_t k = 0;
  for (size_t ai = 0; ai < ARM_NUM; ++ai) {
    for (size_t ji = 0; ji < JOINT_NUM; ++ji) {
      const uint8_t id = DXL_IDS[ai][ji];
      // init goal buffer to zeros; will be updated via changeParam before txPacket
      goal_param_[k][0] = 0; goal_param_[k][1] = 0; goal_param_[k][2] = 0; goal_param_[k][3] = 0;
      if (!sync_write_->addParam(id, goal_param_[k])) {return false;}
      if (!sync_read_->addParam(id)) {return false;}
      if (!sync_read_hw_->addParam(id)) {return false;}
      ++k;
    }
  }

  return true;
}

void Dynamixel::close_() {
  if (sync_write_) { delete sync_write_; sync_write_ = nullptr; }
  if (sync_read_)  { delete sync_read_;  sync_read_  = nullptr; }
  if (sync_read_hw_) { delete sync_read_hw_; sync_read_hw_ = nullptr; }
  if (port_) {
    try { port_->closePort(); } catch (...) {}
    port_ = nullptr;
  }
  pkt_ = nullptr;
}

bool Dynamixel::read_all_present_rad_(double out_q_rad[20], const double fallback_q_rad[20], uint32_t& comm_err_cnt) {
  if (!sync_read_) {return false;}

  const int comm = sync_read_->txRxPacket();
  if (comm != COMM_SUCCESS) {
    ++comm_err_cnt;
    return false;
  }

  size_t k = 0;
  for (size_t ai = 0; ai < ARM_NUM; ++ai) {
    for (size_t ji = 0; ji < JOINT_NUM; ++ji) {
      const uint8_t id = DXL_IDS[ai][ji];
      if (!sync_read_->isAvailable(id, ADDR_PRESENT_POSITION, 4)) {
        ++comm_err_cnt;
        out_q_rad[k] = fallback_q_rad[k];
        ++k;
        continue;
      }
      const int32_t ppr = static_cast<int32_t>(sync_read_->getData(id, ADDR_PRESENT_POSITION, 4));
      out_q_rad[k++] = ppr_2_rad_(static_cast<int>(ji), static_cast<int>(ppr));
    }
  }
  return true;
}

bool Dynamixel::write_all_goal_rad_(const double q_rad[20], uint32_t& comm_err_cnt) {
  if (!sync_write_) {return false;}

  size_t k = 0;
  for (size_t ai = 0; ai < ARM_NUM; ++ai) {
    for (size_t ji = 0; ji < JOINT_NUM; ++ji) {
      const uint8_t id = DXL_IDS[ai][ji];
      const int32_t ppr_goal = static_cast<int32_t>(rad_2_ppr_(static_cast<int>(ji), q_rad[k]));

      // Update persistent buffer (little endian)
      goal_param_[k][0] = DXL_LOBYTE(DXL_LOWORD(ppr_goal));
      goal_param_[k][1] = DXL_HIBYTE(DXL_LOWORD(ppr_goal));
      goal_param_[k][2] = DXL_LOBYTE(DXL_HIWORD(ppr_goal));
      goal_param_[k][3] = DXL_HIBYTE(DXL_HIWORD(ppr_goal));

      // Update existing param without rebuilding the list
      if (!sync_write_->changeParam(id, goal_param_[k])) {
        ++comm_err_cnt;
        return false;
      }
      ++k;
    }
  }

  const int comm = sync_write_->txPacket();
  if (comm != COMM_SUCCESS) {
    ++comm_err_cnt;
    return false;
  }
  return true;
}

void Dynamixel::freeze_now_(double q_mea_rad[20], double q_cmd_rad[20], uint32_t& comm_err_cnt, uint8_t hw_error_any) {
  // Freeze: goal <- present (best-effort). Torque stays enabled.
  if (!sync_write_) return;

  double cur[20];
  std::memcpy(cur, q_mea_rad, sizeof(cur));

  if (sync_read_) {
    if (read_all_present_rad_(cur, q_mea_rad, comm_err_cnt)) {
      std::memcpy(q_mea_rad, cur, sizeof(cur));
      std::memcpy(q_cmd_rad, cur, sizeof(cur));
    }
  }

  (void)write_all_goal_rad_(q_cmd_rad, comm_err_cnt);

  // Publish final snapshot
  {
    SeqWriteGuard g(seq_);
    buf_.host_time_ns = now_steady_ns();
    std::memcpy(buf_.q_mea, q_mea_rad, sizeof(buf_.q_mea));
    std::memcpy(buf_.q_cmd, q_cmd_rad, sizeof(buf_.q_cmd));
    buf_.comm_error_count = comm_err_cnt;
    buf_.hw_error_any = hw_error_any;
  }
  frame_count_.fetch_add(1, std::memory_order_relaxed);
}

void Dynamixel::run() {
  stop_request_.store(false, std::memory_order_relaxed);

  try_pin_cpu(param::CPU_DXL);
  try_set_prior(param::DXL_PRIOR);

  uint32_t comm_err_cnt = 0;
  uint8_t hw_error_any = 0;

  double q_mea_rad[20] = {0.0};
  double q_cmd_rad[20] = {0.0};

  bool need_kill = false;
  const char* kill_msg = nullptr;
  bool did_freeze = false;

  try {
    // Retry init
    bool ok = false;
    if (open_and_init_()) {ok = true;}
    else {close_();}

    if (!ok) { // Startup fatal: no freeze possible, kill.
      call_kill_("[DXL] open/init failed");
      close_();
      return;
    }

    // Initial alignment: present -> cmd=present -> hold
    double init_q[20] = {0.0};
    if (!read_all_present_rad_(init_q, init_q, comm_err_cnt)) {
      call_kill_("[DXL] initial present-position read failed");
      close_();
      return;
    }

    std::memcpy(q_mea_rad, init_q, sizeof(q_mea_rad));
    std::memcpy(q_cmd_rad, init_q, sizeof(q_cmd_rad));
    (void)write_all_goal_rad_(q_cmd_rad, comm_err_cnt);

    if (comm_err_cnt > 1u) {
      call_kill_("[DXL] startup comm_error_count > 1");
      close_();
      return;
    }

    { // Publish initial snapshot
      SeqWriteGuard g(seq_);
      buf_.host_time_ns = now_steady_ns();
      std::memcpy(buf_.q_mea, q_mea_rad, sizeof(buf_.q_mea));
      std::memcpy(buf_.q_cmd, q_cmd_rad, sizeof(buf_.q_cmd));
      buf_.comm_error_count = comm_err_cnt;
      buf_.hw_error_any = 0;
    }
    frame_count_.fetch_add(1, std::memory_order_relaxed);

    const std::chrono::duration<double> period(1.0 / loop_hz_);
    std::chrono::steady_clock::time_point next_tick = std::chrono::steady_clock::now();

    int consec_wr_fail = 0;
    int consec_rd_fail = 0;

    // Hz monitor
    static constexpr uint32_t kHzWindow = 10;
    static constexpr double   Min_Hz = 45;
    uint32_t win_count = 0;
    uint64_t win_t0_ns = now_steady_ns();
    uint32_t win_late = 0;
    const std::chrono::steady_clock::duration max_late = std::chrono::duration_cast<std::chrono::steady_clock::duration>(period * 1.5);

    uint64_t loop_count = 0;

    while (!stop_request_.load(std::memory_order_relaxed)) {
      // ---- goal fetch (seqlock) ----
      GoalBuf goal_local;
      {
        bool got = false;
        for (int i = 0; i < 50; ++i) {
          const uint64_t s0 = goal_seq_.load(std::memory_order_acquire);
          if (s0 & 1ULL) continue;
          goal_local = goal_buf_;
          const uint64_t s1 = goal_seq_.load(std::memory_order_acquire);
          if (s0 == s1) { got = true; break; }
        }
        if (!got) goal_local.has = false;
      }

      // Command update
      if (goal_local.has) { std::memcpy(q_cmd_rad, goal_local.q_des, sizeof(q_cmd_rad)); }

      // ---- write ----
      if (!write_all_goal_rad_(q_cmd_rad, comm_err_cnt)) { ++consec_wr_fail; }
      else { consec_wr_fail = 0; }

      // ---- read ----
      double cur_q[20];
      if (!read_all_present_rad_(cur_q, q_mea_rad, comm_err_cnt)) { ++consec_rd_fail; }
      else {
        consec_rd_fail = 0;
        std::memcpy(q_mea_rad, cur_q, sizeof(q_mea_rad));
      }

      // Repeated comm failures => freeze & kill
      if (consec_wr_fail >= 3) { need_kill = true; kill_msg = "[DXL] repeated SyncWrite failure"; break; }
      if (consec_rd_fail >= 3) { need_kill = true; kill_msg = "[DXL] repeated SyncRead failure";  break; }

      // HW error poll (low rate)
      if ((loop_count++ & 0x3FULL) == 0ULL) {
        uint8_t any = 0;
        if (!sync_read_hw_) {++comm_err_cnt;}
        else {
          const int comm = sync_read_hw_->txRxPacket(); // ONE packet for all registered IDs
          if (comm != COMM_SUCCESS) {++comm_err_cnt;}
          else {
            for (size_t ai = 0; ai < ARM_NUM; ++ai) {
              for (size_t ji = 0; ji < JOINT_NUM; ++ji) {
                const uint8_t id = DXL_IDS[ai][ji];
                if (!sync_read_hw_->isAvailable(id, ADDR_HARDWARE_ERROR_STATUS, 1)) {
                  ++comm_err_cnt;
                  continue;
                }
                const uint8_t hw = static_cast<uint8_t>(sync_read_hw_->getData(id, ADDR_HARDWARE_ERROR_STATUS, 1));
                any = static_cast<uint8_t>(any | hw);
              }
            }
          }
        }
        hw_error_any = any;

        if (hw_error_any != 0) {
          std::fprintf(stderr, "\n[DXL] hardware error detected (OR=0x%02X)\n", (unsigned)hw_error_any);
          need_kill = true;
          kill_msg = "[DXL] hardware error";
          break;
        }
      }

      // Publish snapshot
      {
        SeqWriteGuard g(seq_);
        buf_.host_time_ns = now_steady_ns();
        std::memcpy(buf_.q_mea, q_mea_rad, sizeof(buf_.q_mea));
        std::memcpy(buf_.q_cmd, q_cmd_rad, sizeof(buf_.q_cmd));
        buf_.comm_error_count = comm_err_cnt;
        buf_.hw_error_any = hw_error_any;
      }
      frame_count_.fetch_add(1, std::memory_order_relaxed);

      // ---- timing ----
      next_tick = next_tick + std::chrono::duration_cast<std::chrono::steady_clock::duration>(period);
      const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();

      if (now > next_tick + max_late) {
        ++win_late;
        next_tick = now; // resync to avoid lag accumulation
      }

      if (now < next_tick) std::this_thread::sleep_until(next_tick);

      // ---- Hz monitoring window ----
      ++win_count;
      if (win_count >= kHzWindow) {
        const uint64_t t1 = now_steady_ns();
        const double dt = static_cast<double>(t1 - win_t0_ns) * 1e-9;
        const double hz = (dt > 0.0) ? (static_cast<double>(win_count) / dt) : 0.0;

        if (hz < Min_Hz) {
          std::fprintf(stderr, "\n[DXL] loop hz drop: %.1f < %.1f (target %.1f)\n", hz, Min_Hz, loop_hz_);
          need_kill = true;
          kill_msg = "[DXL] loop hz drop";
          break;
        }
        
        if (win_late > 2) {
          std::fprintf(stderr, "\n[DXL] too many late cycles: %u/%u\n", win_late, kHzWindow);
          need_kill = true;
          kill_msg = "[DXL] realtime degrade";
          break;
        }

        win_count = 0;
        win_late = 0;
        win_t0_ns = t1;
      }
    }

    // Normal exit (stop_request or break): always freeze once.
    freeze_now_(q_mea_rad, q_cmd_rad, comm_err_cnt, hw_error_any);
    did_freeze = true;

    if (need_kill && kill_msg) call_kill_(kill_msg);
  }
  catch (const std::exception& e) {
    std::fprintf(stderr, "\n[DXL] exception: %s\n", e.what());
    if (!did_freeze) freeze_now_(q_mea_rad, q_cmd_rad, comm_err_cnt, hw_error_any);
    call_kill_("[DXL] exception");
  }
  catch (...) {
    std::fprintf(stderr, "\n[DXL] unknown exception\n");
    if (!did_freeze) freeze_now_(q_mea_rad, q_cmd_rad, comm_err_cnt, hw_error_any);
    call_kill_("[DXL] unknown exception");
  }

  close_();
}
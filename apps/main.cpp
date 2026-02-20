#include "mmap_logger.hpp"
#include "mpc_wrapper.hpp"
#include "fdcl_control.hpp"
#include "teensy.hpp"
#include "dynamixel.hpp"
#include "sbus.hpp"
#include "t265.hpp"
#include "opti.hpp"
#include "utils.hpp"

#include <algorithm>  // std::clamp
#include <thread>
#include <immintrin.h> // _mm_pause
#include <condition_variable>
#include <csignal>
#include <pybind11/embed.h>
#include <cstdio>

static std::atomic<bool> g_killed{false};
static void sigint_handler(int) {g_killed.store(true);}

// Kill must be idempotent (Called from callback thread).
static void KILL(const char* msg) {
  if (g_killed.exchange(true, std::memory_order_relaxed)) return;
  std::fprintf(stderr, "\n\n [KILL!] -> %s\n\n", (msg ? msg : "(null)"));
  g_killed.store(true, std::memory_order_relaxed);
}

static std::mutex mpc_mtx;
static std::condition_variable mpc_cv;
static strider_mpc::MPCInput g_mpc_input;
static strider_mpc::MPCOutput g_mpc_output;
static std::atomic<bool> g_mpc_activated{false};
static std::atomic<uint32_t> g_mpc_epoch{1};
static bool g_mpc_busy = false;

static inline void mpc_reset_locked(uint32_t& mpc_key) {
  g_mpc_epoch.fetch_add(1, std::memory_order_relaxed);
  g_mpc_input.has = false;
  g_mpc_output.has = false;
  g_mpc_busy = false;
  mpc_key += 1; // force key++
}

int main() {
  try_pin_cpu(param::CPU_MAIN);
  std::signal(SIGINT, sigint_handler); // SIGINT handler(ctrl+C)

  // --- CAN first check before start ---
  Teensy teensy;
  teensy.open();
  if (!teensy.check()) {
    std::fprintf(stdout, "\n\n [ CAN DOES NOT STARTED. >> ABORT << ]\n\n"); std::fflush(stdout);
    teensy.close();
    return 1;
  }

  // -------------- [ 1. MPC thread ] -----------------
  std::thread th_mpc([&]() {
    try_pin_cpu(param::CPU_MPC);
    try_set_prior(param::MPC_PRIOR);
    pybind11::scoped_interpreter py_guard{false};
    strider_mpc::acados_wrapper mpc; // compile acados before start.
    std::fprintf(stdout, " ||----------------------------------||\n ||       Acados compile done.       ||\n ||     You can turn on the MRG.     ||\n ||----------------------------------||\n\n");  std::fflush(stdout);

    while (!g_killed.load()) {
      strider_mpc::MPCInput in_local;
      {
        std::unique_lock<std::mutex> lk(mpc_mtx);
        mpc_cv.wait(lk, [&]{
          if (g_killed.load()) return true;
          if (!g_mpc_activated.load(std::memory_order_relaxed)) return false;
          return g_mpc_input.has;
        });
        if (g_killed.load()) break;
        if (g_mpc_input.has == true) {
          in_local = g_mpc_input;
          g_mpc_input.has = false;
          g_mpc_busy = true;
        }
      }

      strider_mpc::MPCOutput out_local;
      try { out_local = mpc.compute(in_local); }
      catch (const std::exception&) { out_local.solve_ms = 0.0; out_local.state = 99; std::printf("OH SHIT");}

      {
        std::lock_guard<std::mutex> lk(mpc_mtx);
        g_mpc_output = out_local;
        g_mpc_output.t = in_local.t;
        g_mpc_output.key = in_local.key;
        g_mpc_output.epoch = in_local.epoch;
        g_mpc_output.has = true;
        g_mpc_busy = false;
      }
    }
  });

  // -------------- [ 2. T265 thread ] ----------------
  T265 t265;
  t265.set_kill_callback(&KILL);
  std::thread th_t265(&T265::run, &t265);
  uint64_t last_t265_cnt = 0;
  T265frame t265_frame{};

  // -------------- [ 3. Opti thread ] ----------------
  Opti opti;
  opti.set_kill_callback(&KILL);
  std::thread th_opti(&Opti::run, &opti);
  uint64_t last_opti_cnt = 0;
  OptiFrame opti_frame{};

  // -------------- [ 4. SBUS thread ] ----------------
  SBUS sbus;
  sbus.set_kill_callback(KILL);
  std::thread th_sbus([&]() {sbus.run();});
  uint64_t last_sbus_cnt = 0;
  SBUSFrame sbus_frame{};

  // -------------- [ 5. Dxl thread ] -----------------
  Dynamixel dxl;
  dxl.set_kill_callback(&KILL);
  std::thread th_dxl(&Dynamixel::run, &dxl);
  uint64_t last_dxl_cnt = 0;
  DynamixelFrame dxl_frame{};

  // -------------------------------------------------- //
  //                 [ 0. Main thread ]                 //
  // -------------------------------------------------- //

  // --- geometry SO3 controller ---
  fdcl::state_t    gac_state;
  fdcl::command_t  gac_cmd;
  fdcl::state_t*   gac_state_ptr = &gac_state;
  fdcl::command_t* gac_cmd_ptr   = &gac_cmd;
  fdcl::control gac(gac_state_ptr, gac_cmd_ptr);

  // --- sensor measurement filters ---
  Butter opti_vel_bf[3] = {Butter(param::OPTI_VEL_CUTOFF_HZ), Butter(param::OPTI_VEL_CUTOFF_HZ), Butter(param::OPTI_VEL_CUTOFF_HZ)};
  Butter gyro_z_bf = Butter(param::GYRO_Z_CUTOFF_HZ);

  // --- other parameters ---
  Phase   phase = Phase::READY;
  State   s{};
  Command cmd{};
  cmd.r1 = param::r1_init;
  cmd.r2 = param::r2_init;
  cmd.r3 = param::r3_init;
  cmd.r4 = param::r4_init;
  Eigen::Vector3d prev_omega = Eigen::Vector3d::Zero();
  uint64_t prev_omega_ns = 0;
  double rising_coeff = param::INITIAL_RISING_COEFF;

  // --- MRG parameters ---
  bool mpc_in_solving = false;
  uint32_t mpc_key = 1;
  strider_mpc::MPCOutput l_mpc_output;
  l_mpc_output.u_rate.setZero();
  l_mpc_output.u_opt.setZero();
  l_mpc_output.t = std::chrono::steady_clock::time_point::max();
  
  { // --- Wait for all sensors to start ---
    std::fprintf(stdout, "\n\n\n\n ||------------------------------------||\n ||--Sensor&Connected device checking--||\n ||------[     T265 -> "); std::fflush(stdout);
    while (!g_killed.load(std::memory_order_relaxed) && !t265.read_latest(t265_frame)) {_mm_pause();}
    if (g_killed.load(std::memory_order_relaxed)) {std::fprintf(stdout, "\n\n T265 ABORT.\n\n"); std::fflush(stdout); g_killed.store(true, std::memory_order_relaxed);}
    else {std::fprintf(stdout, "Good     ]------||\n ||------[     OPTI -> "); std::fflush(stdout);}

    while (!g_killed.load(std::memory_order_relaxed) && !opti.read_latest(opti_frame)) {_mm_pause();}
    if (g_killed.load(std::memory_order_relaxed)) {std::fprintf(stdout, "\n\n OPTI ABORT.\n\n"); std::fflush(stdout); g_killed.store(true, std::memory_order_relaxed);}
    else {std::fprintf(stdout, "Good     ]------||\n ||------[     SBUS -> "); std::fflush(stdout);}

    while (!g_killed.load(std::memory_order_relaxed) && !sbus.read_latest(sbus_frame)) {_mm_pause();}
    if (g_killed.load(std::memory_order_relaxed)) {std::fprintf(stdout, "\n\n SBUS ABORT.\n\n"); std::fflush(stdout); g_killed.store(true, std::memory_order_relaxed);}
    else {std::fprintf(stdout, "Good     ]------||\n ||------[  DYNAMIXEL -> "); std::fflush(stdout);}

    while (!g_killed.load(std::memory_order_relaxed) && !dxl.read_latest(dxl_frame)) {_mm_pause();}
    if (g_killed.load(std::memory_order_relaxed)) {std::fprintf(stdout, "\n\n DXL ABORT.\n\n"); std::fflush(stdout); g_killed.store(true, std::memory_order_relaxed);}
    else {std::fprintf(stdout, "Good   ]------||\n ||------------------------------------||\n\n\n"); std::fflush(stdout);}

    // SBUS toggle check.
    if (sbus_frame.ch[7] != 1024 || sbus_frame.ch[8] != 352) {std::fprintf(stdout, "\n\n SBUS toggle set to wrong.\n Check the transmitter -> ABORT.\n\n"); std::fflush(stdout); g_killed.store(true, std::memory_order_relaxed);}
  
    if (!g_killed.load()) {std::fprintf(stdout, " STRIDER: \"LET'S ROLL.\"\n\n"); std::fflush(stdout);}

    phase = Phase::ARMED;
  }

  // set priority
  try_set_prior(param::MAIN_PRIOR);

  // --- logging ---
  mmap_logger::MMapLogger logger("/tmp/strider_log.mmap", /*reset=*/true);
  const std::chrono::steady_clock::time_point initial_time = std::chrono::steady_clock::now();
  logger.open();

  std::string path = std::string("/home/strider/Desktop/STRIDER_MRG/apps/bag_bin/") + param::Log_File_NAME;
  mmap_logger::log_fp = std::fopen(path.c_str(), "wb");
  setvbuf(mmap_logger::log_fp, nullptr, _IOFBF, 1<<20);
  
  // --- time scope definition ---
  std::chrono::steady_clock::time_point next_control_tick = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point next_mpc_tick     = std::chrono::steady_clock::now();

  while (!g_killed.load()) {
    // measure loop-start tick
    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();

    // IMU read
    const uint64_t cur_t265_cnt = t265.get_frame_count();
    if (cur_t265_cnt > last_t265_cnt) {
      if (t265.read_latest(t265_frame)) {
        s.R = quat_to_R(t265_frame.quat[0], t265_frame.quat[1], t265_frame.quat[2], t265_frame.quat[3]);

        // gyro frame transformation
        s.omega(0) = -t265_frame.omega[2]; s.omega(1) =  t265_frame.omega[0]; s.omega(2) = gyro_z_bf.update(-t265_frame.omega[1], t265_frame.host_time_ns);

        s.alpha = diff(s.omega, prev_omega, t265_frame.host_time_ns, prev_omega_ns); prev_omega = s.omega; prev_omega_ns = t265_frame.host_time_ns;

        last_t265_cnt = cur_t265_cnt;
      }
    }

    // MOCAP read
    const uint64_t cur_opti_cnt = opti.get_frame_count();
    if (cur_opti_cnt > last_opti_cnt) {
      const uint64_t prev_time_ns = opti_frame.host_time_ns;
      const Eigen::Vector3d prev_pos = s.pos;
      const Eigen::Vector3d prev_vel = s.vel;
      if (opti.read_latest(opti_frame)) {
        // apply frame transformation (opti uses z-up convention)
        s.pos(0)=opti_frame.pos[0]-param::OPTI_X_OFFSET; s.pos(1)=-opti_frame.pos[1]+param::OPTI_Y_OFFSET; s.pos(2)=-opti_frame.pos[2];

        const Eigen::Vector3d vel_raw = diff(s.pos, prev_pos, opti_frame.host_time_ns, prev_time_ns);
        s.vel(0) = opti_vel_bf[0].update(vel_raw(0), opti_frame.host_time_ns); s.vel(1) = opti_vel_bf[1].update(vel_raw(1), opti_frame.host_time_ns); s.vel(2) = opti_vel_bf[2].update(vel_raw(2), opti_frame.host_time_ns);

        s.acc = diff(s.vel, prev_vel, opti_frame.host_time_ns, prev_time_ns);

        last_opti_cnt = cur_opti_cnt;
      }
    }

    // SBUS read
    const uint64_t cur_sbus_cnt = sbus.get_frame_count();
    if (cur_sbus_cnt > last_sbus_cnt) {
      if (sbus.read_latest(sbus_frame)) {
        cmd.pos      = sbus_pos_map(sbus_frame.ch[0], sbus_frame.ch[1], sbus_frame.ch[2]);
        cmd.heading  = sbus_yaw_map(cmd.yaw, sbus_frame.ch[3]); // cmd yaw & cmd heading are updated simultaneously
        const double z = sbus_cotz_map(sbus_frame.ch[10]);
        cmd.r1(2) = z; cmd.r2(2) = z; cmd.r3(2) = z; cmd.r4(2) = z;

        if (phase == Phase::GAC_FLIGHT) {
          if (sbus_frame.ch[7] == 1696) {
            g_mpc_activated.store(true, std::memory_order_relaxed);
            mpc_reset_locked(mpc_key);
            phase = Phase::MRG_YES_COT;
            std::fprintf(stdout, "flight state -> [MRG_YES_COT]\n"); std::fflush(stdout);
          }
          if (sbus_frame.ch[7] == 352) {
            g_mpc_activated.store(true, std::memory_order_relaxed);
            mpc_reset_locked(mpc_key);
            phase = Phase::MRG_NO_COT;
            std::fprintf(stdout, "flight state -> [MRG_NO_COT]\n"); std::fflush(stdout);
          }
        }
        else if (phase == Phase::MRG_NO_COT) {
          if (sbus_frame.ch[7] == 1024) {
            cmd.d_theta.setZero();         // reset previous optimal u
            l_mpc_output.u_rate.setZero(); // reset previous optimal u
            g_mpc_activated.store(false, std::memory_order_relaxed);
            phase = Phase::GAC_FLIGHT;
            std::fprintf(stdout, "flight state -> [GAC_FLIGHT]\n"); std::fflush(stdout);
          }
        }
        else if (phase == Phase::MRG_YES_COT) {
          if (sbus_frame.ch[7] == 1024) {
            cmd.d_theta.setZero();         // reset previous optimal u
            l_mpc_output.u_rate.setZero(); // reset previous optimal u
            g_mpc_activated.store(false, std::memory_order_relaxed);
            phase = Phase::GAC_FLIGHT;
            std::fprintf(stdout, "flight state -> [GAC_FLIGHT]\n"); std::fflush(stdout);
          }
        }
        else if (phase == Phase::ARMED) {
          if (sbus_frame.ch[7] != 1024) {std::fprintf(stdout, "\n\n You cannot run MPC before [GAC_FLIGHT] phase. -> ABORT.\n\n"); std::fflush(stdout); g_killed.store(true, std::memory_order_relaxed);}
          else {
            if (sbus_frame.ch[8] == 1024) {phase = Phase::IDLE;std::fprintf(stdout, "flight state -> [IDLE]\n"); std::fflush(stdout);}
          }
        }
        else if (phase == Phase::IDLE) {
          if (sbus_frame.ch[7] != 1024) {std::fprintf(stdout, "\n\n You cannot run MPC before [GAC_FLIGHT] phase. -> ABORT.\n\n"); std::fflush(stdout); g_killed.store(true, std::memory_order_relaxed);}
          else {
            if (sbus_frame.ch[8] != 1024) {
              if(sbus_frame.ch[8] == 1360) {phase = Phase::RISING;std::fprintf(stdout, "flight state -> [RISING]\n"); std::fflush(stdout);}
              else {std::fprintf(stdout, "\n\n You cannot turn back to [ARMED] phase in [IDLE] phase. -> ABORT.\n\n"); std::fflush(stdout); g_killed.store(true, std::memory_order_relaxed);}
            }
          }
        }

        last_sbus_cnt = cur_sbus_cnt;
      }
    }

    // Dynamixel read
    const uint64_t cur_dxl_cnt = dxl.get_frame_count();
    if (cur_dxl_cnt > last_dxl_cnt) {
      if (dxl.read_latest(dxl_frame)) {
        for(uint8_t i = 0; i < 20; ++i) {s.arm_q[i] = dxl_frame.q_mea[i];}
        FK(s.arm_q, s.r_cot, s.r1, s.r2, s.r3, s.r4);
        s.r_com(0) = param::COM_OFF_X + param::COT_2_COM_X * s.r_cot(0);
        s.r_com(1) = param::COM_OFF_Y + param::COT_2_COM_Y * s.r_cot(1);
        last_dxl_cnt = cur_dxl_cnt;
      }
    }

    // ==== POSITION CONTROL ====
    gac_cmd.xd  = cmd.pos;
    gac_cmd.xd_dot = cmd.vel;
    gac_cmd.xd_2dot = cmd.acc;
    gac_cmd.b1d = cmd.heading;
    gac_state.x = s.pos;
    gac_state.v = s.vel;
    gac_state.a = s.acc;
    gac_state.R = s.R;
    gac_state.W = s.omega;
    gac.position_control();
    const Eigen::Matrix3d R_raw = gac_cmd.Rd;
    const Eigen::Vector3d omega_raw = gac_cmd.Wd;
    const double F_des = -gac.f_total; // (f_total > 0)
    
    // ==== MRG CALC ====
    const Eigen::Vector3d euler_rpy = R_to_rpy(s.R);
    { // MPC send
      if (phase == Phase::MRG_NO_COT || phase == Phase::MRG_YES_COT) {
        std::lock_guard<std::mutex> mpc_lk(mpc_mtx);
        if (!g_mpc_busy && !g_mpc_input.has && !g_mpc_output.has) { // push next solve immediately after the previous output
          mpc_key += 1;

          int k = 0; // fill initial state(x)
          g_mpc_input.x_0(k++) = euler_rpy(0); g_mpc_input.x_0(k++) = euler_rpy(1); g_mpc_input.x_0(k++) = euler_rpy(2); // theta(0,1,2)
          g_mpc_input.x_0(k++) = s.omega(0); g_mpc_input.x_0(k++) = s.omega(1); g_mpc_input.x_0(k++) = s.omega(2); // omega(3,4,5)
          g_mpc_input.x_0(k++) = s.r1(0); g_mpc_input.x_0(k++) = s.r2(0); g_mpc_input.x_0(k++) = s.r3(0); g_mpc_input.x_0(k++) = s.r4(0); // r_rotor_x(6,7,8,9)
          g_mpc_input.x_0(k++) = s.r1(1); g_mpc_input.x_0(k++) = s.r2(1); g_mpc_input.x_0(k++) = s.r3(1); g_mpc_input.x_0(k++) = s.r4(1); // r_rotor_y(10,11,12,13)
          g_mpc_input.x_0(k++) = cmd.d_theta(0); g_mpc_input.x_0(k++) = cmd.d_theta(1); g_mpc_input.x_0(k++) = cmd.d_theta(2); // delta_theta(14,15,16)
          g_mpc_input.x_0(k++) = cmd.r1(0); g_mpc_input.x_0(k++) = cmd.r2(0); g_mpc_input.x_0(k++) = cmd.r3(0); g_mpc_input.x_0(k++) = cmd.r4(0); // r_rotor_cmd_x(17,18,19,20)
          g_mpc_input.x_0(k++) = cmd.r1(1); g_mpc_input.x_0(k++) = cmd.r2(1); g_mpc_input.x_0(k++) = cmd.r3(1); g_mpc_input.x_0(k++) = cmd.r4(1); // r_rotor_cmd_y(21,22,23,24)

          // fill initial control input(u)
          for (int l=0; l<11; ++l) {g_mpc_input.u_0(l) = l_mpc_output.u_rate(l, 0);}

          int m = 0; // fill initial parameter(p)
          for (int j=0; j<3; ++j) {for (int i=0; i<3; ++i) {g_mpc_input.p(m++) = R_raw(i, j);}} // R_raw(0~8), column-major order to match CasADi reshape
          g_mpc_input.p(m++) = omega_raw(0); g_mpc_input.p(m++) = omega_raw(1); g_mpc_input.p(m++) = omega_raw(2); // omega_raw(9~11)
          g_mpc_input.p(m++) = -gac.f_total; // T_des(12)

          if (phase==Phase::MRG_YES_COT) {g_mpc_input.use_cot = true;}
          else {g_mpc_input.use_cot = false;}

          g_mpc_input.steps_req = param::N_STEPS_REQ;
          g_mpc_input.t = now;
          g_mpc_input.key = mpc_key;
          g_mpc_input.epoch = g_mpc_epoch.load(std::memory_order_relaxed);
          g_mpc_input.has = true;
          g_mpc_busy = true;
          mpc_cv.notify_one();
        }
      }
    }

    { // MPC get
      std::lock_guard<std::mutex> mpc_lk(mpc_mtx);
      if (g_mpc_output.has) {
        const bool mpc_on = (phase == Phase::MRG_NO_COT || phase == Phase::MRG_YES_COT);
        const bool epoch_ok = (g_mpc_output.epoch == g_mpc_epoch.load(std::memory_order_relaxed));
        const bool key_ok = (g_mpc_output.key == mpc_key);
        const bool solve_ok = (g_mpc_output.state == 0);

        // l_mpc_output updated only when solve succeed.
        if (mpc_on && epoch_ok && key_ok && solve_ok) {l_mpc_output = g_mpc_output;}
        else if (mpc_on && epoch_ok && !key_ok) {mpc_reset_locked(mpc_key);}
        l_mpc_output.state = g_mpc_output.state; // *BUT l_mpc_output state indicates previous solve state(for logging)*
        g_mpc_output.has = false;
      }
    }

    if (phase == Phase::MRG_NO_COT || phase == Phase::MRG_YES_COT) { // MPC unpack
      const bool epoch_ok = (l_mpc_output.epoch == g_mpc_epoch.load(std::memory_order_relaxed));
      const bool time_ok = ((now - l_mpc_output.t) < param::MPC_TIMEOUT_DURATUION);
      const bool solve_ok = (l_mpc_output.state == 0);
      if (epoch_ok && time_ok && solve_ok) {
        const std::size_t idx = static_cast<std::size_t>(std::floor(std::chrono::duration<double>(now - l_mpc_output.t).count() / param::MPC_STEP_DT));
        cmd.d_theta = l_mpc_output.u_opt.col(idx).head<3>();
        cmd.r1(0) = l_mpc_output.u_opt(3, idx); cmd.r1(1) = l_mpc_output.u_opt(7, idx);
        cmd.r2(0) = l_mpc_output.u_opt(4, idx); cmd.r2(1) = l_mpc_output.u_opt(8, idx);
        cmd.r3(0) = l_mpc_output.u_opt(5, idx); cmd.r3(1) = l_mpc_output.u_opt(9, idx);
        cmd.r4(0) = l_mpc_output.u_opt(6, idx); cmd.r4(1) = l_mpc_output.u_opt(10, idx);
        // if (!workspace_guard(cmd.r1, cmd.r2, cmd.r3, cmd.r4)) {
        //   // cmd.d_theta *= param::GOES_2_ZERO_A;
        //   // cmd.r1 = param::GOES_2_ZERO_A*cmd.r1 + param::GOES_2_ZERO_B*param::r1_init;
        //   // cmd.r2 = param::GOES_2_ZERO_A*cmd.r2 + param::GOES_2_ZERO_B*param::r2_init;
        //   // cmd.r3 = param::GOES_2_ZERO_A*cmd.r3 + param::GOES_2_ZERO_B*param::r3_init;
        //   // cmd.r4 = param::GOES_2_ZERO_A*cmd.r4 + param::GOES_2_ZERO_B*param::r4_init;
        //   // l_mpc_output.u_rate.setZero();
        //   l_mpc_output.state = 1;
        // }
      }
      else { // solve failed timeout
        cmd.d_theta *= param::GOES_2_ZERO_A;
        cmd.r1 = param::GOES_2_ZERO_A*cmd.r1 + param::GOES_2_ZERO_B*param::r1_init;
        cmd.r2 = param::GOES_2_ZERO_A*cmd.r2 + param::GOES_2_ZERO_B*param::r2_init;
        cmd.r3 = param::GOES_2_ZERO_A*cmd.r3 + param::GOES_2_ZERO_B*param::r3_init;
        cmd.r4 = param::GOES_2_ZERO_A*cmd.r4 + param::GOES_2_ZERO_B*param::r4_init;
        l_mpc_output.u_rate.setZero();
      }
    }
    else { // only GAC flight
      cmd.d_theta *= param::GOES_2_ZERO_A;
      cmd.r1 = param::GOES_2_ZERO_A*cmd.r1 + param::GOES_2_ZERO_B*param::r1_init;
      cmd.r2 = param::GOES_2_ZERO_A*cmd.r2 + param::GOES_2_ZERO_B*param::r2_init;
      cmd.r3 = param::GOES_2_ZERO_A*cmd.r3 + param::GOES_2_ZERO_B*param::r3_init;
      cmd.r4 = param::GOES_2_ZERO_A*cmd.r4 + param::GOES_2_ZERO_B*param::r4_init;
      l_mpc_output.u_rate.setZero();
    }

    // ==== ATTITUDE CONTROL ====
    const Eigen::Matrix3d R_d = R_raw * expm_hat(cmd.d_theta);
    Eigen::Vector3d tau_des = gac.attitude_control(R_d);
    
    // ==== CONTORL ALLOCATION ====
    Eigen::Vector4d thrust_des   = Eigen::Vector4d::Zero(); // (thrust_des > 0)
    Eigen::Vector4d tilt_ang_des = Eigen::Vector4d::Zero();
    Sequential_Allocation(F_des, tau_des, cmd.tauz_bar, s.arm_q, s.r_com, thrust_des, tilt_ang_des);

    // --- thruster constraint ---
    Eigen::Vector4d thrust_cmd   = Eigen::Vector4d::Zero(); // (thrust_cmd > 0)
    for (uint8_t i=0; i<4; ++i) {thrust_cmd(i) = std::clamp(thrust_des(i), 0.0, param::SATURATION_THRUST);}

    // --- get joint angle commands ---
    double q_d[20] = {0};
    IK(param::r1_init, param::r2_init, param::r3_init, param::r4_init, tilt_ang_des, q_d);

    // --- get pwm ---
    Eigen::Vector4d pwm;
    for (int i = 0; i < 4; ++i) {
      pwm(i) = std::sqrt(std::max(0.0, (thrust_cmd(i) - param::PWM_B) / param::PWM_A));
      pwm(i) = std::clamp(pwm(i), 0.0, 1.0);
    }

    // --- overridings for soft start ---
    switch (phase) {
      case Phase::ARMED: {
        for (int i = 0; i < 4; ++i) { pwm(i) = 0.0; }
        gac.integral_reset();
        cmd.tauz_bar = 0.0;
      } break;

      case Phase::IDLE: {
        for (int i = 0; i < 4; ++i) { pwm(i) = param::IDLE_PWM_DUTY; }
        gac.integral_reset();
        cmd.tauz_bar = 0.0;
      } break;

      case Phase::RISING: {
        rising_coeff += param::RISING_COEFF_INC;
        if (rising_coeff >= 1.0) {
          rising_coeff = 1.0;
          for (int i = 0; i < 4; ++i) { pwm(i) *= rising_coeff; }
          phase = Phase::GAC_FLIGHT;
          std::fprintf(stdout, "flight state -> [GAC_FLIGHT]\n"); std::fflush(stdout);
        }
      } break;
    }

    // --- write pwm ---
    if (!g_killed.load(std::memory_order_relaxed)) {if (!teensy.write_pwm(pwm(0), pwm(1), pwm(2), pwm(3))) {KILL("[CAN] : PLZ CALL GOD");}}
    else {teensy.write_zeros();}

    // --- write dynamixel ---
    if (!g_killed.load(std::memory_order_relaxed)) {dxl.write_goal(q_d);}

    // ------ [Data logging] -----------------------------------------------------------------------------
    {
      mmap_logger::LogData ld{};

      ld.t = std::chrono::duration<float>(now - initial_time).count();

      ld.pos_d[0] = static_cast<float>(cmd.pos(0));
      ld.pos_d[1] = static_cast<float>(cmd.pos(1));
      ld.pos_d[2] = static_cast<float>(cmd.pos(2));
      ld.vel_d[0] = static_cast<float>(cmd.vel(0));
      ld.vel_d[1] = static_cast<float>(cmd.vel(1));
      ld.vel_d[2] = static_cast<float>(cmd.vel(2));
      ld.acc_d[0] = static_cast<float>(cmd.acc(0));
      ld.acc_d[1] = static_cast<float>(cmd.acc(1));
      ld.acc_d[2] = static_cast<float>(cmd.acc(2));

      ld.pos[0] = static_cast<float>(s.pos(0));
      ld.pos[1] = static_cast<float>(s.pos(1));
      ld.pos[2] = static_cast<float>(s.pos(2));
      ld.vel[0] = static_cast<float>(s.vel(0));
      ld.vel[1] = static_cast<float>(s.vel(1));
      ld.vel[2] = static_cast<float>(s.vel(2));
      ld.acc[0] = static_cast<float>(s.acc(0));
      ld.acc[1] = static_cast<float>(s.acc(1));
      ld.acc[2] = static_cast<float>(s.acc(2));

      {
        const Eigen::Vector3d rpy_raw = R_to_rpy(R_raw);
        ld.rpy_raw[0] = static_cast<float>(rpy_raw(0));
        ld.rpy_raw[1] = static_cast<float>(rpy_raw(1));
        ld.rpy_raw[2] = static_cast<float>(rpy_raw(2));
      }
      {
        const Eigen::Vector3d rpy_d = R_to_rpy(R_d);
        ld.rpy_d[0] = static_cast<float>(rpy_d(0));
        ld.rpy_d[1] = static_cast<float>(rpy_d(1));
        ld.rpy_d[2] = static_cast<float>(rpy_d(2));
      }
      ld.omega_d[0] = static_cast<float>(omega_raw(0));
      ld.omega_d[1] = static_cast<float>(omega_raw(1));
      ld.omega_d[2] = static_cast<float>(omega_raw(2));
      ld.alpha_d[0] = static_cast<float>(gac_cmd.Wd_dot(0));
      ld.alpha_d[1] = static_cast<float>(gac_cmd.Wd_dot(1));
      ld.alpha_d[2] = static_cast<float>(gac_cmd.Wd_dot(2));

      ld.rpy[0]   = static_cast<float>(euler_rpy(0));
      ld.rpy[1]   = static_cast<float>(euler_rpy(1));
      ld.rpy[2]   = static_cast<float>(euler_rpy(2));
      ld.omega[0] = static_cast<float>(s.omega(0));
      ld.omega[1] = static_cast<float>(s.omega(1));
      ld.omega[2] = static_cast<float>(s.omega(2));
      ld.alpha[0] = static_cast<float>(s.alpha(0));
      ld.alpha[1] = static_cast<float>(s.alpha(1));
      ld.alpha[2] = static_cast<float>(s.alpha(2));

      ld.f_total  = static_cast<float>(gac.f_total);
      ld.tau_d[0] = static_cast<float>(tau_des(0));
      ld.tau_d[1] = static_cast<float>(tau_des(1));
      ld.tau_d[2] = static_cast<float>(tau_des(2));

      ld.tau_z_t        = static_cast<float>(cmd.tauz_bar);
      ld.tilt_rad[0]    = static_cast<float>(tilt_ang_des(0));
      ld.tilt_rad[1]    = static_cast<float>(tilt_ang_des(1));
      ld.tilt_rad[2]    = static_cast<float>(tilt_ang_des(2));
      ld.tilt_rad[3]    = static_cast<float>(tilt_ang_des(3));
      ld.f_thrst[0]     = static_cast<float>(thrust_des(0));
      ld.f_thrst[1]     = static_cast<float>(thrust_des(1));
      ld.f_thrst[2]     = static_cast<float>(thrust_des(2));
      ld.f_thrst[3]     = static_cast<float>(thrust_des(3));
      ld.f_thrst_con[0] = static_cast<float>(thrust_cmd(0));
      ld.f_thrst_con[1] = static_cast<float>(thrust_cmd(1));
      ld.f_thrst_con[2] = static_cast<float>(thrust_cmd(2));
      ld.f_thrst_con[3] = static_cast<float>(thrust_cmd(3));

      {
        const Eigen::Vector2d tau_off(F_des*(s.r_cot(1)-s.r_com(1)), -F_des*(s.r_cot(0)-s.r_com(0)));
        ld.tau_off[0] = static_cast<float>(tau_off(0));
        ld.tau_off[1] = static_cast<float>(tau_off(1));
      }
      {
        const Eigen::Vector3d tau_thrust = Wrench_2_Torque(thrust_cmd, s.r1, s.r2, s.r3, s.r4, s.r_com);
        ld.tau_thrust[0] = static_cast<float>(tau_thrust(0));
        ld.tau_thrust[1] = static_cast<float>(tau_thrust(1));
        ld.tau_thrust[2] = static_cast<float>(tau_thrust(2));
      }

        ld.r_rotor1[0] = static_cast<float>(s.r1(0));
        ld.r_rotor1[1] = static_cast<float>(s.r1(1));
        ld.r_rotor2[0] = static_cast<float>(s.r2(0));
        ld.r_rotor2[1] = static_cast<float>(s.r2(1));
        ld.r_rotor3[0] = static_cast<float>(s.r3(0));
        ld.r_rotor3[1] = static_cast<float>(s.r3(1));
        ld.r_rotor4[0] = static_cast<float>(s.r4(0));
        ld.r_rotor4[1] = static_cast<float>(s.r4(1));
        ld.r_cot[0] = static_cast<float>(s.r_cot(0));
        ld.r_cot[1] = static_cast<float>(s.r_cot(1));

        ld.r_rotor1_d[0] = static_cast<float>(cmd.r1(0));
        ld.r_rotor1_d[1] = static_cast<float>(cmd.r1(1));
        ld.r_rotor2_d[0] = static_cast<float>(cmd.r2(0));
        ld.r_rotor2_d[1] = static_cast<float>(cmd.r2(1));
        ld.r_rotor3_d[0] = static_cast<float>(cmd.r3(0));
        ld.r_rotor3_d[1] = static_cast<float>(cmd.r3(1));
        ld.r_rotor4_d[0] = static_cast<float>(cmd.r4(0));
        ld.r_rotor4_d[1] = static_cast<float>(cmd.r4(1));
        {
          const Eigen::Vector3d r_cot_d = (cmd.r1 + cmd.r2 + cmd.r3 + cmd.r4) / 4.0;
          ld.r_cot_d[0] = static_cast<float>(r_cot_d(0));
          ld.r_cot_d[1] = static_cast<float>(r_cot_d(1));
        }

      for (uint8_t i=0; i<20; ++i){ld.q[i]     = static_cast<float>(s.arm_q[i]);}
      for (uint8_t i=0; i<20; ++i){ld.q_cmd[i] = static_cast<float>(q_d[i]);}

      ld.solve_ms = static_cast<float>(l_mpc_output.solve_ms);
      ld.solve_status = static_cast<int32_t>(l_mpc_output.state);

      ld.phase = static_cast<uint8_t>(phase);

      logger.push(ld);

      if (mmap_logger::log_fp) {std::fwrite(&ld, sizeof(ld), 1, mmap_logger::log_fp);}
    }

    // delay for keeping control Hz. May wake up a little early for fresh imu data.
    const std::chrono::steady_clock::time_point done_tick = std::chrono::steady_clock::now();
    if (done_tick < next_control_tick) {
      if(t265.wait_new_frame_until(next_control_tick, last_t265_cnt)){
        std::chrono::steady_clock::duration pull_tick = next_control_tick - std::chrono::steady_clock::now();
        pull_tick = std::clamp(pull_tick, std::chrono::steady_clock::duration::zero(), param::MAX_PULL_TICK);
        next_control_tick -= pull_tick;
      }
    }
    next_control_tick += param::CTRL_DT;
  }

  teensy.close();
  mpc_cv.notify_all();
  t265.request_stop();
  opti.request_stop();
  sbus.request_stop();
  dxl.request_stop();

  if (mmap_logger::log_fp) {
    fclose(mmap_logger::log_fp);
    mmap_logger::log_fp = nullptr;
  }

  if (th_t265.joinable()) th_t265.join();
  if (th_opti.joinable()) th_opti.join();
  if (th_sbus.joinable()) th_sbus.join();
  if (th_dxl.joinable()) th_dxl.join();
  if (th_mpc.joinable()) th_mpc.join();

  return 0;
}
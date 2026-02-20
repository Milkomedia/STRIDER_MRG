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
        g_mpc_output.has = true;
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
  double rising_coeff = param::INITIAL_RISING_COEFF;

  // --- MRG parameters ---
  bool mpc_in_solving = false;
  uint32_t mpc_key = 1;
  strider_mpc::MPCOutput l_mpc_output;
  
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
        s.omega(0) = -t265_frame.omega[2]; s.omega(1) =  t265_frame.omega[0]; s.omega(2) = gyro_bf[2].update(-t265_frame.omega[1], now);

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
        s.vel(0) = opti_vel_bf[0].update(vel_raw(0), now); s.vel(1) = opti_vel_bf[1].update(vel_raw(1), now); s.vel(2) = opti_vel_bf[2].update(vel_raw(2), now);

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
        cmd.r_cot(2) = sbus_cotz_map(sbus_frame.ch[10]);
        cmd.l        = sbus_l_map(sbus_frame.ch[11]);

        if (phase == Phase::GAC_FLIGHT) {
          if (sbus_frame.ch[7] == 1696) {
            g_mpc_activated.store(true, std::memory_order_relaxed);
            phase = Phase::MRG_YES_COT;
            std::fprintf(stdout, "flight state -> [MRG_YES_COT]\n"); std::fflush(stdout);
          }
          if (sbus_frame.ch[7] == 352) {
            g_mpc_activated.store(true, std::memory_order_relaxed);
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
        s.r_cot = FK(s.arm_q);
        s.r_com = 0.523 * s.r_cot; 
        last_dxl_cnt = cur_dxl_cnt;
      }
    }

    // ==== POSITION CONTROL ====
    gac_cmd.xd  = cmd.pos;
    gac_cmd.b1d = cmd.heading;
    gac_state.x = s.pos;
    gac_state.v = s.vel;
    gac_state.a = s.acc;
    gac_state.R = s.R;
    gac_state.W = s.omega;
    gac.position_control();
    const Eigen::Matrix3d R_raw = gac_cmd.Rd;
    const double F_des = -gac.f_total; // (f_total > 0)
    
    // ==== MRG CALC ====
    const Eigen::Vector3d euler_rpy = R_to_rpy(s.R);
    { // MPC send
      std::lock_guard<std::mutex> mpc_lk(mpc_mtx);
      if (g_mpc_activated.load(std::memory_order_relaxed)) {
        if (!mpc_in_solving) {
          if (now >= next_mpc_tick) {
            if (!g_mpc_output.has) {
              next_mpc_tick += param::MPC_DT;
              mpc_key += 1;
              
              int k = 0; // fill initial state(x)
              g_mpc_input.x_0(k++) = euler_rpy(0); g_mpc_input.x_0(k++) = euler_rpy(1); g_mpc_input.x_0(k++) = euler_rpy(2); // theta(0,1,2)
              g_mpc_input.x_0(k++) = s.omega(0); g_mpc_input.x_0(k++) = s.omega(1); g_mpc_input.x_0(k++) = s.omega(2); // omega(3,4,5)
              g_mpc_input.x_0(k++) = s.r_cot(0); g_mpc_input.x_0(k++) = s.r_cot(1); // r_cot(6,7)
              g_mpc_input.x_0(k++) = cmd.d_theta(0); g_mpc_input.x_0(k++) = cmd.d_theta(1); g_mpc_input.x_0(k++) = cmd.d_theta(2); // delta_theta(8,9,10)
              g_mpc_input.x_0(k++) = cmd.r_cot(0); g_mpc_input.x_0(k++) = cmd.r_cot(1); // r_cot_cmd(11,12)

              // fill initial control input(u)
              for (int l=0; l<5; ++l) {g_mpc_input.u_0(l) = l_mpc_output.u_rate(l);}

              int m = 0; // fill initial parameter(p)
              for (int j=0; j<3; ++j) {for (int i=0; i<3; ++i) {g_mpc_input.p(m++) = R_raw(i, j);}} // Rraw_mpc(0~8), column-major order to match CasADi reshape
              g_mpc_input.p(m++) = 0.5 * cmd.l; // l(9)
              g_mpc_input.p(m++) = gac.f_total; // F_des(10)

              int n = 0;
              g_mpc_input.log(n++) = s.pos(0); g_mpc_input.log(n++) = s.pos(1); g_mpc_input.log(n++) = s.pos(2); // pos_cur
              g_mpc_input.log(n++) = cmd.pos(0); g_mpc_input.log(n++) = cmd.pos(1); g_mpc_input.log(n++) = cmd.pos(2); // pos_des
              g_mpc_input.log(n++) = 0.; g_mpc_input.log(n++) = 0.; g_mpc_input.log(n++) = 0.; g_mpc_input.log(n++) = 0.; // F1234

              if (phase == Phase::MRG_YES_COT) {g_mpc_input.use_cot = true;}
              else if (phase == Phase::MRG_NO_COT) {g_mpc_input.use_cot = false;}
              g_mpc_input.t = now;
              g_mpc_input.key = mpc_key;
              g_mpc_input.has = true;
              mpc_cv.notify_one();
            }
          }
        }
      }
      else { // cot goes to zero when MPC deactivated
        // cmd.r_cot(0) *= 0.995;
        // cmd.r_cot(1) *= 0.995;
      }
    }

    bool got_mpc = false;
    { // check got_mpc
      std::lock_guard<std::mutex> mpc_lk(mpc_mtx);
      if (g_mpc_output.has) {
        if (g_mpc_output.key == mpc_key) {
          l_mpc_output = g_mpc_output;
          got_mpc = true;
        }
        // else{std::printf("\n\n[MPC KEY WRONG ERROR. RESTART NOW]\n\n");}
      }
    }

    if (got_mpc) { // MPC get
      if (l_mpc_output.state == 0) {
        cmd.d_theta = l_mpc_output.u_opt.template head<3>();                  // [0,1,2]
        // cmd.r_cot(0) = 0.995 * cmd.r_cot(0) + 0.005 * l_mpc_output.u_opt(3);  // [3]
        // cmd.r_cot(1) = 0.995 * cmd.r_cot(1) + 0.005 * l_mpc_output.u_opt(4);  // [4]
      }
      else { // solve failed
        cmd.d_theta  *= 0.9;
        // cmd.r_cot(0)    *= 0.9;
        // cmd.r_cot(1)    *= 0.9;
        l_mpc_output.u_rate.setZero();
      }

      l_mpc_output.has = false;

      // next time to solve
      if (now >= l_mpc_output.t + param::MPC_DT) {next_mpc_tick = now;}
      else {next_mpc_tick = l_mpc_output.t + param::MPC_DT;}
    }

    // ==== ATTITUDE CONTROL ====
    const Eigen::Matrix3d R_d = R_raw * expm_hat(cmd.d_theta);
    Eigen::Vector3d tau_des = gac.attitude_control(R_d);
    
    // ==== CONTORL ALLOCATION ====
    Eigen::Vector4d thrust_des   = Eigen::Vector4d::Zero(); // (f_1234 > 0)
    Eigen::Vector4d tilt_ang_des = Eigen::Vector4d::Zero();
    Sequential_Allocation(F_des, tau_des, cmd.tauz_bar, s.arm_q, s.r_com, thrust_des, tilt_ang_des);

    // --- get joint angle commands ---
    double q_d[20] = {0};
    IK(cmd.r_cot, cmd.R_cot, tilt_ang_des, cmd.l, q_d);

    // --- get pwm ---
    Eigen::Vector4d pwm;
    for (int i = 0; i < 4; ++i) {
      pwm(i) = std::sqrt(std::max(0.0, (thrust_des(i) - param::PWM_B) / param::PWM_A));
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

      ld.pos[0] = static_cast<float>(s.pos(0));
      ld.pos[1] = static_cast<float>(s.pos(1));
      ld.pos[2] = static_cast<float>(s.pos(2));

      ld.vel[0] = static_cast<float>(s.vel(0));
      ld.vel[1] = static_cast<float>(s.vel(1));
      ld.vel[2] = static_cast<float>(s.vel(2));

      ld.rpy[0] = static_cast<float>(euler_rpy(0));
      ld.rpy[1] = static_cast<float>(euler_rpy(1));
      ld.rpy[2] = static_cast<float>(euler_rpy(2));

      ld.omega[0] = static_cast<float>(s.omega(0));
      ld.omega[1] = static_cast<float>(s.omega(1));
      ld.omega[2] = static_cast<float>(s.omega(2));

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

      ld.tau_d[0] = static_cast<float>(tau_des(0));
      ld.tau_d[1] = static_cast<float>(tau_des(1));
      ld.tau_d[2] = static_cast<float>(tau_des(2));

      {
        const double f_total = gac.f_total;
        const Eigen::Vector2d tau_off(-f_total * s.r_cot(1), f_total * s.r_cot(0));
        ld.tau_off[0] = static_cast<float>(tau_off(0));
        ld.tau_off[1] = static_cast<float>(tau_off(1));

        ld.tau_thrust[0] = static_cast<float>(tau_des(0) - tau_off(0));
        ld.tau_thrust[1] = static_cast<float>(tau_des(1) - tau_off(1));
      }

      for (int i = 0; i < 4; ++i) {
        ld.tilt_rad[i] = static_cast<float>(tilt_ang_des(i));
        ld.f_thrust[i] = static_cast<float>(thrust_des(i));
      }
      ld.f_total = static_cast<float>(gac.f_total);

      ld.r_cot[0] = static_cast<float>(s.r_cot(0));
      ld.r_cot[1] = static_cast<float>(s.r_cot(1));
      ld.r_cot[2] = static_cast<float>(s.r_cot(2));

      ld.r_cot_cmd[0] = static_cast<float>(cmd.r_cot(0));
      ld.r_cot_cmd[1] = static_cast<float>(cmd.r_cot(1));
      ld.r_cot_cmd[2] = static_cast<float>(cmd.r_cot(2));

      for (int i = 0; i < 20; ++i) {
        ld.q_mea[i] = static_cast<float>(s.arm_q[i]);
        ld.q_d[i]   = static_cast<float>(q_d[i]);
      }

      // solve
      ld.solve_ms = static_cast<float>(l_mpc_output.solve_ms);
      ld.solve_status = static_cast<int32_t>(l_mpc_output.state);

      ld.sbus_used[0] = sbus_frame.ch[0];   // pos_x
      ld.sbus_used[1] = sbus_frame.ch[1];   // pos_y
      ld.sbus_used[2] = sbus_frame.ch[2];   // pos_z
      ld.sbus_used[3] = sbus_frame.ch[3];   // heading/yaw
      ld.sbus_used[4] = sbus_frame.ch[7];   // mode
      ld.sbus_used[5] = sbus_frame.ch[8];   // toggle
      ld.sbus_used[6] = sbus_frame.ch[10];  // L-dial
      ld.sbus_used[7] = sbus_frame.ch[11];  // R-dial

      logger.push(ld);
      if (mmap_logger::log_fp) {
        std::fwrite(&ld, sizeof(ld), 1, mmap_logger::log_fp);
      }
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
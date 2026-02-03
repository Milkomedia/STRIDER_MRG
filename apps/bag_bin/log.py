import os
import sys
import numpy as np
import matplotlib.pyplot as plt

fileday = "controller_log"
filename = fileday + sys.argv[1] + ".bin" if len(sys.argv) >= 2 else fileday + ".bin"

THRUST_LP_CUTOFF_HZ = 0.1

ZUP_POS_FLIP_YZ = True
ZUP_RPY_FLIP_PY = True

def lowpass_1pole(x, t, cutoff_hz):
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    tau = 1.0 / (2.0 * np.pi * max(cutoff_hz, 1e-6))
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        if dt <= 0 or not np.isfinite(dt):
            y[k] = y[k - 1]
            continue
        alpha = dt / (tau + dt)
        y[k] = y[k - 1] + alpha * (x[k] - y[k - 1])
    return y.squeeze()

log_dtype = np.dtype([
    ("t",        "<f4"),
    ("pos_d",    "<f4", (3,)),
    ("pos",      "<f4", (3,)),
    ("vel",      "<f4", (3,)),
    ("omega",    "<f4", (3,)),
    ("rpy",      "<f4", (3,)),
    ("rpy_raw",  "<f4", (3,)),
    ("rpy_d",    "<f4", (3,)),
    ("tau_d",     "<f4", (3,)),
    ("tau_off",   "<f4", (2,)),
    ("tau_thrust","<f4", (2,)),
    ("tilt_rad", "<f4", (4,)),
    ("f_thrust", "<f4", (4,)),
    ("f_total",  "<f4"),
    ("r_cot",     "<f4", (2,)),
    ("r_cot_cmd", "<f4", (2,)),
    ("solve_ms",    "<f4"),
    ("solve_status","<i4"),
], align=False)

def load_log_new_only(fn):
    if not os.path.exists(fn):
        raise FileNotFoundError(fn)
    fsize = os.path.getsize(fn)
    if fsize == 0:
        raise ValueError(f"Empty file: {fn}")
    rec = log_dtype.itemsize
    if fsize % rec != 0:
        raise ValueError(f"struct mismatch: file={fsize} bytes, record={rec} bytes")
    data = np.fromfile(fn, dtype=log_dtype)
    if data.size == 0:
        raise ValueError(f"Empty data: {fn}")
    return data

data = load_log_new_only(filename)

t = data["t"].astype(np.float64)

pos   = data["pos"].astype(np.float64)
pos_d = data["pos_d"].astype(np.float64)
v     = data["vel"].astype(np.float64)
omega = data["omega"].astype(np.float64)

rpy     = data["rpy"].astype(np.float64)
rpy_d   = data["rpy_d"].astype(np.float64)
rpy_raw = data["rpy_raw"].astype(np.float64)

tau_d = data["tau_d"].astype(np.float64)

if ZUP_POS_FLIP_YZ:
    pos[:, 1] *= -1.0
    pos[:, 2] *= -1.0
    pos_d[:, 1] *= -1.0
    pos_d[:, 2] *= -1.0
    v[:, 1] *= -1.0
    v[:, 2] *= -1.0

if ZUP_RPY_FLIP_PY:
    rpy[:, 1] *= -1.0
    rpy[:, 2] *= -1.0
    rpy_d[:, 1] *= -1.0
    rpy_d[:, 2] *= -1.0
    rpy_raw[:, 1] *= -1.0
    rpy_raw[:, 2] *= -1.0
    omega[:, 1] *= -1.0
    omega[:, 2] *= -1.0
    tau_d[:, 1] *= -1.0
    tau_d[:, 2] *= -1.0

att   = rpy * 180.0 / np.pi
att_d = rpy_d * 180.0 / np.pi
omega_deg = omega * 180.0 / np.pi

dt = np.gradient(t)
dt = np.where((dt > 1e-9) & np.isfinite(dt), dt, np.nan)
hz = np.nan_to_num(1.0 / dt, nan=0.0, posinf=0.0, neginf=0.0)

wrench = np.zeros((len(t), 4), dtype=np.float64)
wrench[:, 0] = data["f_total"].astype(np.float64)
wrench[:, 1:4] = tau_d

f_thrust = data["f_thrust"].astype(np.float64)
f_thrust_lp = lowpass_1pole(f_thrust, t, THRUST_LP_CUTOFF_HZ)

print(f"File: {filename}")
print(f"Samples: {len(t)}, Time: {t[0]:.3f} ~ {t[-1]:.3f} s, dt~ {np.nanmedian(dt):.6f} s")
print("Plot frame: Z-up (python-side remap)")

fig, axs = plt.subplots(4, 3, figsize=(15, 10), sharex=True)

labels_xyz = ['x', 'y', 'z']
for i in range(3):
    axs[0, i].plot(t, pos[:, i], label=labels_xyz[i])
    axs[0, i].plot(t, pos_d[:, i], '--', label=f'{labels_xyz[i]}_d')
    axs[0, i].set_ylabel('pos [m]')
    axs[0, i].legend()
    axs[0, i].grid(True)

for i in range(3):
    axs[1, i].plot(t, v[:, i])
    axs[1, i].set_ylabel('vel [m/s]')
    axs[1, i].grid(True)

labels_rpy = ['roll', 'pitch', 'yaw']
for i in range(3):
    axs[2, i].plot(t, att[:, i], label=labels_rpy[i])
    axs[2, i].plot(t, att_d[:, i], '--', label=f'{labels_rpy[i]}_d')
    axs[2, i].set_ylabel('att [deg]')
    axs[2, i].legend()
    axs[2, i].grid(True)

labels_omega = ['ω_roll', 'ω_pitch', 'ω_yaw']
for i in range(3):
    axs[3, i].plot(t, omega_deg[:, i])
    axs[3, i].set_ylabel('deg/s')
    axs[3, i].set_xlabel('time [s]')
    axs[3, i].grid(True)

fig.suptitle("State / Desired / Velocity / Attitude (Z-up view)")
plt.tight_layout()

fig2, axs2 = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

w_labels = ['Fz (f_total)', 'tau_roll', 'tau_pitch', 'tau_yaw']
for i in range(4):
    axs2[i].plot(t, wrench[:, i])
    axs2[i].set_ylabel(w_labels[i])
    axs2[i].grid(True)

axs2[4].plot(t, f_thrust_lp[:, 0], label='f1')
axs2[4].plot(t, f_thrust_lp[:, 1], label='f2')
axs2[4].plot(t, f_thrust_lp[:, 2], label='f3')
axs2[4].plot(t, f_thrust_lp[:, 3], label='f4')
axs2[4].set_ylabel(f"thrust [N]\nLPF {THRUST_LP_CUTOFF_HZ:.1f}Hz")
axs2[4].legend()
axs2[4].grid(True)

axs2[5].plot(t, hz)
axs2[5].set_ylabel('Hz')
axs2[5].set_xlabel('time [s]')
axs2[5].grid(True)

fig2.suptitle("Wrench / Thrust(4) / Control Loop Frequency")
plt.tight_layout()

plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
from typing import Dict, Tuple

import numpy as np
from PyQt5 import QtWidgets
import pyqtgraph as pg


# ============================================================
# UI
# ============================================================
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
pg.setConfigOption("antialias", True)


# ============================================================
# Plot/filter/model params
# ============================================================
ROLL_RAD2DEG = 180.0 / np.pi

FC_THRUST_SUM_HZ = 0.1
FC_DUTY_HZ = 1.0

PWM_A = 46.5435
PWM_B = 8.6111

# colors
C_BLUE   = "#1f77b4"
C_ORANGE = "#ff7f0e"
C_GREEN  = "#2ca02c"
C_RED    = "#d62728"
C_PURPLE = "#9467bd"
C_BROWN  = "#8c564b"
C_PINK   = "#e377c2"
C_GRAY   = "#7f7f7f"
C_OLIVE  = "#bcbd22"
C_CYAN   = "#17becf"


# ============================================================
# Helpers
# ============================================================
def pen(c, w=2, style=None):
    if style is None:
        return pg.mkPen(color=c, width=w)
    return pg.mkPen(color=c, width=w, style=style)


def style_plot(p, ylab=None):
    p.showGrid(x=True, y=True, alpha=0.25)
    p.addLegend(offset=(8, 8))
    if ylab is not None:
        p.setLabel("left", ylab)
    p.setLabel("bottom", "time", units="s")


def lpf_1pole(x: np.ndarray, t: np.ndarray, fc_hz: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    if x.size == 0 or fc_hz <= 0.0:
        return x.copy()

    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]

    for k in range(1, x.size):
        dt = t[k] - t[k - 1]
        if not np.isfinite(dt) or dt <= 0.0:
            y[k] = y[k - 1]
            continue

        a = 1.0 - math.exp(-2.0 * math.pi * fc_hz * dt)
        y[k] = y[k - 1] + a * (x[k] - y[k - 1])

    return y


def thrust_to_duty_percent(thrust_n: np.ndarray) -> np.ndarray:
    thrust_n = np.asarray(thrust_n, dtype=np.float64)
    duty = np.sqrt(np.clip((thrust_n - PWM_B) / PWM_A, 0.0, None))
    duty_pct = 100.0 * duty
    return np.clip(duty_pct, 0.0, 100.0)


def resolve_path() -> str:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="npz file path")
    args = ap.parse_args()
    return args.path


def load_log(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if not path.lower().endswith(".npz"):
        raise ValueError(f"npz file only: {path}")

    data = np.load(path)

    if "t" not in data.files:
        raise KeyError("npz must contain key 't'")

    t = np.asarray(data["t"], dtype=np.float64)
    ch = {k: np.asarray(data[k]) for k in data.files if k != "t"}

    required = [
        "pos", "pos_d",
        "rpy", "rpy_d", "rpy_raw",
        "tau_d", "tau_off", "tau_thrust",
        "f_thrst_con",
    ]
    missing = [k for k in required if k not in ch]
    if missing:
        raise KeyError(f"missing keys in npz: {missing}")

    return t, ch


# ============================================================
# Main
# ============================================================
def main() -> None:
    path = resolve_path()

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    t, ch = load_log(path)

    app = QtWidgets.QApplication(sys.argv)

    x = np.asarray(t, dtype=np.float64) - float(t[0])

    # --------------------------------------------------
    # data
    # --------------------------------------------------
    pos_y   = np.asarray(ch["pos"][:, 1], dtype=np.float64)
    pos_d_y = np.asarray(ch["pos_d"][:, 1], dtype=np.float64)

    roll     = np.asarray(ch["rpy"][:, 0], dtype=np.float64) * ROLL_RAD2DEG
    roll_d   = np.asarray(ch["rpy_d"][:, 0], dtype=np.float64) * ROLL_RAD2DEG
    roll_raw = np.asarray(ch["rpy_raw"][:, 0], dtype=np.float64) * ROLL_RAD2DEG

    tau_d_roll      = np.asarray(ch["tau_d"][:, 0], dtype=np.float64)
    tau_off_roll    = np.asarray(ch["tau_off"][:, 0], dtype=np.float64)
    tau_thrust_roll = np.asarray(ch["tau_thrust"][:, 0], dtype=np.float64)

    f = np.asarray(ch["f_thrst_con"], dtype=np.float64)

    # thrust sum LPF
    f1_sum = lpf_1pole(f[:, 0], x, FC_THRUST_SUM_HZ)
    f2_sum = lpf_1pole(f[:, 1], x, FC_THRUST_SUM_HZ)
    f3_sum = lpf_1pole(f[:, 2], x, FC_THRUST_SUM_HZ)
    f4_sum = lpf_1pole(f[:, 3], x, FC_THRUST_SUM_HZ)

    f12 = f1_sum + f2_sum
    f34 = f3_sum + f4_sum
    f_diff_abs = np.abs(f34 - f12)

    # thrust each LPF
    f1_each = lpf_1pole(f[:, 0], x, FC_THRUST_SUM_HZ)
    f2_each = lpf_1pole(f[:, 1], x, FC_THRUST_SUM_HZ)
    f3_each = lpf_1pole(f[:, 2], x, FC_THRUST_SUM_HZ)
    f4_each = lpf_1pole(f[:, 3], x, FC_THRUST_SUM_HZ)

    xmin = float(x.min())
    xmax = float(x.max())

    info_text = (
        f"path:{path} | samples:{len(x)} | "
        f"t0:{x[0]:.3f} | t1:{x[-1]:.3f} | "
        f"LPF thrust={FC_THRUST_SUM_HZ:.1f} Hz"
    )

    # ==================================================
    # window 1 : position / attitude / torque
    # ==================================================
    win1 = QtWidgets.QWidget()
    win1.setWindowTitle(f"state_view - {os.path.basename(path)}")
    win1.resize(1500, 950)

    layout1 = QtWidgets.QVBoxLayout(win1)
    info1 = QtWidgets.QLabel(info_text)
    layout1.addWidget(info1)

    glw1 = pg.GraphicsLayoutWidget()
    layout1.addWidget(glw1)

    p1 = glw1.addPlot(row=0, col=0, title="Position Y [m]")
    style_plot(p1, "m")
    p1.plot(x, pos_y,   pen=pen(C_BLUE, 2.5), name="pos_y")
    p1.plot(x, pos_d_y, pen=pen(C_ORANGE, 2.5, style=pg.QtCore.Qt.DashLine), name="pos_d_y")

    p2 = glw1.addPlot(row=1, col=0, title="Roll [deg]")
    style_plot(p2, "deg")
    p2.plot(x, roll,     pen=pen(C_BLUE, 2.5), name="roll")
    p2.plot(x, roll_d,   pen=pen(C_ORANGE, 2.5, style=pg.QtCore.Qt.DashLine), name="roll_d")
    p2.plot(x, roll_raw, pen=pen(C_GREEN, 2.0), name="roll_raw")

    p3 = glw1.addPlot(row=2, col=0, title="Roll Torque [Nm]")
    style_plot(p3, "Nm")
    p3.plot(x, tau_d_roll,      pen=pen(C_BLUE, 2.5),   name="tau_d_roll")
    p3.plot(x, tau_off_roll,    pen=pen(C_RED, 2.0),    name="tau_off_roll")
    p3.plot(x, tau_thrust_roll, pen=pen(C_PURPLE, 2.0), name="tau_thrust_roll")

    p2.setXLink(p1)
    p3.setXLink(p1)
    p1.setXRange(xmin, xmax, padding=0.0)

    # ==================================================
    # window 2 : thrust
    # ==================================================
    win2 = QtWidgets.QWidget()
    win2.setWindowTitle(f"thrust_view - {os.path.basename(path)}")
    win2.resize(1500, 950)

    layout2 = QtWidgets.QVBoxLayout(win2)
    info2 = QtWidgets.QLabel(info_text)
    layout2.addWidget(info2)

    glw2 = pg.GraphicsLayoutWidget()
    layout2.addWidget(glw2)

    p4 = glw2.addPlot(row=0, col=0, title=f"Thrust Sum [N]  LPF={FC_THRUST_SUM_HZ:.1f} Hz")
    style_plot(p4, "N")
    p4.plot(x, f12, pen=pen(C_BLUE, 2.5),   name="F1+F2")
    p4.plot(x, f34, pen=pen(C_ORANGE, 2.5), name="F3+F4")

    p5 = glw2.addPlot(row=1, col=0, title=f"|F34 - F12| [N]  LPF={FC_THRUST_SUM_HZ:.1f} Hz")
    style_plot(p5, "N")
    p5.plot(x, f_diff_abs, pen=pen(C_GREEN, 2.5), name="abs(F34-F12)")

    p6 = glw2.addPlot(row=2, col=0, title=f"Each Thrust [N]  LPF={FC_THRUST_SUM_HZ:.1f} Hz")
    style_plot(p6, "N")
    p6.plot(x, f1_each, pen=pen(C_BLUE, 2.0),   name="F1")
    p6.plot(x, f2_each, pen=pen(C_ORANGE, 2.0), name="F2")
    p6.plot(x, f3_each, pen=pen(C_GREEN, 2.0),  name="F3")
    p6.plot(x, f4_each, pen=pen(C_RED, 2.0),    name="F4")

    p5.setXLink(p4)
    p6.setXLink(p4)
    p4.setXRange(xmin, xmax, padding=0.0)

    win1.show()
    win2.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
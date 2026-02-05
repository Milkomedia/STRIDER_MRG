# STRIDER

### Features

- Trajectory-following simulation of the STRIDER
- MPC Reference Governor(MRG) & real-time view.
&nbsp;
- Flight Control: Geometric(SE(3)) Controller [[paper]](https://arxiv.org/pdf/1003.2005), [[reference]](https://fdcl-gwu.github.io/uav_geometric_control/)
- Control allocation: Sequential control allocation [[paper]](https://ieeexplore.ieee.org/document/11016760)
- Arm morphing: Acados NMPC software package [[github]](https://docs.acados.org)

---

### Dependencies
- C++ : MuJoCo / Eigen3 / GLFW3 / OpenGL
- Python3 : acados_template / pybind11 / pyqt / vispy
- Tested(ONLY): STRIDER ThinkStation

### Build (CMake)

```bash
# (in root dir)
mkdir -p bild && cd build
cmake ..
make -j
```

### execution
- Flight

```bash
cd build
./strider
```
- Real-time flight viewer

```bash
cd apps
python3 strider_logger.py
```
- Real-time MRG single-horizon viewer
```bash
cd resources/mpc_py
python3 mpc_viewer.py
```

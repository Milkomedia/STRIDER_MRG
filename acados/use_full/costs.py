import numpy as np

# h_expr constraint
F_MIN   = 8.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 35.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [N]

DELTA_MIN = -0.1745 * np.array([1.0, 1.0, 1.0/2.0]) # 10deg
DELTA_MAX =  0.1745 * np.array([1.0, 1.0, 1.0/2.0])

# input constraint
RHO_DOT_MIN    = -0.3 * np.array([1.0, 1.0, 1.0, 1.0]) # [m/s]
RHO_DOT_MAX    =  0.3 * np.array([1.0, 1.0, 1.0, 1.0]) # [m/s]

ALPHA_DOT_MIN  = -0.5 * np.array([1.0, 1.0, 1.0, 1.0]) # [rad/s]
ALPHA_DOT_MAX  =  0.5 * np.array([1.0, 1.0, 1.0, 1.0]) # [rad/s]

# input cost
Q_THETA = 2000.0 * np.array([1.0, 1.0, 40.0])

# thurst deviation cost
Q_FDEV  = 2.0 * np.array([1.0, 1.0, 1.0, 1.0])

# rate cost
c_a = 1.00  # arm cost (m/s scale)
c_b = 0.01 # base cost (rad/s scale)
R_ROTOR = 1.0 * np.array([c_a, c_b, c_a, c_b, c_a, c_b, c_a, c_b])
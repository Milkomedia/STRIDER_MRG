import numpy as np

# h_expr constraint
F_MIN   = 2.00  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 25.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [N] 23.0

RHO_DOT_MIN    = -0.1 * np.array([1.0, 1.0, 1.0, 1.0]) # [m/s]
RHO_DOT_MAX    =  0.1 * np.array([1.0, 1.0, 1.0, 1.0]) # [m/s]

ALPHA_DOT_MIN  = -0.2 * np.array([1.0, 1.0, 1.0, 1.0]) # [rad/s]
ALPHA_DOT_MAX  =  0.2 * np.array([1.0, 1.0, 1.0, 1.0]) # [rad/s]

# input constraint
DTHETA_MIN = -10.0 * np.array([1.0, 1.0, 0.5]) * np.pi / 180.0
DTHETA_MAX =  10.0 * np.array([1.0, 1.0, 0.5]) * np.pi / 180.0

# input cost
Q_THETA = 3980.0 * np.array([1.0, 1.0, 40.0])

# thurst deviation cost
Q_FDEV  = 1.0 * np.array([1.0, 1.0, 1.0, 1.0])

# h_expr cost
c_a = 1.00  # arm cost (m/s scale)
c_b = 0.10  # base cost (rad/s scale)
R_ROTOR = 200.0 * np.array([c_a, c_b, c_a, c_b, c_a, c_b, c_a, c_b])
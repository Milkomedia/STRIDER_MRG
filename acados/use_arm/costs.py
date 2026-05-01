import numpy as np

# h_expr constraint
F_MIN   = 2.00  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 43.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [N] 23.0

RHO_DOT_MIN    = -0.1 * np.array([1.0, 1.0, 1.0, 1.0]) # [m/s]
RHO_DOT_MAX    =  0.1 * np.array([1.0, 1.0, 1.0, 1.0]) # [m/s]

ALPHA_DOT_MIN  = -0.2 * np.array([1.0, 1.0, 1.0, 1.0]) # [rad/s]
ALPHA_DOT_MAX  =  0.2 * np.array([1.0, 1.0, 1.0, 1.0]) # [rad/s]

# thurst deviation cost
Q_FDEV  = 0.621 * np.array([1.0, 1.0, 1.0, 1.0])

# h_expr cost
c_a = 1.0  # arm cost (m/s scale)
c_b = 0.1 # base cost (rad/s scale)
R_ROTOR = 99.378 * np.array([c_a, c_b, c_a, c_b, c_a, c_b, c_a, c_b])
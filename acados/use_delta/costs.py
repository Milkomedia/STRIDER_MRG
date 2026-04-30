import numpy as np

# h_expr constraint
F_MIN   = 8.0  * np.array([1.0, 1.0, 1.0, 1.0]) # thrust bound (F1,F2,F3,F4)
F_MAX   = 27.0 * np.array([1.0, 1.0, 1.0, 1.0]) # [N]
# 24.5로 final 박스없이 실험함 

# input cost
Q_THETA = 1.0 * np.array([1.0, 1.0, 40.0])
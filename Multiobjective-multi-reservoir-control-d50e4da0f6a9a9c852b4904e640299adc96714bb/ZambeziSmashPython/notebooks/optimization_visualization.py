import numpy as np
import hvwfg

# Fitness vector assuming minimization
obj = np.array([[0.3, 0.6],
                [0.4, 0.4],
                [0.6, 0.2]])

ref = np.array([1.1, 1.1])

hvwfg.wfg(obj, ref)
import math

import numpy as np

choose2 = np.vectorize(lambda x: math.comb(x, 2))
"""
numpy-vectorized choose(n, 2)
"""

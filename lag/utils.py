import math

import numpy as np
from numpy.typing import NDArray

choose2 = np.vectorize(lambda x: math.comb(x, 2))
"""
numpy-vectorized choose(n, 2)
"""


def is_nondecreasing(x: NDArray):
    return (x[:-1] <= x[1:]).all()

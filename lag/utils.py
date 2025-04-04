import math
from itertools import groupby

import numpy as np

choose2 = np.vectorize(lambda x: math.comb(x, 2))
"""
numpy-vectorized choose(n, 2)
"""


def rle_vals(x) -> list:
    """
    Equivalent to rle(x)$values in R
    """
    return [val for val, _ in groupby(x)]

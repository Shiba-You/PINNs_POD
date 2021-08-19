import numpy as np
np.random.seed(12345)

def test(a):
    k = np.random.choice(a, 3)
    print(k)
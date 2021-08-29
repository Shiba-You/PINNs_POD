# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
sys.path.append("../../utils")
import warnings
warnings.simplefilter('ignore')
import importlib


from pinns import PhysicsInformedNN
from make_data import make_data, make_noisy_data
from send_line import send_line
from make_results import make_results


import numpy as np
import tensorflow as tf
import time


# %%
# importlib.reload()

# %%
'''
mode_th : 各モード諸条件の上位何%を教師データ候補群とするのか
'''

pro = "asymmetric_squares"
path = "../../../data/{}/".format(pro)

layers = [3] + 8 * [20] + [2]
Itration = 1*10**5
rs = 1234
N_train = .001

ns_lv = .2
alpha = .75

n_modes = 0
subject = "basicNoisy"
mode_th = .05


snap = [10]
debug = False
np.random.seed(rs)
tf.set_random_seed(rs)


# %%
X_star, x_train, y_train, t_train, u_train, v_train, TT, UU, VV, PP = make_noisy_data(path=path, ns_lv=ns_lv, N_train=N_train, n_modes=n_modes, subject=subject, mode_th=mode_th)


# %%
t1 = time.time()
model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers, debug, alpha)
model.train(Itration)
t2 = time.time()
elps = (t2 - t1) / 60.
print("elps:", elps)
send_line('解析終了')


# %%
make_results(pro, subject, model, X_star, TT, snap, UU, VV, PP, n_modes, mode_th, N_train, Itration, elps, rs, ns_lv)


# %%




# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%


# %%
get_ipython().run_line_magic('load_ext', 'autoreload')


# %%
import os
import sys
sys.path.append("../../utils")
import warnings
warnings.simplefilter('ignore')

from make_images import test_image, test_images
from make_data import make_data

import numpy as np
import tensorflow as tf
import time


# %%
get_ipython().run_line_magic('autoreload', '')

# %%

pro = "04_circle_water"
# pro = "03_asymmetric_squares"
path = "../../../data/{}/".format(pro)

data_form = False

layers = [3] + 8 * [20] + [2]
Itration = 2*10**5
rs = 1234
N_train = .001

ns_lv = 0

n_modes = 0
subject = "UU"
mode_th = .05


snap = [10]
debug = False
np.random.seed(rs)
tf.set_random_seed(rs)


# %%

# 分布図の単体を生成
X_star, x_train, y_train, t_train, u_train, v_train, TT, UU, VV, PP = make_data(path=path, ns_lv=ns_lv, N_train=N_train, n_modes=n_modes, subject=subject, mode_th=mode_th, data_form=data_form)

x_star = X_star[:,0:1]
y_star = X_star[:,1:2]
t_star = TT[:,snap]

u_star = UU[:,snap]
v_star = VV[:,snap]
p_star = PP[:,snap]

print(X_star.shape)
print(u_star.shape)
print(v_star.shape)
print(p_star.shape)

test_images(X_star, snap, "../../../output/{}_test".format(pro), u_star, v_star, p_star)


# %%

# ノイズありの画像・動画生成
for ns_lv in ns_lvs:
    X_star, x_train, y_train, t_train, u_train, v_train, TT, UU, VV, PP = make_noisy_data(path=path, ns_lv=ns_lv, N_train=N_train, n_modes=n_modes, subject=subject, mode_th=mode_th)

    for snap in range(200):
        x_star = X_star[:,0:1]
        y_star = X_star[:,1:2]
        t_star = TT[:,snap]

        u_star = UU[:,snap]
        v_star = VV[:,snap]
        p_star = PP[:,snap]
    
        dir_name_u = "../../../output/{}_{}_test_u".format(pro, ns_lv)
        dir_name_v = "../../../output/{}_{}_test_v".format(pro, ns_lv)
        if not os.path.exists(dir_name_u):#ディレクトリがなかったら
            os.mkdir(dir_name_u)#作成したいフォルダ名を作成
        if not os.path.exists(dir_name_v):#ディレクトリがなかったら
            os.mkdir(dir_name_v)#作成したいフォルダ名を作成
        
        test_image(X_star, snap, "../../../output/test/{}_{}_u".format(pro, ns_lv), u_star)
        test_image(X_star, snap, "../../../output/test/{}_{}_v".format(pro, ns_lv), v_star)
# %%

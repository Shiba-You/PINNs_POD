import pandas as pd
import numpy as np

def make_data(path, N_train):
    U_star, P_star, Vor_star = read_data(path)
    X_star, t_star, N, T     = init_data()
        
    '''
    XX      = (N, T)      ,  YY      = (N, T)      ,  TT      = (N, T)      ,  UU      = (N, T)      ,  VV      = (N, T)      ,  PP      = (N, T)      
    x       = (N*T, 1)    ,  y       = (N*T, 1)    ,  t       = (N*T, 1)    ,  u       = (N*T, 1)    ,  v       = (N*T, 1)    ,  p       = (N*T, 1)    
    x_train = (N_train, 1),  y_train = (N_train, 1),  t_train = (N_train, 1),  u_train = (N_train, 1),  v_train = (N_train, 1),  p_train = (N_train, 1)
    '''

    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T
    UU = U_star[:,0].reshape(T, N).T            # N x T X軸方向速度
    VV = - U_star[:,1].reshape(T, N).T        # N x T Y軸方向速度
    PP = P_star.reshape(T, N).T 

    x = XX.flatten()[:,None] # NT x 1
    y = YY.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1
    u = UU.flatten()[:,None] # NT x 1
    v = VV.flatten()[:,None] # NT x 1
    p = PP.flatten()[:,None] # NT x 1

    idx = np.random.choice(N*T, int(N_train * N * T), replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    t_train = t[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]

    log(x_train, y_train, t_train, u_train, v_train)

    return X_star, x_train, y_train, t_train, u_train, v_train, TT, UU, VV, PP


def read_data(path):
    U_star = pd.read_csv(path + 'U_arrange_data.csv', header = None, sep = '\s+').values
    P_star = pd.read_csv(path + 'p_arrange_data.csv', header = None, sep = '\s+').values
    Vor_star = pd.read_csv(path + 'vorticity_arrange_data.csv', header = None, sep = '\s+').values
    return U_star, P_star, Vor_star

def init_data():
    t0, t1, dt =  0, 20, .1
    t_star = np.arange(t0, t1, dt)
    t_star = t_star.reshape(-1, 1)

    x0, x1, nx =  1,  8, 99
    x_star, dx = np.linspace(x0, x1, nx, retstep=True)   # true domain: [1.075, 7.950]

    y0, y1, ny = -2,  2, 50
    y_star, dy = np.linspace(y0, y1, ny, retstep=True)   # true domain: [-1.942, 1.943]

    x_star, y_star = np.meshgrid(x_star, y_star)
    x_star, y_star = x_star.reshape(-1, 1), y_star.reshape(-1, 1)
    X_star = np.c_[x_star, y_star]

    N = X_star.shape[0]
    T = t_star.shape[0]

    return X_star, t_star, N, T

def log(x_train, y_train, t_train, u_train, v_train):
    print("x_train : ", x_train.shape)
    print("y_train : ", y_train.shape)
    print("t_train : ", t_train.shape)
    print("u_train : ", u_train.shape)
    print("v_train : ", v_train.shape)
import os
import datetime
from make_images import make_image
import numpy as np 

def make_results(train_rate, model, X_star, TT, snap, UU, VV, PP, N_train, Itration, elps, rs):
    dir_name = make_dir(train_rate)
    u_pred, v_pred, p_pred, u_star, v_star, p_star, error_u, error_v, error_p, error_lambda_1, error_lambda_2, N, T = model_pred(model, X_star, TT, snap, UU, VV, PP)
    train_data = N * T * N_train
    make_info(dir_name, Itration, train_rate, elps, rs, error_u, error_v, error_p, error_lambda_1, error_lambda_2, train_data)
    make_image(X_star, snap, dir_name, u_pred, v_pred, p_pred, u_star, v_star, p_star)


def make_dir(train_rate):
    dir_name = "Basic_{}_{}".format(train_rate, str(datetime.date.today()))
    if not os.path.exists(dir_name):#ディレクトリがなかったら
        os.mkdir(dir_name)#作成したいフォルダ名を作成
    return dir_name

def model_pred(model, X_star, TT, snap, UU, VV, PP):
    x_star = X_star[:,0:1]
    y_star = X_star[:,1:2]
    t_star = TT[:,snap]

    u_star = UU[:,snap]
    v_star = VV[:,snap]
    p_star = PP[:,snap]
    
    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
    lambda_1_value = model.sess.run(model.lambda_1)
    lambda_2_value = model.sess.run(model.lambda_2)

    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)
    error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
    error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100

    N = X_star.shape[0]
    T = t_star.shape[0]

    return u_pred, v_pred, p_pred, u_star, v_star, p_star, error_u, error_v, error_p, error_lambda_1, error_lambda_2, N, T

def make_info(dir_name, Itration, train_rate, elps, rs, error_u, error_v, error_p, error_lambda_1, error_lambda_2, train_data):

    f = open(dir_name+"/info.txt", "a", encoding="UTF-8")
    f.write("\n")
    f.write("********************************\n")
    f.write("********************************\n")
    f.write("Date       : {} \n".format(str(datetime.date.today())))
    f.write("Itration   : {} \n".format(Itration))
    f.write("train rate : {} \n".format(train_rate))
    f.write("train data : {} \n".format(train_data))
    f.write("Time       : {} \n".format(elps))
    f.write("seed       : {} \n".format(rs))
    f.write("================================\n")
    f.write("============Error===============\n")
    f.write('Error u  : %e \n' % (error_u))    
    f.write('Error v  : %e \n' % (error_v))    
    f.write('Error p  : %e \n' % (error_p))   
    f.write('Error l1 : %.5f%% \n' % (error_lambda_1))                             
    f.write('Error l2 : %.5f%% \n' % (error_lambda_2))   
    f.write("********************************\n")
    f.close()

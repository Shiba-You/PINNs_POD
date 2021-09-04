import numpy as np
import tensorflow as tf

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, y, t, u, v, layers, debug, alpha=.5):
        self.debug = debug

        X = np.concatenate([x, y, t], 1)
        
        self.lb = X.min(0); self.ub = X.max(0)
        self.X = X
        self.x = X[:,0:1]; self.y = X[:,1:2]; self.t = X[:,2:3]
        
        self.u = u; self.v = v
        
        self.layers = layers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers) 

        # track loss values
        self.loss_log      = np.array([])
        self.loss_pred_log = np.array([])
        self.loss_phys_log = np.array([])       
        
        # Initialize parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)

        # track parameter estimation
        self.rhoi_log = np.array([])       
        self.nu_log = np.array([])       
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf.placeholder(tf.float32, shape=[None, self.v.shape[1]])
        
        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred = self.net_NS(self.x_tf, self.y_tf, self.t_tf)
        
        self.loss_pred = alpha       * (tf.reduce_sum(tf.square(self.u_tf - self.u_pred)) + tf.reduce_sum(tf.square(self.v_tf - self.v_pred)))
        self.loss_phys = (1 - alpha) * (tf.reduce_sum(tf.square(self.f_u_pred)) + tf.reduce_sum(tf.square(self.f_v_pred)))
        self.loss      = self.loss_pred + self.loss_phys


                    
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})            
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def net_NS(self, x, y, t):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        psi_and_p = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
        psi = psi_and_p[:,0:1]
        p = psi_and_p[:,1:2]
        
        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]  
        
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        
        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        
        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        f_u = u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy)
        f_v = v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)

        return u, v, p, f_u, f_v

    def callback(self, loss, loss_pred, loss_phys):
        if self.debug:
            print('loss: %.6e, loss_pred: %.6e, loss_phys: %.6e' % (loss, loss_pred, loss_phys))
        else:
            print(".", end="")

    def train(self, nIter, tol=1e-5): 
        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.t_tf: self.t,
                   self.u_tf: self.u, self.v_tf: self.v}

        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            self.loss_pred_log = np.append(self.loss_pred_log, np.array([self.sess.run(self.loss_pred, tf_dict)]))
            self.loss_phys_log = np.append(self.loss_phys_log, np.array([self.sess.run(self.loss_phys, tf_dict)]))
            self.loss_log      = np.append(self.loss_log,      np.array([self.sess.run(self.loss,      tf_dict)]))

            self.rhoi_log      = np.append(self.rhoi_log,      np.array([self.sess.run(self.lambda_1,      tf_dict)]))
            self.nu_log        = np.append(self.nu_log,        np.array([self.sess.run(self.lambda_2,      tf_dict)]))

            if it % 10 == 0:
                loss_pred_val  = self.sess.run(self.loss_pred, tf_dict)
                loss_phys_val  = self.sess.run(self.loss_phys, tf_dict)
                loss_val       = self.sess.run(self.loss,      tf_dict)

                lambda_1_value = self.sess.run(self.lambda_1)
                lambda_2_value = self.sess.run(self.lambda_2)
                if self.debug:
                    print('it: %d, loss: %.6e, loss_pred: %.6e, loss_phys: %.6e' % (it, loss_val, loss_pred_val, loss_phys_val))
                elif it % 200 == 0:
                    print(".\n", end="")
                else:
                    print(".", end="")
            if loss_val < tol:
                print(">>>>> program terminating with the loss reaching its tolerance.")
                break
                
        
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss, self.lambda_1, self.lambda_2],
                                loss_callback = self.callback)
    
    def predict(self, x_star, y_star, t_star):
        
        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.t_tf: t_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        
        return u_star, v_star, p_star
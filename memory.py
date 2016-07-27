import numpy as np
import tensorflow as tf

class Memory:
     def __init__(self, shape = [3, 224, 224], lstm_size = 512, N = 14, T = 16):
         self.shape = shape
         self.lstm_size = lstm_size
         self.T = T
         self.N = N
         self.att_size = N * shape[0] * N
         self.share = None

     def linear(self, x,output_dim):
         """
         affine transformation Wx+b
         assumes x.shape = (batch_size, num_features)
         """
         w=tf.get_variable("w", [x.get_shape()[1], output_dim], initializer=tf.constant_initializer(0.0)) 
         b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
         return tf.matmul(x,w)+b

     def cal_sensor(self, lstm_h, scope, eps=1e-8):
         N = self.N
         with tf.variable_scope(scope,reuse=self.share):
              params=self.linear(lstm_h,5)
         gx_,gy_,log_sigma2,log_delta,log_gamma=tf.split(1,5,params)
         gx=(self.shape[1] + 1.0) / 2.0 * (gx_ + 1)
         gy=(self.shape[2] + 1.0) / 2.0 * (gy_ + 1)
         sigma2=tf.exp(log_sigma2)
         delta=(max(self.shape[1],self.shape[2]) - 1) * 1.0 / (N - 1) * tf.exp(log_delta)
         grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
         mu_x = gx + (grid_i - N / 2.0 - 0.5) * delta # batch_size * N
         mu_y = gy + (grid_i - N / 2.0 - 0.5) * delta # 
         a = tf.reshape(tf.cast(tf.range(self.shape[1]), tf.float32), [1, 1, -1]) # 1 * 1 * A
         b = tf.reshape(tf.cast(tf.range(self.shape[2]), tf.float32), [1, 1, -1])
         mu_x = tf.reshape(mu_x, [-1, N, 1]) # batch_size * N * 1
         mu_y = tf.reshape(mu_y, [-1, N, 1])
         sigma2 = tf.reshape(sigma2, [-1, 1, 1]) # batch_size * 1 * 1
         Fx = tf.exp(-tf.square(a - mu_x) / (2*sigma2)) # 2*sigma2?
         Fy = tf.exp(-tf.square(b - mu_y) / (2*sigma2)) # batch x N x B
         # normalize, sum over A and B dims
         Fx=Fx / tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
         Fy=Fy / tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
         return Fx,Fy,tf.exp(log_gamma)

     def read(self, x, Fx, Fy, gamma):
        Fxr = tf.reshape(Fx, [-1, 1, self.N, self.shape[1]])
        Fyr = tf.reshape(Fy, [-1, 1, self.N, self.shape[2]])
        Fxr3 = tf.concat(1, [Fxr, Fxr, Fxr]) # batch * 3 * N * A
        Fyr3 = tf.concat(1, [Fyr, Fyr, Fyr])
        Fxt3 = tf.transpose(Fxr3, perm=[0, 1, 3, 2])
        glimpse = tf.batch_matmul(Fyr3, tf.batch_matmul(x, Fxt3))
        glimpse = tf.reshape(glimpse, [-1, self.att_size])
        return glimpse * tf.reshape(gamma, [-1,1])

     def write(self, lstm_h, Fx, Fy, gamma):
          with tf.variable_scope("writeW",reuse=self.share):
               w = self.linear(lstm_h, self.N * self.N) # batch x (write_n*write_n)
          w = tf.reshape(w, [-1, self.N, self.N])
          Fyt = tf.transpose(Fy, perm=[0, 2, 1])
          wr = tf.batch_matmul(Fyt, tf.batch_matmul(w, Fx))
          return wr*tf.reshape(1.0/gamma, [-1,1,1])

     def build(self, images, batch_size, debug=False, train=True):
          images = tf.transpose(images, (0, 3, 1, 2)) # batch_size * c * h * w
          mems = [0] * (self.T + 1)
          lstm = tf.nn.rnn_cell.LSTMCell(self.lstm_size, state_is_tuple=True)
          lstm_h = tf.zeros((batch_size, self.lstm_size))
          lstm_state=lstm.zero_state(batch_size, tf.float32)
          if train:
               self.share = None
          else:
               self.share = True

          for i in range(self.T):
               Fx,Fy,gamma = self.cal_sensor(lstm_h, "sensor")
               r = self.read(images, Fx, Fy, gamma)
               with tf.variable_scope("lstm",reuse=self.share):
                    lstm_h, lstm_state = lstm(r, lstm_state)
               mems[i+1] = mems[i] + self.write(lstm_h, Fx, Fy, gamma)
               self.share = True
          mems_last = tf.reshape(mems[-1], (batch_size, self.shape[1], self.shape[2], 1))
          return mems_last
     def use_memory(self, images, mems, num_classes, kernal_size, wd = 0, train=True):
        if train:
          reuse = None
        else:
          reuse = True
        images_with_mems = tf.concat(3, [images, mems])
        with tf.variable_scope("cal_with_mem", reuse=reuse) as scope:
          initializer = tf.truncated_normal_initializer(stddev=0.0001)
          var = tf.get_variable('weights', shape=[kernal_size, kernal_size, num_classes+1, num_classes],
                                initializer=initializer)

          if wd and (not tf.get_variable_scope().reuse):
              weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
              tf.add_to_collection('losses', weight_decay)
          conv = tf.nn.conv2d(images_with_mems, var, [1, 1, 1, 1], padding='SAME')

          initializer = tf.constant_initializer(0.0)
          conv_biases = tf.get_variable(name='biases', shape=[num_classes],
                               initializer=initializer)
          bias = tf.nn.bias_add(conv, conv_biases)
          return tf.add(bias, images)
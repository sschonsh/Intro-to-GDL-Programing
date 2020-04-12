import tensorflow as tf
import numpy as np


def weight_var(shape):
    #initial = tf.truncated_normal(shape=shape, stddev=1)
    initial = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial(shape))


def fc_var(shape):
    #initial = tf.truncated_normal(shape=shape, stddev=1)
    initial = tf.contrib.layers.xavier_initializer()
    return tf.Variable(initial(shape))
    
def create_bias(shape, n_pts):
    bias = weight_var([shape])
    bias = tf.reshape(bias, [1, -1])
    bias = tf.tile(bias, [n_pts, 1])
    return bias


class Siam:

    #Create model
    def __init__(self, n_pts, n_feats, n_channel, n_ker, ker_size):
        #Hyper Parameters
        self.n_pts     = n_pts
        self.n_feats   = n_feats
        self.n_channel = n_channel
        self.n_ker     = n_ker
        #Parameters
        self.mu     = tf.placeholder(tf.float32)
        self.gamma1 = tf.placeholder(tf.float32)
        self.gamma2 = tf.placeholder(tf.float32)

		#Variables
        self.ker1 = tf.transpose(weight_var([self.n_channel * self.n_ker[1], ker_size]))
        self.ker2 = tf.transpose(weight_var([self.n_ker[1] * self.n_ker[2], ker_size]))
        self.ker3 = tf.transpose(weight_var([self.n_ker[2] * self.n_ker[3], ker_size]))
        self.ker4 = tf.transpose(weight_var([self.n_ker[3] * self.n_ker[4], ker_size])) 
        
        self.ker5 = tf.transpose(weight_var([self.n_channel * self.n_ker[1], ker_size]))
        self.ker6 = tf.transpose(weight_var([self.n_ker[1] * self.n_ker[2], ker_size]))
        self.ker7 = tf.transpose(weight_var([self.n_ker[2] * self.n_ker[3], ker_size]))
        self.ker8 = tf.transpose(weight_var([self.n_ker[3] * self.n_ker[4], ker_size])) 
        
        self.bias1 = create_bias(self.n_ker[1], self.n_pts)
        self.bias2 = create_bias(self.n_ker[2], self.n_pts) 
        self.bias3 = create_bias(self.n_ker[3], self.n_pts)
        self.bias4 = create_bias(self.n_ker[4], self.n_pts)
        
        self.bias5 = create_bias(self.n_ker[1], self.n_pts)
        self.bias6 = create_bias(self.n_ker[2], self.n_pts)
        self.bias7 = create_bias(self.n_ker[3], self.n_pts)
        self.bias8 = create_bias(self.n_ker[4], self.n_pts)
 
        #Placeholders
        self.x1 = tf.placeholder(tf.float32, shape=[self.n_pts, self.n_feats])
        self.x2 = tf.placeholder(tf.float32, shape=[self.n_pts, self.n_feats])
        self.x3 = tf.placeholder(tf.float32, shape=[self.n_pts, self.n_feats])
        
        self.mass1_1 = tf.sparse_placeholder(tf.float32)
        self.mass1_2 = tf.sparse_placeholder(tf.float32)
        self.mass1_3 = tf.sparse_placeholder(tf.float32)
        self.mass2_1 = tf.sparse_placeholder(tf.float32)
        self.mass2_2 = tf.sparse_placeholder(tf.float32)
        self.mass2_3 = tf.sparse_placeholder(tf.float32)
        self.mass3_1 = tf.sparse_placeholder(tf.float32)
        self.mass3_2 = tf.sparse_placeholder(tf.float32)
        self.mass3_3 = tf.sparse_placeholder(tf.float32)
        
        self.weight1_1 = tf.sparse_placeholder(tf.float32)
        self.weight1_2 = tf.sparse_placeholder(tf.float32)
        self.weight1_3 = tf.sparse_placeholder(tf.float32)        
        self.weight2_1 = tf.sparse_placeholder(tf.float32)
        self.weight2_2 = tf.sparse_placeholder(tf.float32)
        self.weight2_3 = tf.sparse_placeholder(tf.float32)
        self.weight3_1 = tf.sparse_placeholder(tf.float32)
        self.weight3_2 = tf.sparse_placeholder(tf.float32)
        self.weight3_3 = tf.sparse_placeholder(tf.float32)
        
        self.ind1_1  = tf.placeholder(tf.int64)
        self.ind1_2  = tf.placeholder(tf.int64)
        self.ind1_3  = tf.placeholder(tf.int64)
        self.ind2_1  = tf.placeholder(tf.int64)
        self.ind2_2  = tf.placeholder(tf.int64)
        self.ind2_3  = tf.placeholder(tf.int64)
        self.ind3_1  = tf.placeholder(tf.int64)
        self.ind3_2  = tf.placeholder(tf.int64)
        self.ind3_3  = tf.placeholder(tf.int64)
        
        self.mv1_1  = tf.placeholder(tf.int64)
        self.mv1_2  = tf.placeholder(tf.int64)
        self.mv1_3  = tf.placeholder(tf.int64)
        self.mv2_1  = tf.placeholder(tf.int64)
        self.mv2_2  = tf.placeholder(tf.int64)
        self.mv2_3  = tf.placeholder(tf.int64)
        self.mv3_1  = tf.placeholder(tf.int64)
        self.mv3_2  = tf.placeholder(tf.int64)
        self.mv3_3  = tf.placeholder(tf.int64)
        
        self.ds1_1  = tf.placeholder(tf.int64)
        self.ds1_2  = tf.placeholder(tf.int64)
        self.ds1_3  = tf.placeholder(tf.int64)
        self.ds2_1  = tf.placeholder(tf.int64)
        self.ds2_2  = tf.placeholder(tf.int64)
        self.ds2_3  = tf.placeholder(tf.int64)
        self.ds3_1  = tf.placeholder(tf.int64)
        self.ds3_2  = tf.placeholder(tf.int64)
        self.ds3_3  = tf.placeholder(tf.int64)
        
        self.l1_1 = tf.placeholder(tf.int32)
        self.l1_2 = tf.placeholder(tf.int32)
        self.l1_3 = tf.placeholder(tf.int32)
        self.l2_1 = tf.placeholder(tf.int32)
        self.l2_2 = tf.placeholder(tf.int32)
        self.l2_3 = tf.placeholder(tf.int32)
        self.l3_1 = tf.placeholder(tf.int32)
        self.l3_2 = tf.placeholder(tf.int32)
        self.l3_3 = tf.placeholder(tf.int32)


        #network connections
        self.x1int = self.intrinsic(self.x1, self.mass1_1, self.weight1_1, self.        ind1_1, self.mv1_1, self.ds1_1, self.l1_1,
                 self.mass1_2, self.weight1_2, self.ind1_2, self.mv1_2, self.ds1_2, self.l1_2,
                 self.mass1_3, self.weight1_3, self.ind1_3, self.mv1_3, self.ds1_3, self.l1_3)
        self.x1ext = self.extrinsic(self.x1, self.mass1_1, self.weight1_1, self.        ind1_1, self.mv1_1, self.ds1_1, self.l1_1,
                 self.mass1_2, self.weight1_2, self.ind1_2, self.mv1_2, self.ds1_2, self.l1_2,
                 self.mass1_3, self.weight1_3, self.ind1_3, self.mv1_3, self.ds1_3, self.l1_3)

        
        self.x2int = self.intrinsic(self.x2, self.mass2_1, self.weight2_1, self.        ind2_1, self.mv2_1, self.ds2_1, self.l2_1,
                 self.mass2_2, self.weight2_2, self.ind2_2, self.mv2_2, self.ds2_2, self.l2_2,
                 self.mass2_3, self.weight2_3, self.ind2_3, self.mv2_3, self.ds2_3, self.l2_3)
        self.x2ext = self.extrinsic(self.x2, self.mass2_1, self.weight2_1, self.        ind2_1, self.mv2_1, self.ds2_1, self.l2_1,
                 self.mass2_2, self.weight2_2, self.ind2_2, self.mv2_2, self.ds2_2, self.l2_2,
                 self.mass2_3, self.weight2_3, self.ind2_3, self.mv2_3, self.ds2_3, self.l2_3)
        
        self.x3int = self.intrinsic(self.x3, self.mass3_1, self.weight3_1, self.        ind3_1, self.mv3_1, self.ds3_1, self.l3_1,
                 self.mass3_2, self.weight3_2, self.ind3_2, self.mv3_2, self.ds3_2, self.l3_2,
                 self.mass3_3, self.weight3_3, self.ind3_3, self.mv3_3, self.ds3_3, self.l3_3)
        self.x3ext = self.extrinsic(self.x3, self.mass3_1, self.weight3_1, self.        ind3_1, self.mv3_1, self.ds3_1, self.l3_1,
                 self.mass3_2, self.weight3_2, self.ind3_2, self.mv3_2, self.ds3_2, self.l3_2,
                 self.mass3_3, self.weight3_3, self.ind3_3, self.mv3_3, self.ds3_3, self.l3_3)
        
        self.intloss = tf.norm(self.x1int-self.x2int)**2+tf.nn.relu(margin-tf.norm(self.x1int - self.x3int)**2)
        self.extloss = tf.norm(self.x1ext-self.x3ext)**2+tf.nn.relu(margin-tf.norm(self.x1ext - self.x2ext)**2)
        self.TotalLoss = self.intloss +self.extloss

    def intrinsic(self, x, mass_1, weight_1, ind_1, mv_1, ds_1, l_1, 
            mass_2, weight_2, ind_2, mv_2, ds_2, l_2,
            mass_3, weight_3, ind_3, mv_3, ds_3, l_3):

        f1 = self.surface_conv(x,  mass_1, weight_1, ind_1, mv_1, ds_1, l_1, self.ker1, self.bias1, self.n_channel, self.n_ker[0])
        f2 = self.surface_conv(f1, mass_2, weight_2, ind_2, mv_2, ds_2, l_2, self.ker2, self.bias2, self.n_channel, self.n_ker[1])
        f3 = self.surface_conv(f2, mass_3, weight_3, ind_3, mv_3, ds_3, l_3, self.ker3, self.bias3, self.n_channel, self.n_ker[2])
        f4 = self.surface_conv(f3, mass_3, weight_3, ind_3, mv_3, ds_3, l_3, self.ker4, self.bias4, self.n_channel, self.n_ker[4])
        print(f4)
        f5 = tf.flatten(f4)
        print(f5)
        return f4
        
    def extrinsic(self, x, mass_1, weight_1, ind_1, mv_1, ds_1, l_1, 
            mass_2, weight_2, ind_2, mv_2, ds_2, l_2,
            mass_3, weight_3, ind_3, mv_3, ds_3, l_3):

        f1 = self.surface_conv(x,  mass_1, weight_1, ind_1, mv_1, ds_1, l_1, self.ker5, self.bias5, self.n_channel, self.n_ker[0])
        f2 = self.surface_conv(f1, mass_2, weight_2, ind_2, mv_2, ds_2, l_2, self.ker6, self.bias6, self.n_channel, self.n_ker[1])
        f3 = self.surface_conv(f2, mass_3, weight_3, ind_3, mv_3, ds_3, l_3, self.ker7, self.bias7, self.n_channel, self.n_ker[2])
        f4 = self.surface_conv(f3, mass_3, weight_3, ind_3, mv_3, ds_3, l_3, self.ker8, self.bias8, self.n_channel, self.n_ker[3])
        return f4

    def surface_conv(self, x, mass, weight, ind, mv, ds, l, kernel, bias, input_dim, output_size):
        z = tf.sparse_tensor_dense_matmul(mass, x)
        z_long  = tf.reshape(tf.transpose(z), [self.n_pts * input_dim ])
        val     = tf.gather(z_long, mv, 0)
        zmr     = tf.SparseTensor(indices=ind, values=val, dense_shape=ds)
        wk      = tf.transpose(tf.sparse_tensor_dense_matmul(weight, kernel))
        wk_long = tf.reshape(wk, [l * input_dim, output_size])
        c_long  = tf.sparse_tensor_dense_matmul(zmr, wk_long)
        c_full  = tf.reshape(c_long, [input_dim, self.n_pts, output_size])
        f = tf.reduce_sum(c_full, 0) + bias
        return tf.nn.elu(f)
        
    def surface_conv2(self, x, mass, weight, ind, mv, ds, l, kernel, bias, input_dim, output_size):
        z = tf.sparse_tensor_dense_matmul(mass, tf.nn.elu(x))
        z_long  = tf.reshape(tf.transpose(z), [self.n_pts * input_dim ])
        val     = tf.gather(z_long, mv, 0)
        zmr     = tf.SparseTensor(indices=ind, values=val, dense_shape=ds)
        wk      = tf.transpose(tf.sparse_tensor_dense_matmul(weight, kernel))
        wk_long = tf.reshape(wk, [l * input_dim, output_size])
        c_long  = tf.sparse_tensor_dense_matmul(zmr, wk_long)
        c_full  = tf.reshape(c_long, [input_dim, self.n_pts, output_size])
        f = tf.transpose(tf.reduce_sum(c_full, 2)) + bias
        return tf.nn.elu(f)    
        
    def surface_norm_reg(self,f,mass):
        mat = tf.matmul(tf.transpose(f),tf.sparse_tensor_dense_matmul(mass, f))
        vec = tf.linalg.diag_part(mat)
        fout = tf.multiply(1/vec,f)
        return fout 
        
    
    def loss_func(self):
        loss1 = tf.reduce_sum(tf.norm((self.out1-self.out2),axis = 0)**2)
        loss2 = tf.reduce_sum(tf.nn.relu(self.mu - tf.norm(self.out1 - tf.matmul(self.perm1,self.out2), axis = 0))**2) 
        loss3 = tf.reduce_sum(tf.nn.relu(self.mu - tf.norm(self.out1 - tf.matmul(self.out2,self.perm2), axis = 0))**2) 
       # loss  = self.gamma1*loss1 + self.gamma2*(loss2+loss3)
       #soft correspondence loss
        Funmap = tf.matmul(self.out1,tf.transpose(self.out2))
        Spread = tf.math.l2_normalize(tf.sparse_tensor_dense_matmul(self.Stiff,Funmap),axis = 1)
        SoftLoss = tf.norm(tf.multiply(Spread,self.Dist))**2
        loss = SoftLoss + loss1+loss2+loss3
        
        
        return loss, loss1, loss2, loss3
        
        
        
        
        
        
        
        
        
        
        

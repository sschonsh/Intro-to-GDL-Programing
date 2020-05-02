#exec(open("E:/Dropbox/Research/PTCnet/TFMNIST/TFMNIST2.py").read()) 


#User Parameters###############################################################################################################

max_its    = 5000 #iterations
alpha      = 1e-3 #learning rate 
nChannel   = 1   #number channels to output
nKer1      = 16   #number of Kernels in first layers
gamma1     = .1#regulaization

report =  50#How often to report loss

#Import packages###################################################################################################################
import os
import numpy as np
import pickle 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf 
import tensorflow.keras
import scipy.sparse as spsp
import scipy.io as spio
import time
from tensorflow.python.keras.layers import Lambda

# Helper Funcitons##########################################################################################################################################################################################################################

def weight_var(shape):
	initial = tf.random.truncated_normal(shape = shape, stddev = .1)
	return tf.Variable(initial)

def FC_var(shape):
	initial = tf.random.truncated_normal(shape = shape, stddev = .1)
	return tf.Variable(initial)
	
def Bias_var(shape):
	initial = tf.constant(0.01, shape=shape)
	return tf.Variable(initial)

def Cell_to_Sparse(C,flag):
	#Unpack into tensor
	if flag == 1:
		Ctemp = C.tocoo()
		Ctemp = Ctemp.transpose()
	else:
		Ctemp = C.tocoo()
	indices = np.mat([Ctemp.row, Ctemp.col]).transpose()
	return tf.SparseTensor(indices, np.float32(Ctemp.data), Ctemp.shape) 

def WtoTensor(W):
	WF  = W[1]
	Wcoo = WF.tocoo()
	indices = np.mat([Wcoo.row, Wcoo.col]).transpose()
	return tf.SparseTensor(indices, Wcoo.data, Wcoo.shape)
	
def MtoTensor(W):
	Wcoo = (2*W).tocoo()
	indices = np.mat([Wcoo.row, Wcoo.col]).transpose()
	return tf.SparseTensor(indices, Wcoo.data, Wcoo.shape)
	
def sigmoid(x):
	return 1 / (1 + np.exp(-x))
    
####Load data####################################################################################################################
from tensorflow.keras.datasets import mnist
(x_train_im, y_train), (x_test_im, y_test) = mnist.load_data()
x_train = x_train_im.astype('float32') / 255.
x_test  = x_test_im.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test  = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


## Load Manifold and WeightMatricies#################################################################################
mat1   = spio.loadmat('E:/Dropbox/Research/PTCnet/TFMNIST/ExSurf.mat', squeeze_me=True)
#mat1   = spio.loadmat('C:/Users/sscho/Dropbox/Research/PTCnet/TFMNIST/ExSurf.mat', squeeze_me=True)
surf1  = mat1['Surf']
W1cell = surf1['Wfull1'][()] 
W1mat  = Cell_to_Sparse(W1cell,0)
R1cell = surf1['Rfull1'][()]
R1mat  = Cell_to_Sparse(R1cell,1)
L1mat  = surf1['L1'][()]
I1val  = np.asarray(surf1['Ifull1'][()])-1
I1RowIndex  = np.asarray(surf1['IRowIndex1'][()])-1#minus 1 for 0 indexing
I1ColIndex  = np.asarray(surf1['IColumnIndex1'][()])-1
#format for TF 
Vals1 = np.int64(I1val)
R1 = np.int64(I1RowIndex)
C1 = np.int64(I1ColIndex)
Indicies1 = np.transpose(np.mat([R1, C1]))
DenseShape1 = np.int64(np.asarray([28**2, len(C1)]))

#Custom PTC layer ###################################################################################################

#Geometirc Varialbs
W1   = tf.Variable(W1cell.todense, trainable=False)
Ind1 = tf.Variable(np.int64(Indicies1), trainable=False)
MV1  = tf.Variable(Vals1, trainable=False)
DS1  = tf.Variable(DenseShape1, trainable=False)
L1   = tf.Variable(L1mat, trainable=False)
n_pts = 28**2


def PTCconv(x):
    z_long  = tf.reshape(tf.transpose(x), [n_pts * 1 ,])
    val     = tf.gather(z_long, MV1)
    zmr     = tf.sparse.SparseTensor(indices=Indicies1, values=val, dense_shape=DenseShape1)
    #wk      = tf.transpose(tf.sparse.sparse_dense_matmul(W1, Ker1))
    wk      = tf.transpose(tf.matmul(tf.cast(W1,tf.float32), Ker1))
    wk_long = tf.reshape(wk, [L1 * 1, nKer1,])
    c_long  = tf.sparse.sparse_dense_matmul(zmr, wk_long)
    c_full  = tf.reshape(c_long, [1, 28, 28,nKer1,])
    f = tf.reduce_sum(c_full, 0)
    C1 = tf.nn.relu(f);
    return tf.expand_dims(C1, axis = 0)

	
#Model##########################################################################################################################################################################################################################


#Trainable Parameters
Ker1  = tf.transpose(weight_var([nKer1,13]))	

#input
x1 = tf.keras.layers.Input(shape=(28,28,))
y_true = tf.keras.layers.Input(shape=(10,))

#Foraward model
x2 = Lambda(PTCconv)(x1)
x3 = tf.keras.layers.Reshape((-1,))(x2)
y_pred = tf.keras.layers.Dense(10, activation='softmax')(x3)
	

#Compile model#########################################################################################################	
	
train_model = tf.keras.Model(inputs = [x1], outputs = y_pred, name = 'Trainer') 
train_model.compile(optimizer='rmsprop',
                   loss='categorical_crossentropy')
                   
                   
                  
## Train Model	#########################################################################################################	
train_model.fit(x_train, tf.keras.utils.to_categorical(y_train, num_classes=10),
                batch_size=1,
                epochs = 1)	
	

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
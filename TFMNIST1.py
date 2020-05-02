#exec(open("E:/Dropbox/Research/PTCnet/TFMNIST/TFMNIST1.py").read()) 


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
import scipy.sparse as spsp
import scipy.io as spio
import time

#Funcitons
def weight_var(shape):
	initial = tf.truncated_normal(shape = shape, stddev = .1)
	return tf.Variable(initial)

def FC_var(shape):
	initial = tf.truncated_normal(shape = shape, stddev = .1)
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
	return tf.SparseTensorValue(indices, Ctemp.data, Ctemp.shape) 

def WtoTensor(W):
	WF  = W[1]
	Wcoo = WF.tocoo()
	indices = np.mat([Wcoo.row, Wcoo.col]).transpose()
	return tf.SparseTensorValue(indices, Wcoo.data, Wcoo.shape)
	
def MtoTensor(W):
	Wcoo = (2*W).tocoo()
	indices = np.mat([Wcoo.row, Wcoo.col]).transpose()
	return tf.SparseTensorValue(indices, Wcoo.data, Wcoo.shape)
	
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

	
	
#Model##########################################################################################################################################################################################################################


#Place holders for data
x1   = tf.placeholder(tf.float32)
y    = tf.placeholder(tf.float32)
#Placehodlers for goemtric info
W1   = tf.sparse_placeholder(tf.float32)
Ind1 = tf.placeholder(tf.int64)
MV1  = tf.placeholder(tf.int64)
DS1  = tf.placeholder(tf.int64)
L1   = tf.placeholder(tf.int32)

n_pts = tf.placeholder(tf.int32)

#Intialize
Ker1  = tf.transpose(weight_var([nKer1,13]))	
FC1   = weight_var([28**2*nKer1,10])
B1    = weight_var([10])

##Conv
z_long  = tf.reshape(tf.transpose(x1), [n_pts * 1 ])
val     = tf.gather(z_long, MV1, 0)
zmr     = tf.SparseTensor(indices=Ind1, values=val, dense_shape=DS1)
wk      = tf.transpose(tf.sparse_tensor_dense_matmul(W1, Ker1))
wk_long = tf.reshape(wk, [L1 * 1, nKer1])
c_long  = tf.sparse_tensor_dense_matmul(zmr, wk_long)
c_full  = tf.reshape(c_long, [1, n_pts,nKer1])
f = tf.reduce_sum(c_full, 0)
C1 = tf.nn.relu(f);
#FC
C1Flat = tf.reshape(C1,[(28**2)*nKer1,1])
F1 = tf.matmul(tf.transpose(C1Flat),FC1)+B1	

##Loss
reg = tf.norm(Ker1)+tf.norm(B1)+tf.norm(FC1)
output = tf.nn.softmax(F1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=F1,labels=y)+gamma1*reg


#Session and training#########################################################################################################


#Session
sess = tf.Session()
#define train method	
train_step = tf.train.AdamOptimizer(alpha).minimize(cross_entropy)
#Initiatlize and begin saver
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#Write info for graph
#writer = tf.summary.FileWriter("E:/Dropbox/Research/PTCnet/TFimplementation3/Vis",sess.graph)
	
#Load MNSIT from tutorial data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	
	
#Load Manifold and WeightMatricies
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
Vals1 = I1val
R1 = np.int64(I1RowIndex)
C1 = np.int64(I1ColIndex)
Indicies1 = np.transpose(np.mat([R1, C1]))
DenseShape1 = np.int64(np.asarray([28**2, len(C1)]))

#Training Loop
accuracy = []
print('Begining Training')
for i in range(max_its):
	#Choose data
	batch = mnist.train.next_batch(1)
	
	#Train Step
	train_step.run(session=sess, feed_dict={x1:batch[0],y:batch[1],W1:W1mat,Ind1:Indicies1,MV1:Vals1,DS1:DenseShape1,L1:L1mat,n_pts:784})
	
	#Report
	if i%report == 0:
		count = 0;
		for j in range(25):
				batch = mnist.train.next_batch(1)
				ypred = F1.eval(session=sess, feed_dict={x1:batch[0],y:batch[1],W1:W1mat,Ind1:Indicies1,MV1:Vals1,DS1:DenseShape1,L1:L1mat,n_pts:784})
				if np.argmax(ypred) == np.argmax(batch[1]):
					count = count+1
		acc = count/25
		accuracy.append(acc)
		print('i:',i,'acc:',acc)

#Testing####################################################################################
	
print('Begining Testing')
count = 0
for i in range(10000):
	ypred = F1.eval(session=sess, feed_dict={x1:mnist.test.images[i],y:mnist.test.labels[i],W1:W1mat,Ind1:Indicies1,MV1:Vals1,DS1:DenseShape1,L1:L1mat,n_pts:784})
	if np.argmax(ypred) == np.argmax(mnist.test.labels[i]):
		count = count+1
		if i%1000 == 0:
			print(i, 'Tests Complete')
acc = count/10000
print('Test Accuacy:', acc)


#Plot
its = np.arange(int(max_its/report))*report
plt.figure(0)
fig0,ax1 = plt.subplots()
ax1.semilogy(its,accuracy,'g-', label ="Loss")
ax1.set_ylabel('Accuracy', color='g')
ax1.tick_params('y', colors='g')
ax1.set_xlabel('Iterations')

plt.show()
	
	
	
	

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
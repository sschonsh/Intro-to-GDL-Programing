#exec(open("E:/Dropbox/Research/Tubular/PTCFeatureMain.py").read())

#exec(open("C:/Users/sscho/Dropbox/Research/Tubular/PTCFeatureMain.py").read())

computerID = 1;#1:RPI 2:laptop


import numpy as np
import tensorflow as tf
import time
from os import listdir
import scipy.io as spio
import sys
if computerID == 1:
    sys.path.insert(0, "E:/Dropbox/Research/Tubular")
if computerID == 2:
    sys.path.insert(0, "C:/Users/sscho/Dropbox/Research/Tubular")
import PTCModel
import data_loader_featurelearning
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Train Parameters
alpha     = 1e-4
iters     = 200
n_load    = 3
report    = 1

#Model parametrers
ker_size  = 13
n_ker     = [16]*8
n_pts     = 6890
magrin    = 10


######Utilities##############################################################################
def choose_samples():
    A       = range(9)
    Classes = np.random.choice(A, size=2, replace=False)
    Poses   = np.random.choice(A, size=2, replace=False)
    idx1 = 10*Classes[0]+Poses[0]
    idx2 = 10*Classes[0]+Poses[1]
    idx3 = 10*Classes[1]+Poses[0]
    return idx1, idx2, idx3

#Load Data
print('Loading Data')
if computerID == 1:
    path = 'E:/Dropbox/Research/PTCnet/UnetCorrespondence/DataV2/'
if computerID == 2:
    path = 'C:/Users/sscho/Dropbox/Research/PTCnet/UnetCorrespondence/DataV2/'
[stiffness, distances, m_mat_1, w_mat_1, h_mat_1, l_mat_1, val_1, idx_1, dense_shape_1, interp_mat_1,
    m_mat_2, w_mat_2, h_mat_2, l_mat_2, val_2, idx_2, dense_shape_2, interp_mat_2,
    m_mat_3, w_mat_3, h_mat_3, l_mat_3, val_3, idx_3, dense_shape_3, interp_mat_3,
    m_mat_4, w_mat_4, h_mat_4, l_mat_4, val_4, idx_4, dense_shape_4, interp_mat_4] = data_loader_featurelearning.load_mat(path, 3, 90)


#Set up Session
sess = tf.Session()
siam = PTCModel.Siam(n_pts, 16, 3, n_ker, ker_size)
train_step = tf.train.AdamOptimizer(alpha).minimize(siam.Loss)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

#Trainer
#### Train #####
Losses = []
intLosses = []
extLosses = []

for i in range(n_train):
    #idx1, idx2, idx3 = choose_samples()
    idx1 = 1
    idx2 = 2
    idx3 = 3
    
    _, l1, l2, l3 = sess.run([train_step, mdl.TotalLoss, mdl.intloss, mdl.extloss],
            feed_dict={
                #input
                siam.x1: h_mat_1[idx1],
                siam.x2: h_mat_1[idx2],
                siam.x3: h_mat_1[idx3],

                #multi level mass
                siam.mass1_1 : m_mat_1[idx1],
                siam.mass1_2 : m_mat_2[idx1],
                siam.mass1_3 : m_mat_3[idx1],
                siam.mass2_1 : m_mat_1[idx2],
                siam.mass2_2 : m_mat_2[idx2],
                siam.mass2_3 : m_mat_3[idx2],
                siam.mass3_1 : m_mat_1[idx3],
                siam.mass3_2 : m_mat_2[idx3],
                siam.mass3_3 : m_mat_3[idx3],                
  
                #interpolations
                siam.weight1_1 : w_mat_1[idx1],
                siam.weight1_2 : w_mat_2[idx1],
                siam.weight1_3 : w_mat_3[idx1],
                siam.weight2_1 : w_mat_1[idx2],
                siam.weight2_2 : w_mat_2[idx2],
                siam.weight2_3 : w_mat_3[idx2],
                siam.weight3_1 : w_mat_1[idx3],
                siam.weight3_2 : w_mat_2[idx3],
                siam.weight3_3 : w_mat_3[idx3],
                
                #index for z
                siam.ind1_1 : idx_1[idx1],
                siam.ind1_2 : idx_2[idx1],
                siam.ind1_3 : idx_3[idx1],
                siam.ind2_1 : idx_1[idx2],
                siam.ind2_2 : idx_2[idx2],
                siam.ind2_3 : idx_3[idx2],
                siam.ind3_1 : idx_1[idx3],
                siam.ind3_2 : idx_2[idx3],
                siam.ind3_3 : idx_3[idx3],

                
                #index for m
                siam.mv1_1 : val_1[idx1],
                siam.mv1_2 : val_2[idx1],
                siam.mv1_3 : val_3[idx1],
                siam.mv2_1 : val_1[idx2],
                siam.mv2_2 : val_2[idx2],
                siam.mv2_3 : val_3[idx2],
                siam.mv3_1 : val_1[idx3],
                siam.mv3_2 : val_2[idx3],
                siam.mv3_3 : val_3[idx3],
                
                #dense shape of z
                siam.ds1_1 : dense_shape_1[idx1],
                siam.ds1_2 : dense_shape_2[idx1], 
                siam.ds1_3 : dense_shape_3[idx1], 
                siam.ds2_1 : dense_shape_1[idx2],
                siam.ds2_2 : dense_shape_2[idx2], 
                siam.ds2_3 : dense_shape_3[idx2], 
                siam.ds2_1 : dense_shape_1[idx2],
                siam.ds2_2 : dense_shape_2[idx2], 
                siam.ds2_3 : dense_shape_3[idx2], 
                
                #reshape stride
                siam.l1_1 : l_mat_1[idx1],
                siam.l1_2 : l_mat_2[idx1],
                siam.l1_3 : l_mat_3[idx1],
                siam.l2_1 : l_mat_1[idx2],
                siam.l2_2 : l_mat_2[idx2],
                siam.l2_3 : l_mat_3[idx2],
                siam.l3_1 : l_mat_1[idx3],
                siam.l3_2 : l_mat_2[idx3],
                siam.l3_3 : l_mat_3[idx3]                 
                    })  
    

    if not i % report:
        if not i == 0:
            Losses.append(l1)
            intLosses.append(l2)
            extLosses.append(l3)
        print('Iteration %03d: Samples:%02d,%02d,%02d int/ext/Total:%.3e,%.3e,%.3e' % (i, idx1,idx2,idx3, l2,l3,l1))

        


#Compute Features    
intFeats = np.zeros([n_load,16])
extFeats = np.zeros([n_load,16])          
for i in range(n_load):
    intFeats[i,:], extFeats[i,:] = sess.run([mdl.x1int,mdl.x1ext],
        feed_dict ={ 
                siam.x1: h_mat_1[i],

                #multi level mass
                siam.mass1_1 : m_mat_1[i],
                siam.mass1_2 : m_mat_2[i],
                siam.mass1_3 : m_mat_3[i],
                #interpolations
                siam.weight1_1 : w_mat_1[i],
                siam.weight1_2 : w_mat_2[i],
                siam.weight1_3 : w_mat_3[i],
                #index for z
                siam.ind1_1 : idx_1[i],
                siam.ind1_2 : idx_2[i],
                siam.ind1_3 : idx_3[i],
                #index for m
                siam.mv1_1 : val_1[i],
                siam.mv1_2 : val_2[i],
                siam.mv1_3 : val_3[i],               
                #dense shape of z
                siam.ds1_1 : dense_shape_1[i],
                siam.ds1_2 : dense_shape_2[i], 
                siam.ds1_3 : dense_shape_3[i],  
                #reshape stride
                siam.l1_1 : l_mat_1[i],
                siam.l1_2 : l_mat_2[i],
                siam.l1_3 : l_mat_3[i]
                    })


print('Done')
spio.savemat('C:/Users/sscho/Dropbox/Research/Tubular/PYTHONOUT_PTC', 
        mdict={'intFeats':intFeats, 'extFeats':extFeats})  


#Visualize Loss
max_its = epochs*n_train
its = np.arange(int(max_its/report))*report
plt.figure(0)
fig0,ax1 = plt.subplots()
ax1.semilogy(its,Losses,'g-', label ="Loss")
ax1.set_ylabel('Loss', color='g')
ax1.tick_params('y', colors='g')
ax1.set_xlabel('Iterations')
plt.show()


	


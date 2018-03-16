import mxnet as mx
import numpy as np
import logging
import data
from math import sqrt
from autoencoder import AutoEncoderModel
import os

#####################
# variables defined 
#####################
lambda_u = 1 # lambda_u in CDL
lambda_v = 10 # lambda_v in CDL
K = 50
p = 1
is_dummy = False
num_iter = 100 # about 68 iterations/epoch, the recommendation results at the end need 100 epochs
batch_size = 256

np.random.seed(1234) # set seed
lv = 1e-2 # lambda_v/lambda_n in CDL
dir_save = 'cdl%d' % p


#####################
# Create dir and log file 
#####################
if not os.path.isdir(dir_save):
    os.system('mkdir %s' % dir_save)
fp = open(dir_save+'/cdl.log','w')
# print 'p%d: lambda_v/lambda_u/ratio/K: %f/%f/%f/%d' % (p,lambda_v,lambda_u,lv,K)
# print "p%d: lambda_v/lambda_u/ratio/K: %f/%f/%f/%d".format(p,lambda_v,lambda_u,lv,K)
fp.write('p%d: lambda_v/lambda_u/ratio/K: %f/%f/%f/%d\n' % \
        (p,lambda_v,lambda_u,lv,K))
fp.close()



#####################
# Load dAta 
#####################

if is_dummy:
    X = data.get_dummy_mult()
    R = data.read_dummy_user()
else:
    X = data.get_mult()
    R = data.read_user()


#####################
# Network definition 
#####################

logging.basicConfig(level=logging.INFO)
cdl_model = AutoEncoderModel(mx.cpu(2), [X.shape[1],100,K],
    pt_dropout=0.2, internal_act='relu', output_act='relu')

# initialize the variable
train_X = X
V = np.random.rand(train_X.shape[0],K)/10
lambda_v_rt = np.ones((train_X.shape[0],K))*sqrt(lv)


# TRAINING
U, V, theta, BCD_loss = cdl_model.finetune(train_X, R, V, lambda_v_rt, lambda_u,
        lambda_v, dir_save, batch_size,
        num_iter, 'sgd', l_rate=0.1, decay=0.0,
        lr_scheduler=mx.misc.FactorScheduler(20000,0.1))
print('Training ends')


# SAVING THE MODEL
cdl_model.save(dir_save+'/cdl_pt.arg')
np.savetxt(dir_save+'/final-U.dat.demo',U,fmt='%.5f',comments='')
np.savetxt(dir_save+'/final-V.dat.demo',V,fmt='%.5f',comments='')
np.savetxt(dir_save+'/final-theta.dat.demo',theta,fmt='%.5f',comments='')



#######

# shifted the below code to reco.py, so that i dont have to train the model
# again if i get an error.

#######

# # GENERATE RECCOMENDATIONS
# import numpy as np
# from data import read_user
# def cal_rec(p,cut):
#     R_true = read_user('cf-test-1-users.dat')
#     dir_save = 'cdl'+str(p)
#     U = np.mat(np.loadtxt(dir_save+'/final-U.dat.demo'))
#     V = np.mat(np.loadtxt(dir_save+'/final-V.dat.demo'))
#     R = U*V.T
#     num_u = R.shape[0]
#     num_hit = 0
#     fp = open(dir_save+'/rec-list.dat','w')
#     for i in range(num_u):
#         if i!=0 and i%100==0:
#             print('User '+str(i))
#         l_score = R[i,:].A1.tolist()
#         pl = sorted(enumerate(l_score),key=lambda d:d[1],reverse=True)
#         l_rec = list(zip(*pl))[0][:cut]
#         s_rec = set(l_rec)
#         s_true = set(np.where(R_true[i,:]>0)[1].A1)
#         cnt_hit = len(s_rec.intersection(s_true))
#         fp.write('%d:' % cnt_hit)
#         fp.write(' '.join(map(str,l_rec)))
#         fp.write('\n')
#     fp.close()

# cal_rec(1,8)
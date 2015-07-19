"""
This is the file for DBN
Di Wu   stevenwudi@gmail.com
2015-06-12
"""
from numpy import log
from glob import glob
import os
import sys
import cPickle

from classes import GestureSample
# customized imports

#  modular imports
# the hyperparameter set the data dir, use etc classes, it's important to modify it according to your need
from classes.hyperparameters import use, lr, batch, reg, mom, tr, drop,\
                                     net,  DataLoader_with_skeleton_normalisation
from functions.test_functions import *
from functions.train_functions import _shared, _avg, write, ndtensor, print_params, lin,\
                                      training_report, epoch_report, _batch,\
                                      save_results, move_results, save_params, test_lio_skel
from classes.hyperparameters import batch
from dbn.utils import normalize

# customized imports
from dbn.GRBM_DBN import GRBM_DBN
from convnet3d_grbm_early_fusion import convnet3d_grbm_early_fusion


import scipy.io as sio  
from time import localtime, time

# number of hidden states for each gesture class
STATE_NO = 5
#data path and store path definition
pc="linux"
pc="windows"

if pc=="linux":
    data = "/idiap/user/dwu/chalearn/Test_video_skel"
    save_dst = "/idiap/user/dwu/chalearn/Test_DBN_state_matrix"
    res_dir_ = "/idiap/user/dwu/chalearn/result/"
elif pc=="windows":
    data = "/idiap/user/dwu/chalearn/Test_video_skel"
    save_dst = "/idiap/user/dwu/chalearn/Test_DBN_state_matrix"
    res_dir_ = "/idiap/user/dwu/chalearn/result/"


os.chdir(data)
samples=glob("*.zip") 
print len(samples), "samples found"

used_joints = ['ElbowLeft', 'WristLeft', 'ShoulderLeft', 'HandLeft',
               'ElbowRight', 'WristRight', 'ShoulderRight', 'HandRight',
               'Head', 'Spine', 'HipCenter']

lt = localtime()
res_dir = res_dir_+"/try/"+str(lt.tm_year)+"."+str(lt.tm_mon).zfill(2)+"." \
            +str(lt.tm_mday).zfill(2)+"."+str(lt.tm_hour).zfill(2)+"."\
            +str(lt.tm_min).zfill(2)+"."+str(lt.tm_sec).zfill(2)
os.makedirs(res_dir)

# we need to parse an absolute path for HPC to load
load_path = '/idiap/home/dwu/chalearn2014_wudi_lio/chalearn2014_wudi_lio'
######################################################################
net_convnet3d_grbm_early_fusion = convnet3d_grbm_early_fusion(res_dir, load_path)

load_path = '/idiap/home/dwu/chalearn2014_wudi_lio/chalearn2014_wudi_lio'
net_convnet3d_grbm_early_fusion.load_params(os.path.join(load_path,'paramsbest.zip'))
x_ = _shared(empty(tr.in_shape))
x_skeleton_ = _shared(empty(tr._skeleon_in_shape))
p_y_given_x = net_convnet3d_grbm_early_fusion.prediction_function(x_, x_skeleton_)
#############################
# load normalisation constant given load_path
Mean_skel, Std_skel, Mean_CNN, Std_CNN = net_convnet3d_grbm_early_fusion.load_normalisation_constant(load_path)


####################################################################
# DBN for skeleton modules
#################################################################### 
# ------------------------------------------------------------------------------
# symbolic variables
x_skeleton = ndtensor(len(tr._skeleon_in_shape))(name = 'x_skeleton') # video input
x_skeleton_ = _shared(empty(tr._skeleon_in_shape))

dbn = GRBM_DBN(numpy_rng=random.RandomState(123), n_ins=891, \
                hidden_layers_sizes=[2000, 2000, 1000], n_outs=101, input_x=x_skeleton, label=y )  
# we load the pretrained DBN skeleton parameteres here
load_path = '/idiap/user/dwu/chalearn/result/try/36.7% 2015.07.09.17.53.10'
dbn.load_params_DBN(os.path.join(load_path,'paramsbest.zip'))

test_model = function([], dbn.logLayer.p_y_given_x, 
            givens={x_skeleton: x_skeleton_}, 
            on_unused_input='ignore')


for file_count, file in enumerate(samples):
    condition = (file_count > -1)   
    if condition:   #wudi only used first 650 for validation !!! Lio be careful!
        save_path= os.path.join(data, file)
        print file
        time_start = time()
        # we load precomputed feature set or recompute the whole feature set
        if os.path.isfile(save_path):
            print "loading exiting file"
            data_dic = cPickle.load(open(save_path,'rb'))
            video = data_dic["video"]
            Feature_gesture = data_dic["Feature_gesture"]
            assert video.shape[0] == Feature_gesture.shape[0]            
        else:
            print("\t Processing file " + file)
            # Create the object to access the sample
            sample = GestureSample(os.path.join(data,file))
            print "finish loading samples"
            video, Feature_gesture = sample.get_test_data_wudi_lio(used_joints)
            assert video.shape[0] == Feature_gesture.shape[0]# -*- coding: utf-8 -*-
            
            print "finish preprocessing"
            out_file = open(save_path, 'wb')
            cPickle.dump({"video":video, "Feature_gesture":Feature_gesture}, out_file, protocol=cPickle.HIGHEST_PROTOCOL)
            out_file.close()

        print "start computing likelihood"
        observ_likelihood = numpy.empty(shape=(video.shape[0],20*STATE_NO+1)) # 20 classed * 5 states + 1 ergodic state
        for batchnumber in xrange(video.shape[0]/batch.micro):

            skel_temp =  Feature_gesture[batch.micro*batchnumber:batch.micro*(batchnumber+1),:]  
            x_skeleton_.set_value(normalize(skel_temp,Mean_skel, Std_skel).astype("float32"), borrow=True)         
            observ_likelihood[batch.micro*batchnumber:batch.micro*(batchnumber+1),:] =  test_model()

        # because input batch number should be 64, so here it is a bit of hack:
        skel_temp_1 = Feature_gesture[batch.micro* (batchnumber+1):,:]  
        skel_temp_2 = numpy.zeros(shape=(64-skel_temp_1.shape[0],891))
        skel_temp = numpy.concatenate((skel_temp_1, skel_temp_2), axis=0)
        x_skeleton_.set_value(normalize(skel_temp,Mean_skel, Std_skel).astype("float32"), borrow=True)
        ob_temp = test_model()
        observ_likelihood[batch.micro* (batchnumber+1):,:] =  ob_temp[:video_temp_1.shape[0], :]

        ##########################
        # save state matrix
        #####################
        save_path= os.path.join(save_dst, file)
        out_file = open(save_path, 'wb')
        cPickle.dump(observ_likelihood, out_file, protocol=cPickle.HIGHEST_PROTOCOL)
        out_file.close()
        
        print "use %f second"% (time()-time_start)
        
      

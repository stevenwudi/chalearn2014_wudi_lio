"""
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
                                      save_results, move_results, save_params, test_lio_skel, load_params
from classes.hyperparameters import batch
from dbn.utils import normalize

from conv3d_chalearn import conv3d_chalearn
from convnet3d import LogRegr
import theano.tensor as T
from theano import function

import scipy.io as sio  
from time import localtime, time

# number of hidden states for each gesture class
STATE_NO = 5
#data path and store path definition


data = "/idiap/user/dwu/chalearn/Test_video_skel"
save_dst = "/idiap/user/dwu/chalearn/Test_CNN_stata_matrix"
res_dir_ = "/idiap/user/dwu/chalearn/result/"

lt = localtime()
res_dir = res_dir_+"/try/"+str(lt.tm_year)+"."+str(lt.tm_mon).zfill(2)+"." \
            +str(lt.tm_mday).zfill(2)+"."+str(lt.tm_hour).zfill(2)+"."\
            +str(lt.tm_min).zfill(2)+"."+str(lt.tm_sec).zfill(2)
os.makedirs(res_dir)

# we need to parse an absolute path for HPC to load
#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument('path')# the path to load best parameters
#args = parser.parse_args()
#load_path = args.path

load_path='/remote/idiap.svm/user.active/dwu/chalearn/result/try/CNN_normalisation_53.0% 2015.06.23.12.17.31/'
######################################################################
import cPickle
f = open('CNN_normalization.pkl','rb')
CNN_normalization = cPickle.load(f)
Mean_CNN = CNN_normalization ['Mean_CNN']
Std_CNN = CNN_normalization['Std_CNN']

# customized data loader for both video module and skeleton module
#loader = DataLoader_with_skeleton_normalisation(src, tr.batch_size, Mean_CNN, Std_CNN) # Lio changed it to read from HDF5 files
# we load the CNN parameteres here
x = ndtensor(len(tr.in_shape))(name = 'x') # video input
x_ = _shared(empty(tr.in_shape))


use.load=True
use.fast_conv=True
video_cnn = conv3d_chalearn(x, use, lr, batch, net, reg, drop, mom, tr, res_dir, load_path)

out = video_cnn.out
layers = [] # all architecture layers
# softmax layer
if use.load:
    W, b = load_params(use, load_path)
    print W.shape, b.shape
    layers.append(LogRegr(out, rng=tr.rng, n_in=net.hidden_vid, W=W, b=b,
        W_scale=net.W_scale[-1], b_scale=net.b_scale[-1], n_out=net.n_class))
else:
    layers.append(LogRegr(out, rng=tr.rng, n_in=net.hidden_vid, 
        W_scale=net.W_scale[-1], b_scale=net.b_scale[-1], n_out=net.n_class))

print "compiling output"
test_model = function([], layers[-1].p_y_given_x, 
                      givens={x: x_}, 
                    on_unused_input='ignore')

os.chdir(data)
samples=glob("*.zip") 
print len(samples), "samples found"

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

            video_temp = video[batch.micro*batchnumber:batch.micro*(batchnumber+1),:]

            x_.set_value(normalize(video_temp, Mean_CNN, Std_CNN).astype("float32"),borrow=True)
            
            observ_likelihood[batch.micro*batchnumber:batch.micro*(batchnumber+1),:] =  test_model()

        # because input batch number should be 64, so here it is a bit of hack:
        video_temp_1 = video[batch.micro* (batchnumber+1):,:]   
        video_temp_2 = numpy.zeros(shape=(64-video_temp_1.shape[0], 2, 2, 4, 64, 64))
        video_temp = numpy.concatenate((video_temp_1, video_temp_2), axis=0)      
        x_.set_value(normalize(video_temp, Mean_CNN, Std_CNN).astype("float32"),borrow=True)

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
        
      
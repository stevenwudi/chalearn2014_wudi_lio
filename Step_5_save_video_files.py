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
                                      save_results, move_results, save_params, test_lio_skel
from classes.hyperparameters import batch
from dbn.utils import normalize

from convnet3d_grbm_early_fusion import convnet3d_grbm_early_fusion


import scipy.io as sio  

## Load Prior and transitional Matrix
dic=sio.loadmat('Prior_Transition_matrix_5states.mat')
Transition_matrix = dic['Transition_matrix']
Prior = dic['Prior']
# number of hidden states for each gesture class
STATE_NO = 5
#data path and store path definition
pc = "wudi"
if pc=="wudi":
    data = r"E:\CHALEARN2014\Train" # dir of original data -- note that wudi has decompressed it!!!
    data = '/idiap/user/dwu/chalearn/Test'    
    save_dst = "/idiap/user/dwu/chalearn/Test_CNN_precompute"
    res_dir_ = "/idiap/user/dwu/chalearn/result/"
elif pc=="lio":
    data = "/media/lio/Elements/chalearn/trainingset"
    save_dst = " "

load_flag = False

os.chdir(data)
if pc=="wudi":
        samples=glob("*.zip")  # because wudi unzipped all the files already!
elif pc=="lio":
        samples=glob("*.zip")

print len(samples), "samples found"

used_joints = ['ElbowLeft', 'WristLeft', 'ShoulderLeft', 'HandLeft',
               'ElbowRight', 'WristRight', 'ShoulderRight', 'HandRight',
               'Head', 'Spine', 'HipCenter']


for file_count, file in enumerate(samples):
    condition = (file_count > -1)   
    if condition:   #wudi only used first 650 for validation !!! Lio be careful!
        save_path= os.path.join(save_dst, file)
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


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
from time import localtime
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
    save_dst = r"/idiap/user/dwu/chalearn/Test_CNN_precompute"
    res_dir_ = r"/idiap/user/dwu/chalearn/result/"
elif pc=="lio":
    data = r"/media/lio/Elements/chalearn/trainingset"
    save_dst = " "

load_flag = False

os.chdir(data)
if pc=="wudi":
        samples=glob("*")  # because wudi unzipped all the files already!
elif pc=="lio":
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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()
load_path = args.path
######################################################################
net_convnet3d_grbm_early_fusion = convnet3d_grbm_early_fusion(res_dir, load_path)
net_convnet3d_grbm_early_fusion.load_params(os.path.join(load_path,'paramsbest.zip'))
x_ = _shared(empty(tr.in_shape))
x_skeleton_ = _shared(empty(tr._skeleon_in_shape))
p_y_given_x = net_convnet3d_grbm_early_fusion.prediction_function(x_, x_skeleton_)
#############################
# load normalisation constant given load_path
Mean_skel, Std_skel, Mean_CNN, Std_CNN = net_convnet3d_grbm_early_fusion.load_normalisation_constant(load_path)


for file_count, file in enumerate(samples):
    condition = (file_count > -1)   
    if condition:   #wudi only used first 650 for validation !!! Lio be careful!
        print("\t Processing file " + file)
        # Create the object to access the sample
        print os.path.join(data,file)
        sample = GestureSample(os.path.join(data,file))
        print "finish loading samples"
        video, Feature_gesture = sample.get_test_data_wudi_lio(used_joints)
        print video.shape, Feature_gesture.shape
        save_path= os.path.join(save_dst, file)
        #out_file = open(save_path, 'wb')
        #cPickle.dump(video, out_file, protocol=cPickle.HIGHEST_PROTOCOL)
        #out_file.close()

        print "start computing likelihood"
        observ_likelihood = numpy.empty(shape=(video.shape[0],20*STATE_NO+1)) # 20 classed * 5 states + 1 ergodic state
        for batchnumber in xrange(video.shape[0]/batch.micro):

            video_temp = video[batch.micro*batchnumber:batch.micro*(batchnumber+1),:]
            skel_temp =  Feature_gesture[batch.micro*batchnumber:batch.micro*(batchnumber+1),:]  

            x_.set_value(normalize(video_temp, Mean_CNN, Std_CNN).astype("float32"),borrow=True)
            x_skeleton_.set_value(normalize(skel_temp,Mean_skel, Std_skel).astype("float32"), borrow=True)
            
            observ_likelihood[batch.micro*batchnumber:batch.micro*(batchnumber+1),:] =  p_y_given_x()

        # because input batch number should be 64, so here it is a bit of hack:
        video_temp_1 = video[batch.micro* (batchnumber+1):,:]   
        video_temp_2 = numpy.zeros(shape=(64-video_temp_1.shape[0], 2, 2, 4, 64, 64))
        video_temp = numpy.concatenate((video_temp_1, video_temp_2), axis=0)
        skel_temp_1 = Feature_gesture[batch.micro* (batchnumber+1):,:]  
        skel_temp_2 = numpy.zeros(shape=(64-skel_temp_1.shape[0],891))
        skel_temp = numpy.concatenate((skel_temp_1, skel_temp_2), axis=0)
        x_.set_value(normalize(video_temp, Mean_CNN, Std_CNN).astype("float32"),borrow=True)
        x_skeleton_.set_value(normalize(skel_temp,Mean_skel, Std_skel).astype("float32"), borrow=True)

        ob_temp = p_y_given_x()
        observ_likelihood[batch.micro* (batchnumber+1):,:] =  ob_temp[:video_temp_1.shape[0], :]

        ##########################
        # viterbi path decoding
        #####################

        log_observ_likelihood = numpy.log(observ_likelihood.T + numpy.finfo(numpy.float32).eps)
        log_observ_likelihood[-1, 0:5] = 0
        log_observ_likelihood[-1, -5:] = 0

        print("\t Viterbi path decoding " )
        # do it in log space avoid numeric underflow
        [path, predecessor_state_index, global_score] = viterbi_path_log(log(Prior), log(Transition_matrix), log_observ_likelihood)
        #[path, predecessor_state_index, global_score] =  viterbi_path(Prior, Transition_matrix, observ_likelihood)
        
        # Some gestures are not within the vocabulary
        [pred_label, begin_frame, end_frame, Individual_score, frame_length] = viterbi_colab_states(path, global_score, state_no = 5, threshold=-2, mini_frame=19)

        #heuristically we need to add 1 more frame here
        begin_frame += 1 
        end_frame +=5 # because we cut 4 frames as a cuboid so we need add extra 4 frames 

        gesturesList=sample.getGestures()

        import matplotlib.pyplot as plt
        STATE_NO = 5
        im  = imdisplay(global_score)
        plt.clf()
        plt.imshow(im, cmap='gray')
        plt.plot(range(global_score.shape[-1]), path, color='c',linewidth=2.0)
        plt.xlim((0, global_score.shape[-1]))
        # plot ground truth
        for gesture in gesturesList:
        # Get the gesture ID, and start and end frames for the gesture
            gestureID,startFrame,endFrame=gesture
            frames_count = numpy.array(range(startFrame, endFrame+1))
            pred_label_temp = ((gestureID-1) *STATE_NO +2) * numpy.ones(len(frames_count))
            plt.plot(frames_count, pred_label_temp, color='r', linewidth=5.0)
            
        # plot clean path
        for i in range(len(begin_frame)):
            frames_count = numpy.array(range(begin_frame[i], end_frame[i]+1))
            pred_label_temp = ((pred_label[i]-1) *STATE_NO +2) * numpy.ones(len(frames_count))
            plt.plot(frames_count, pred_label_temp, color='#ffff00', linewidth=2.0)

        if False:
            plt.show()
        else:     
            from pylab import savefig
            save_dir=r'D:\Chalearn2014'
            save_path= os.path.join(save_dir,file)
            savefig(save_path, bbox_inches='tight')
                #plt.show()
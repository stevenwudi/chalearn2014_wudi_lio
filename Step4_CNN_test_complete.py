from glob import glob
import os
import sys
import cPickle

from classes import GestureSample
from functions.preproc_functions import *
from functions.test_functions import *
from functions.test_cnn_build import build
from classes.hyperparameters import batch

import scipy.io as sio  
## Load Prior and transitional Matrix
dic=sio.loadmat('Prior_Transition_matrix_5states.mat')
Transition_matrix = dic['Transition_matrix']
Prior = dic['Prior']
#data path and store path definition
pc = "wudi"
if pc=="wudi":
    data = r"I:\Kaggle_multimodal\Training" # dir of original data -- note that wudi has decompressed it!!!
    save_dst = r"I:\Kaggle_multimodal\Test_CNN_precompute"
elif pc=="lio":
    data = r"/media/lio/Elements/chalearn/trainingset"

load_flag = False

os.chdir(data)
if pc=="wudi":
        samples=glob("*")  # because wudi unzipped all the files already!
elif pc=="lio":
        samples=glob("*.zip")

print len(samples), "samples found"



for file_count, file in enumerate(samples):
    condition = (file_count > 650)   
    if condition:   #wudi only used first 650 for validation !!! Lio be careful!
        print("\t Processing file " + file)
        # Create the object to access the sample
        sample = GestureSample(os.path.join(data,file))
        if not load_flag:
            video = sample.get_test_data_wudi_lio()
            save_path= os.path.join(save_dst, file)
            out_file = open(save_path, 'wb')
            cPickle.dump(video, out_file, protocol=cPickle.HIGHEST_PROTOCOL)
            out_file.close()
        else:
            save_path= os.path.join(data,file,'test')
            video = cPickle.load(open(save_path,"rb"))
            print video.shape

        print "start computing likelihood"
        observ_likelihood = numpy.empty(shape=(video.shape[0],20*5+1)) # 20 classed * 5 states + 1 ergodic state
        for batchnumber in xrange(video.shape[0]/batch.micro):
            video_temp = video[batch.micro*batchnumber:batch.micro*(batchnumber+1),:]   
            x_.set_value(video_temp.astype("float32"),borrow=True)
            y_pred, p_y_given_x = evalu_model()
            observ_likelihood[batch.micro*batchnumber:batch.micro*(batchnumber+1),:] =  \
                p_y_given_x

        # because input batch number should be 64, so here it is a bit of hack:
        video_temp_1 = video[batch.micro* (batchnumber+1):,:]   
        video_temp_2 = numpy.zeros(shape=(64-video_temp_1.shape[0], 2, 2, 4, 64, 64))
        video_temp = numpy.concatenate((video_temp_1, video_temp_2), axis=0)
        x_.set_value(video_temp.astype("float32"),borrow=True)
        y_pred, p_y_given_x = evalu_model()
        observ_likelihood[batch.micro* (batchnumber+1):,:] =  \
            p_y_given_x[:video_temp_1.shape[0], :]


        ##########################
        # viterbi path decoding
        #####################

        log_observ_likelihood = log(observ_likelihood.T + numpy.finfo(numpy.float32).eps)
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
            save_dir=r'C:\Users\PC-User\Documents\GitHub\chalearn2014_wudi_lio\SK_path'
            save_path= os.path.join(save_dir,file)
            savefig(save_path, bbox_inches='tight')
                #plt.show()
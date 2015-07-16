from glob import glob
import os
import sys
import cPickle
import scipy.io as sio  
import time

from classes import GestureSample
from functions.preproc_functions import *
from functions.test_functions import *


## Load Prior and transitional Matrix
dic=sio.loadmat('Prior_Transition_matrix_5states.mat')
Transition_matrix = dic['Transition_matrix']
Prior = dic['Prior']


#data path and store path definition

data = "/idiap/user/dwu/chalearn/Test_origin" # dir of original data -- note that wudi has decompressed it!!!
obs_likelihodd_dir = "/idiap/user/dwu/chalearn/Test_fusion_state_state_matrix_precompute"
outPred='/idiap/user/dwu/chalearn/Test_early_fusion_pred'

os.chdir(data)
samples=glob("*")  # because wudi unzipped all the files already!
print len(samples), "samples found"

for file_count, file in enumerate(samples):
        print("\t Processing file " + file)
        time_tic = time.time() 
        # Create the object to access the sample
        sample = GestureSample(os.path.join(data,file))
        load_path = os.path.join(obs_likelihodd_dir,file)
        observ_likelihood = cPickle.load(open(load_path,"rb"))
        #print observ_likelihood.shape
        ##########################
        # viterbi path decoding
        ########################
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

        #plotting
        gesturesList=sample.getGestures()
        #import matplotlib.pyplot as plt
        #STATE_NO = 5
        #im  = imdisplay(global_score)
        #plt.clf()
        #plt.imshow(im, cmap='gray')
        #plt.plot(range(global_score.shape[-1]), path, color='c',linewidth=2.0)
        #plt.xlim((0, global_score.shape[-1]))
        ## plot ground truth
        #for gesture in gesturesList:
        ## Get the gesture ID, and start and end frames for the gesture
        #    gestureID,startFrame,endFrame=gesture
        #    frames_count = numpy.array(range(startFrame, endFrame+1))
        #    pred_label_temp = ((gestureID-1) *STATE_NO +2) * numpy.ones(len(frames_count))
        #    plt.plot(frames_count, pred_label_temp, color='r', linewidth=5.0)
            
        ## plot clean path
        #for i in range(len(begin_frame)):
        #    frames_count = numpy.array(range(begin_frame[i], end_frame[i]+1))
        #    pred_label_temp = ((pred_label[i]-1) *STATE_NO +2) * numpy.ones(len(frames_count))
        #    plt.plot(frames_count, pred_label_temp, color='#39FF14', linewidth=2.0)

        #if False:
        #    plt.show()
        #else:     
        #    from pylab import savefig
        #    save_dir=r'/idiap/user/dwu/chalearn/Test_early_fusion_visualisation'
        #    save_path= os.path.join(save_dir,file)
        #    savefig(save_path, bbox_inches='tight')



        print "Elapsed time %d sec" % int(time.time() - time_tic)

        pred=[]
        for i in range(len(begin_frame)):
            pred.append([ pred_label[i], begin_frame[i], end_frame[i]] )

       
        sample.exportPredictions(pred,outPred)
     # ###############################################
        ## delete the sample
        del sample
        
        
TruthDir=r'I:\Kaggle_multimodal\Code_for_submission\Final_project\training\gt'
final_score = evalGesture(outPred,TruthDir)         
print("The score for this prediction is " + "{:.12f}".format(final_score))

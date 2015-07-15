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
pc = "wudi"
if pc=="wudi":
    data = r"I:\Kaggle_multimodal\Test\Test" # dir of original data -- note that wudi has decompressed it!!!
    cnn_obs_likelihodd_dir = r"I:\Kaggle_multimodal\Precompute_state_matrix_TEST"
    sk_obs_likelihood_dir = r"I:\Kaggle_multimodal\Code_for_submission\Final_project\training\SK_threshold_acc"
    outPred=r'C:\Users\PC-User\Documents\GitHub\chalearn2014_wudi_lio\CNN_test_pred_combine_sk_cnn'
    outPred=r'C:\Users\PC-User\Documents\GitHub\chalearn2014_wudi_lio\test_pred_sk'
elif pc=="lio":
    data = r"/media/lio/Elements/chalearn/trainingset"


os.chdir(data)
if pc=="wudi":
        samples=glob("*")  # because wudi unzipped all the files already!
elif pc=="lio":
        samples=glob("*.zip")

print len(samples), "samples found"

#TL = [-5, -10, -numpy.inf] # is 0.739770037290, 0.772290750017, 0.807449899917

#Alpha = [0.25, 0.5, 0.75]  # 0.807809390788, 0.811087250153, 0.772292829366
 
for alpha in ([0]): 
    for file_count, file in enumerate(samples):
        condition = (file_count != 104)   
        if condition:   #wudi only used first 650 for validation !!! Lio be careful!
            print("\t Processing file " + file)
            time_tic = time.time() 
            # Create the object to access the sample
            sample = GestureSample(os.path.join(data,file))

            #load obser cnn
            load_path = os.path.join(cnn_obs_likelihodd_dir,file)
            observ_likelihood_cnn = cPickle.load(open(load_path,"rb"))
            observ_likelihood_cnn = observ_likelihood_cnn[:-1,:]
            #load obser sk --log!
            load_path = os.path.join(sk_obs_likelihood_dir,file)
            observ_likelihood_file = cPickle.load(open(load_path,"rb"))
            observ_likelihood_sk = observ_likelihood_file['log_observ_likelihood']
            #print observ_likelihood.shape
            ##########################
            # viterbi path decoding
            ########################
            #log_observ_likelihood = log(observ_likelihood_cnn.T + numpy.finfo(numpy.float32).eps) + observ_likelihood_sk
            log_observ_likelihood = alpha * log(observ_likelihood_cnn.T + numpy.finfo(numpy.float32).eps) + (1-alpha) *observ_likelihood_sk     
            log_observ_likelihood[-1, 0:5] = 0
            log_observ_likelihood[-1, -5:] = 0

            print("\t Viterbi path decoding " )
            # do it in log space avoid numeric underflow
            [path, predecessor_state_index, global_score] = viterbi_path_log(log(Prior), log(Transition_matrix), log_observ_likelihood)
            #[path, predecessor_state_index, global_score] =  viterbi_path(Prior, Transition_matrix, observ_likelihood)
        
            # Some gestures are not within the vocabulary
            [pred_label, begin_frame, end_frame, Individual_score, frame_length] = viterbi_colab_states(path, global_score, state_no = 5, threshold=-numpy.inf, mini_frame=19)

            #heuristically we need to add 1 more frame here
            begin_frame += 1 
            end_frame +=5 # because we cut 4 frames as a cuboid so we need add extra 4 frames 

            #plotting
            gesturesList=sample.getGestures()
            import matplotlib.pyplot as plt
            STATE_NO = 5
            im  = imdisplay(global_score)
            fig = plt.figure()
            plt.clf()
            plt.imshow(im, cmap='gray')
            plt.plot(range(global_score.shape[-1]), path, color='c',linewidth=2.0)
            plt.xlim((0, global_score.shape[-1]))
            plt.ylim([101,0])
            plt.yticks([0, 50, 101])
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
                import matplotlib.pyplot as pl
                save_dir=r'C:\Users\PC-User\Documents\GitHub\chalearn2014_wudi_lio\path_combined'
                save_path= os.path.join(save_dir,file)
                savefig(save_path+'.eps', format='eps', bbox_inches='tight')
                #plt.show()

            print "Elapsed time %d sec" % int(time.time() - time_tic)

            pred=[]
            for i in range(len(begin_frame)):
                pred.append([ pred_label[i], begin_frame[i], end_frame[i]] )

       
            sample.exportPredictions(pred,outPred)
         # ###############################################
            ## delete the sample
            del sample
        
    TruthDir=r'I:\Kaggle_multimodal\Test_label'
    final_score = evalGesture(outPred,TruthDir)         
    print("The score for this prediction is " + "{:.12f}".format(final_score))

# combined: 0.804609104245
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
    src_dir = r"I:\Kaggle_multimodal\Test_CNN_precompute"
    save_dst = r"I:\Kaggle_multimodal\Precompute_state_matrix"
elif pc=="lio":
    data = r"/media/lio/Elements/chalearn/trainingset"

load_flag = True

os.chdir(src_dir)
if pc=="wudi":
        samples=glob("*")  # because wudi unzipped all the files already!
elif pc=="lio":
        samples=glob("*.zip")

print len(samples), "samples found"
# compile test functions
evalu_model, x_ = build()
print "finish compiling......"


for file_count, file in enumerate(samples):
    condition = (file_count > -1)   
    if condition:   #wudi only used first 650 for validation !!! Lio be careful!
        print("\t Processing file " + file)
        # Create the object to access the sample
        load_path = os.path.join(src_dir,file)
        video = cPickle.load(open(load_path,"rb"))
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
        # save state matrix
        #####################
        save_path= os.path.join(save_dst, file)
        out_file = open(save_path, 'wb')
        cPickle.dump(observ_likelihood, out_file, protocol=cPickle.HIGHEST_PROTOCOL)
        out_file.close()
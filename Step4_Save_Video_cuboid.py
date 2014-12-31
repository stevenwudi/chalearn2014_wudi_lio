from glob import glob
import os
import sys
import cPickle
import time

from classes import GestureSample
from functions.preproc_functions import *
from functions.test_functions import *
from functions.test_cnn_build import build
from classes.hyperparameters import batch

#data path and store path definition
pc = "wudi"
if pc=="wudi":
    data = r"I:\Kaggle_multimodal\Test\Test" # dir of original data -- note that wudi has decompressed it!!!
    save_dst = r"I:\Kaggle_multimodal\Test_CNN_precompute_TEST"
    
    #data = r"I:\Kaggle_multimodal\Training" # dir of original data -- note that wudi has decompressed it!!!
    #save_dst = r"I:\Kaggle_multimodal\Test_CNN_precompute"
elif pc=="lio":
    data = r"/media/lio/Elements/chalearn/trainingset"


os.chdir(data)
if pc=="wudi":
        samples=glob("*")  # because wudi unzipped all the files already!
elif pc=="lio":
        samples=glob("*.zip")

print len(samples), "samples found"

for file_count, file in enumerate(samples):
    condition = (file_count >= 105)   
    if condition:   #wudi only used first 650 for validation !!! Lio be careful!
        print("\t Processing file " + file)
        time_tic = time.time() 
        # Create the object to access the sample
        sample = GestureSample(os.path.join(data,file))
        video = sample.get_test_data_wudi_lio()
        save_path= os.path.join(save_dst, file)
        out_file = open(save_path, 'wb')
        cPickle.dump(video, out_file, protocol=cPickle.HIGHEST_PROTOCOL)
        out_file.close()
        print "Elapsed time %d sec" % int(time.time() - time_tic)


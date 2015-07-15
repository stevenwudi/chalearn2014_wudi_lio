"""
Video classifier using a 3D deep convolutional neural network
and DBN, fusing the two result together
Data: ChaLearn 2014 gesture challenge: gesture recognition
original code by: Lionel Pigou
Code modulated by: Di Wu   stevenwudi@gmail.com
2015-06-12
"""
# various imports
from cPickle import load
from glob import glob
from time import time, localtime
from gzip import GzipFile
import os
# numpy imports
from numpy import zeros, empty, inf, float32, random, linspace

# theano imports
from theano import function, config, shared
import theano.tensor as T

# customized imports
from dbn.GRBM_DBN import GRBM_DBN
from conv3d_chalearn import conv3d_chalearn
from convnet3d import LogRegr, HiddenLayer, DropoutLayer

#  modular imports
# the hyperparameter set the data dir, use etc classes, it's important to modify it according to your need
from classes.hyperparameters import use, lr, batch, reg, mom, tr, drop,\
                                     net,  DataLoader_with_skeleton_normalisation
from functions.train_functions import _shared, _avg, write, ndtensor, print_params, lin,\
                                      training_report, epoch_report, _batch,\
                                      save_results, move_results, save_params, test_lio_skel

from convnet3d_grbm_early_fusion import convnet3d_grbm_early_fusion
# we need to parse an absolute path for HPC to load
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()
load_path = args.path

print load_path
####################################################################
print "\n%s\n\t initializing \n%s"%(('-'*30,)*2)
# source and result directory
pc = "wudi"
if pc=="wudi":
    src = r"/idiap/user/dwu/chalearn/"
    res_dir_ = r"/idiap/user/dwu/chalearn/result/"# dir of original data -- note that wudi has decompressed it!!!
elif pc=="lio":
    src = "/mnt/wd/chalearn/preproc"
    res_dir_ = "/home/lpigou/chalearn_wudi/try"

lt = localtime()
res_dir = res_dir_+"/try/"+str(lt.tm_year)+"."+str(lt.tm_mon).zfill(2)+"." \
            +str(lt.tm_mday).zfill(2)+"."+str(lt.tm_hour).zfill(2)+"."\
            +str(lt.tm_min).zfill(2)+"."+str(lt.tm_sec).zfill(2)
os.makedirs(res_dir)

######################################################################
net_convnet3d_grbm_early_fusion = convnet3d_grbm_early_fusion(src, res_dir, load_path)

net_convnet3d_grbm_early_fusion.load_params(os.path.join(load_path,'paramsbest.zip'))

x_ = _shared(empty(tr.in_shape))
y_ = _shared(empty(tr.batch_size))
y_int32 = T.cast(y_,'int32')
x_skeleton_ = _shared(empty(tr._skeleon_in_shape))

#############################
        # load normalisation constant given load_path
Mean_skel, Std_skel, Mean_CNN, Std_CNN = net_convnet3d_grbm_early_fusion.load_normalisation_constant(load_path)
loader = DataLoader_with_skeleton_normalisation(src, tr.batch_size, \
                         Mean_CNN, Std_CNN, Mean_skel, Std_skel) # Lio changed it to read from HDF5 files
######################################################################
print "\n%s\n\tcompiling\n%s"%(('-'*30,)*2)
learning_rate = shared(float32(lr.init))
apply_updates, train_model, test_model = net_convnet3d_grbm_early_fusion.build_finetune_functions(x_, y_int32, x_skeleton_,learning_rate)


######################################################################
print "\n%s\n\ttraining\n%s"%(('-'*30,)*2)

time_start = 0
best_valid = inf
lr_decay_epoch = 0
n_lr_decays = 0
train_ce, valid_ce = [], []
out_mean_all, out_std_all = [], []

res_dir = save_results(train_ce, valid_ce, res_dir, params=net_convnet3d_grbm_early_fusion.params)

save_params(net_convnet3d_grbm_early_fusion.params, res_dir)


# Wudi makes thie to explicity control the learning rate
learning_rate_map = linspace(lr.start, lr.stop, tr.n_epochs)

for epoch in xrange(tr.n_epochs):
    learning_rate.set_value(float32(learning_rate_map[epoch]))
    ce = []
    out_mean_train = []
    out_std_train = []
    print_params(net_convnet3d_grbm_early_fusion.params) 
    ####################################################################
    print "\n%s\n\t epoch %d \n%s"%('-'*30, epoch, '-'*30)
    time_start = time()
    for i in range(loader.n_iter_train):     
        #load data
        time_start_iter = time()
        loader.next_train_batch(x_, y_, x_skeleton_)
        ce_temp, out_mean_temp, out_std_temp = _batch(train_model, tr.batch_size, batch, True, apply_updates)
        ce.append(ce_temp)
        out_mean_train.append(out_mean_temp)
        out_std_train.append(out_std_temp)

        print "Training: No.%d iter of Total %d, %d s"% (i,loader.n_iter_train, time()-time_start_iter)  \
                + "\t| negative_log_likelihood "+ training_report(ce[-1]) 
    # End of Epoch
    ####################################################################
    print "\n%s\n\t End of epoch %d, \n printing some debug info.\n%s" \
        %('-'*30, epoch, '-'*30)

    train_ce.append(_avg(ce))
    out_mean_all.append(_avg(out_mean_train))
    out_std_all.append(_avg(out_std_train))
    # validate
    valid_ce.append(test_lio_skel(use, test_model, batch, drop, tr.rng, epoch, tr.batch_size, x_, y_, loader, x_skeleton_))

    # save best params
    res_dir = save_results(train_ce, valid_ce, res_dir, params=net_convnet3d_grbm_early_fusion.params, out_mean_train=out_mean_all,out_std_train=out_std_all)
    if not tr.moved: res_dir = move_results(res_dir)

    if valid_ce[-1][1] < best_valid:
        save_params(net_convnet3d_grbm_early_fusion.params, res_dir, "best")
    save_params(net_convnet3d_grbm_early_fusion.params, res_dir)

    if valid_ce[-1][1] < best_valid:
        best_valid = valid_ce[-1][1]

    # epoch report
    
    epoch_report(epoch, best_valid, time()-time_start, learning_rate.get_value(borrow=True),\
        train_ce[-1], valid_ce[-1], res_dir)

    # decay the learning rate
    
    loader.shuffle_train()






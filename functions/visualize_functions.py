
from gzip import GzipFile
from cPickle import dump, load
import os

import matplotlib.pyplot as pl
#import pylab as pl
from math import sqrt, ceil, floor
import numpy as n

def gray_body_conv1():
    print " draw gray_body_conv1"
    if os.path.isfile('paramsbest.zip'):
        file = GzipFile("paramsbest.zip", "rb")
    else:
        file = GzipFile("params.zip", "rb")
    par = load(file)
    file.close()
    W = par[0]    
    # we need to flip here because the best parameter by
    # Lio was using cudaconv, different from Theano's conv op
    # see: http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html
    if len(W.shape) >4:
        W = W[:, :, :, ::-1, ::-1]

    filters = W[:, 0, 0, :, :,]


    _save_path = r'visualization\\gray_body_conv1_5_5.eps'
    draw_kernel(filters, _save_path)

def gray_hand_conv1():
    print " draw gray_body_conv1"
    if os.path.isfile('paramsbest.zip'):
        file = GzipFile("paramsbest.zip", "rb")
    else:
        file = GzipFile("params.zip", "rb")
    par = load(file)
    file.close()
    W = par[0]    
    # we need to flip here because the best parameter by
    # Lio was using cudaconv, different from Theano's conv op
    # see: http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html
    if len(W.shape) >4:
        W = W[:, :, :, ::-1, ::-1]
    
    filters = W[:, 1, 0, :, :,]
    _save_path = r'visualization\\gray_hand_conv1_5_5.eps'
    draw_kernel(filters, _save_path)

def depth_body_conv1():
    print " draw depth_body_conv1"
    if os.path.isfile('paramsbest.zip'):
        file = GzipFile("paramsbest.zip", "rb")
    else:
        file = GzipFile("params.zip", "rb")
    par = load(file)
    file.close()
    W = par[2]    
    # we need to flip here because the best parameter by
    # Lio was using cudaconv, different from Theano's conv op
    # see: http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html
    if len(W.shape) >4:
        W = W[:, :, :, ::-1, ::-1]

    filters = W[:, 0, 0, :, :,]

    _save_path = r'visualization\\depth_body_conv1_5_5.eps'
    draw_kernel(filters, _save_path)

def depth_hand_conv1():
    print " draw depth_hand_conv1"
    if os.path.isfile('paramsbest.zip'):
        file = GzipFile("paramsbest.zip", "rb")
    else:
        file = GzipFile("params.zip", "rb")
    par = load(file)
    file.close()
    W = par[2]    
    # we need to flip here because the best parameter by
    # Lio was using cudaconv, different from Theano's conv op
    # see: http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html
    if len(W.shape) >4:
        W = W[:, :, :, ::-1, ::-1]
    
    filters = W[:, 1, 0, :, :,]
    _save_path = r'visualization\\depth_hand_conv1_5_5.eps'
    draw_kernel(filters, _save_path)

def draw_kernel(filters, _save_path):
    
    filter_start = 0
    fignum = 0
    num_filters = filters.shape[0]
    filters = (filters - filters.min())/(filters.max()-filters.min())

    FILTERS_PER_ROW = 16
    MAX_ROWS = 16
    MAX_FILTERS = FILTERS_PER_ROW * MAX_ROWS

    f_per_row = FILTERS_PER_ROW 
    filter_end = min(filter_start+MAX_FILTERS, num_filters)
    filter_rows = int(ceil(float(filter_end - filter_start) / f_per_row))
    
    filter_size = int((filters.shape[1]))
    fig = pl.figure(fignum)
    #fig.text(.5, .95, '%s %dx%d filters %d-%d' % (_title, filter_size, filter_size, filter_start, filter_end-1), horizontalalignment='center') 
    num_filters = filter_end - filter_start

    bigpic = n.zeros((filter_size * filter_rows + filter_rows + 1, filter_size * f_per_row + f_per_row + 1), dtype=n.single)

    for m in xrange(filter_start,filter_end ):
        filter_pic = filters[m, :,:]
        y, x = (m - filter_start) / f_per_row, (m - filter_start) % f_per_row
        bigpic[ 1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                1 + (1 + filter_size) * x:1 + (1 + filter_size) * x + filter_size] = filter_pic
                
    pl.xticks([])
    pl.yticks([])

    pl.imshow(bigpic, cmap=pl.cm.gray, interpolation='nearest')
    pl.savefig(_save_path, format='eps',bbox_inches='tight')
    pl.show()


def draw_original():
    print " draw original"
    _save_path = r'visualization\\original.eps'

    pc = "wudi"
    if pc=="wudi":
        src = r"I:\Kaggle_multimodal\Training_prepro"
        res_dir_ = r"I:\Kaggle_multimodal\result"# dir of original data -- note that wudi has decompressed it!!!
    elif pc=="lio":
        src = "/mnt/wd/chalearn/preproc"
        res_dir_ = "/home/lpigou/chalearn_wudi/try"

    import h5py
    from math import floor
    file = h5py.File(src+"/data%d.hdf5", "r", driver="family", memb_size=2**32-1)
    x_train = file["x_train"]
    x_valid = file["x_valid"]
    y_train = file["y_train"]
    y_valid = file["y_valid"]

    # which frame to plot can be changed inside the function
    # here we have chosen a random frame to plot
    # because we save it as h5py, top 1000 frames is chosen as random

    frame_to_plot = ceil(n.random.rand() *1000 )
    images = x_train[frame_to_plot,:,:,:,:,:]
    
    f_per_row = images.shape[2]
    filter_rows = images.shape[0] * images.shape[1]   
    num_filters = images.shape[0] * images.shape[1] * images.shape[2]
    filter_size = images.shape[-1]    

    bigpic = n.zeros((filter_size * filter_rows + filter_rows + 1, filter_size * f_per_row + f_per_row + 1), dtype=n.single)

    for rgb_depth in range(images.shape[0]):
        for body_hand in range(images.shape[1]):
            for frame_num in range(images.shape[2]):
                filter_pic = images[rgb_depth, body_hand, frame_num, :,:]
                x = frame_num
                y = rgb_depth * 2  + body_hand
                bigpic[ 1 + (1 + filter_size) * y:1 + (1 + filter_size) * y + filter_size,
                        1 + (1 + filter_size) * x:1 + (1 + filter_size) * x + filter_size] = filter_pic
                
    pl.xticks([])
    pl.yticks([])
    pl.imshow(bigpic, cmap=pl.cm.gray, interpolation='nearest')
    pl.savefig(_save_path, format='eps', bbox_inches='tight')
    pl.show()

def plot_confusion_matrix():
    """"plot the confusion matrix"""
    from sklearn.metrics import confusion_matrix
    from numpy import ones, array, prod, zeros, empty, inf, float32, random
    import numpy
    import os
    import zipfile
    import shutil
    import csv
    import re
    print "plot confusion matrix"
    _save_path = r'visualization\\cm.eps'
    truth_dir=r'I:\Kaggle_multimodal\Test_label'
    prediction_dir=r'C:\Users\PC-User\Documents\GitHub\chalearn2014_wudi_lio\CNN_test_pred_combine_sk_cnn'
    # Get the list of samples from ground truth
    gold_list = os.listdir(truth_dir)

    # For each sample on the GT, search the given prediction
    numSamples=0.0;
    score=0.0;

    begin_add=0
    end_add=0

    for gold in gold_list:
        # Avoid double check, use only labels file
        if not gold.lower().endswith("_labels.csv"):
            continue

        # Build paths for prediction and ground truth files
        sampleID=re.sub('\_labels.csv$', '', gold)
        labelsFile = os.path.join(truth_dir, sampleID + "_labels.csv")
        dataFile = os.path.join(truth_dir, sampleID + "_data.csv")
        predFile = os.path.join(prediction_dir, sampleID + "_prediction.csv")
                # Get the number of frames for this sample
        with open(dataFile, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                seqlenght=int(row[0])
            del filereader
        # Get the number of frames for this sample
        """ Evaluate this sample agains the ground truth file """
        maxGestures=20

        # Get the list of gestures from the ground truth and frame activation
        gtGestures = []
        binvec_gt = numpy.zeros((maxGestures, seqlenght))
        with open(labelsFile, 'rb') as csvfilegt:
            csvgt = csv.reader(csvfilegt)
            for row in csvgt:
                binvec_gt[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
                gtGestures.append(int(row[0]))

        # Get the list of gestures from prediction and frame activation
        predGestures = []
        binvec_pred = numpy.zeros((maxGestures, seqlenght))
        with open(predFile, 'rb') as csvfilepred:
            csvpred = csv.reader(csvfilepred)
            for row in csvpred:
                binvec_pred[int(row[0])-1, int(row[1])-1+begin_add:int(row[2])-1+end_add] = 1
                predGestures.append(int(row[0]))

        # Get the list of gestures without repetitions for ground truth and predicton
        gtGestures = numpy.unique(gtGestures)
        predGestures = numpy.unique(predGestures)

        # Find false positives
        falsePos=numpy.setdiff1d(gtGestures, numpy.union1d(gtGestures,predGestures))

        # Get overlaps for each gesture
        overlaps = numpy.zeros(maxGestures)
        for idx in gtGestures:
            intersec = sum(binvec_gt[idx-1] * binvec_pred[idx-1])
            aux = binvec_gt[idx-1] + binvec_pred[idx-1]
            union = sum(aux > 0)
            overlaps[idx-1] = intersec/union
        
        print len(overlaps)
        # Use real gestures and false positive gestures to calculate the final score
   
    
    cm = confusion_matrix(y_test, y_pred)
    # Show confusion matrix in a separate window
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.ylabel('True label')
    pl.xlabel('Predicted label')
    pl.savefig(_save_path, format='eps', bbox_inches='tight')
    pl.show()


def plot_cnn_error_rate():
    import numpy
    _save_path = r'visualization\\training_error.eps'

    validation_error = numpy.array([ 57.297,  52.96 ,  51.771,  48.458,  46.633,  45.878,  44.588,
    43.895,  43.834,  42.753,  42.831,  42.511,  42.157,  41.457,
    41.437,  40.649,  41.019,  41.056,  40.82 ,  40.1  ,  40.009,
    39.911,  40.295,  40.187,  40.322,  40.369,  39.925,  39.719,
    39.574,  39.648,  39.456,  39.544,  39.514,  39.595,  39.753,
    39.194,  39.221,  39.271,  39.254,  39.275,  39.423,  39.062,
    39.18 ,  39.14 ])
    validation_error /=100
    training_error = numpy.array([ 64.626,  57.508,  54.064,  51.562,  49.923,  47.967,  46.411,
        44.787,  43.796,  42.879,  41.97 ,  40.882,  39.794,  39.023,
        38.145,  37.494,  36.609,  35.796,  35.651,  35.267,  34.468,
        34.035,  33.516,  33.698,  33.381,  34.155,  33.195,  32.559,
        31.847,  31.676,  31.288,  31.145,  31.189,  31.186,  30.565,
        30.402,  30.038,  31.202,  30.084,  30.167,  29.523,  29.316,
        28.953,  29.184])
    training_error /= 100

    training_cost = numpy.array([ 2.241,  1.78 ,  1.612,  1.505,  1.434,  1.36 ,  1.306,  1.248,
        1.209,  1.184,  1.154,  1.118,  1.08 ,  1.059,  1.029,  1.008,
        0.982,  0.956,  0.954,  0.943,  0.92 ,  0.908,  0.893,  0.899,
        0.889,  0.915,  0.887,  0.866,  0.846,  0.843,  0.83 ,  0.824,
        0.829,  0.828,  0.809,  0.807,  0.797,  0.834,  0.801,  0.802,
        0.784,  0.776,  0.766,  0.775])
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    plt.clf()
    plt.plot(range(validation_error.shape[-1]), validation_error, color='c',linewidth=2.0, label="validation error")
    plt.plot(range(training_error.shape[-1]), training_error, color='r',linewidth=2.0, label="training error" )
    #plt.plot(range(training_cost.shape[-1]), training_cost, color='r',linewidth=2.0, label="training cost: negative loglikelihood")
    plt.legend(prop={'size':20})
    plt.ylabel('frame error rate',  fontsize=20)
    plt.xlabel('epochs',  fontsize=20)
    from pylab import savefig
    savefig(_save_path, format='eps', bbox_inches='tight')
    plt.show()

def plot_sk_error_rate():
    import numpy
    _save_path = r'visualization\\training_error_sk.eps'
    validation_error = numpy.load('result\\validation_loss.npy')
    training_error = numpy.load('result\\minibatch_avg_cost_train.npy')

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20)
    plt.clf()
    plt.plot(range(validation_error.shape[-1]), validation_error, color='c',linewidth=2.0, label="validation error")
    #plt.plot(range(training_error.shape[-1]), training_error, color='r',linewidth=2.0, label="training cost: negative loglikelihood")
    plt.legend(prop={'size':20})
    plt.ylabel('frame error rate', fontsize=20)
    plt.xlabel('epochs', fontsize=20)
    from pylab import savefig
    savefig(_save_path, format='eps', bbox_inches='tight')
    plt.show()
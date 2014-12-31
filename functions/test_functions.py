from numpy import ones, array, prod, zeros, empty, inf, float32, random
import numpy
import os
import zipfile
import shutil
import csv
import re


def viterbi_path_log(prior, transmat, observ_likelihood):
    """ Viterbi path decoding
    Wudi first implement the forward pass.
    Future works include forward-backward encoding
    input: prior probability 1*N...
    transmat: N*N
    observ_likelihood: N*T
    """
    T = observ_likelihood.shape[-1]
    N = observ_likelihood.shape[0]

    path = numpy.zeros(T, dtype=numpy.int32)
    global_score = numpy.zeros(shape=(N,T))
    predecessor_state_index = numpy.zeros(shape=(N,T), dtype=numpy.int32)

    t = 1
    global_score[:, 0] = prior + observ_likelihood[:, 0]
    # need to  normalize the data
    
    for t in range(1, T):
        for j in range(N):
            temp = global_score[:, t-1] + transmat[:, j] + observ_likelihood[j, t]
            global_score[j, t] = max(temp)
            predecessor_state_index[j, t] = temp.argmax()

    path[T-1] = global_score[:, T-1].argmax()
    
    for t in range(T-2, -1, -1):
        path[t] = predecessor_state_index[ path[t+1], t+1]

    return [path, predecessor_state_index, global_score]

def viterbi_colab_states(path, global_score, state_no = 5, threshold=-3, mini_frame=15):
    """
    Clean the viterbi path output according to its global score,
    because some are out of the vocabulary
    """
    # just to accommodate some frame didn't start right from the begining
    all_label = state_no * 20 # 20 vocabularies
    #start_label = numpy.concatenate((range(0,all_label,state_no), range(1,all_label,state_no)))
    #end_label   = numpy.concatenate((range(4,all_label,state_no), range(3,all_label,state_no)))
    start_label = range(0,all_label,state_no)
    end_label   = range(4,all_label,state_no)
    begin_frame = []
    end_frame = []
    pred_label = []

    frame = 1
    while(frame < path.shape[-1]-1):
        if path[frame-1]==all_label and path[frame] in start_label:
            begin_frame.append(frame)
            # python integer divsion will do the floor for us :)
            pred_label .append( path[frame]/state_no + 1)
            while(frame < path.shape[-1]-1):
                if path[frame] in end_label and path[frame+1]==all_label:
                    end_frame.append(frame)
                    break
                else:
                    frame += 1
        frame += 1
        
    

    end_frame = numpy.array(end_frame)
    begin_frame = numpy.array(begin_frame)
    pred_label= numpy.array(pred_label)
    # risky hack! just for validation file 663
    if len(begin_frame)> len(end_frame):
        begin_frame = begin_frame[:-1]
    elif len(begin_frame)< len(end_frame):# risky hack! just for validation file 668
        end_frame = end_frame[1:]
    ## First delete the predicted gesture less than 15 frames
    frame_length = end_frame - begin_frame
    ## now we delete the gesture outside the vocabulary by choosing
    ## frame number small than mini_frame
    mask = frame_length > mini_frame
    begin_frame = begin_frame[mask]
    end_frame = end_frame[mask]
    pred_label = pred_label[mask]


    Individual_score = []
    for idx, g in enumerate(begin_frame):
            score_start = global_score[path[g], g]
            score_end = global_score[path[end_frame[idx]], end_frame[idx]]
            Individual_score.append(score_end - score_start)
    ## now we delete the gesture outside the vocabulary by choosing
    ## score lower than a threshold
    Individual_score = numpy.array(Individual_score)
    frame_length = end_frame - begin_frame
    # should be length independent
    Individual_score = Individual_score/frame_length

    mask = Individual_score > threshold
    begin_frame = begin_frame[mask]
    end_frame = end_frame[mask]
    pred_label = pred_label[mask]
    Individual_score = Individual_score[mask]
    

    return [pred_label, begin_frame, end_frame, Individual_score, frame_length]

def imdisplay(im):
    """ display grayscale images
    """
    im_min = im.min()
    im_max = im.max()
    return (im - im_min) / (im_max -im_min)

def gesture_overlap_csv(csvpathgt, csvpathpred, seqlenght, begin_add, end_add):
    """ Evaluate this sample agains the ground truth file """
    maxGestures=20

    # Get the list of gestures from the ground truth and frame activation
    gtGestures = []
    binvec_gt = numpy.zeros((maxGestures, seqlenght))
    with open(csvpathgt, 'rb') as csvfilegt:
        csvgt = csv.reader(csvfilegt)
        for row in csvgt:
            binvec_gt[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
            gtGestures.append(int(row[0]))

    # Get the list of gestures from prediction and frame activation
    predGestures = []
    binvec_pred = numpy.zeros((maxGestures, seqlenght))
    with open(csvpathpred, 'rb') as csvfilepred:
        csvpred = csv.reader(csvfilepred)
        for row in csvpred:
            binvec_pred[int(row[0])-1, int(row[1])-1+begin_add:int(row[2])-1+end_add] = 1
            predGestures.append(int(row[0]))

    # Get the list of gestures without repetitions for ground truth and predicton
    if len(numpy.unique(gtGestures)) != len(gtGestures):
        print "not unique!"
    gtGestures = numpy.unique(gtGestures)
    predGestures = numpy.unique(predGestures)

    # Find false positives
    falsePos=numpy.setdiff1d(gtGestures, numpy.union1d(gtGestures,predGestures))

    # Get overlaps for each gesture
    overlaps = []
    for idx in gtGestures:
        intersec = sum(binvec_gt[idx-1] * binvec_pred[idx-1])
        aux = binvec_gt[idx-1] + binvec_pred[idx-1]
        union = sum(aux > 0)
        overlaps.append(intersec/union)

    # Use real gestures and false positive gestures to calculate the final score
    return sum(overlaps)/(len(overlaps)+len(falsePos))

def evalGesture(prediction_dir,truth_dir, begin_add=0, end_add=0):
    """ Perform the overlap evaluation for a set of samples """
    worseVal=10000

    # Get the list of samples from ground truth
    gold_list = os.listdir(truth_dir)

    # For each sample on the GT, search the given prediction
    numSamples=0.0;
    score=0.0;
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
                numFrames=int(row[0])
            del filereader

        # Get the score
        numSamples+=1
        
        score_temp = gesture_overlap_csv(labelsFile, predFile, numFrames, begin_add, end_add)
        print "Sample ID: %s, score %f" %(sampleID,score_temp)
        score+=score_temp
    return score/numSamples




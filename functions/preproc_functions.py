"""
functions for preprocessing skeleton and video
"""
from cPickle import dump
from glob import glob
from random import shuffle
import cv2
import os
import sys
import shutil
import errno
import gzip
from itertools import tee, islice
import numpy
from numpy import *
from numpy import linalg
from numpy.random import RandomState

def to_grayscale(v):
    s = v.shape
    v = v.reshape(prod(s[:-2]),s[-2], s[-1])
    v = cv2.cvtColor(v,cv2.cv.CV_RGB2GRAY)
    return v.reshape(s[:-1])

def cut_img(img,s):
    if s==0: return img 
    return img[s:-s,s:-s]

def proc_skelet_wudi(gestures, used_joints, gesture, STATE_NO=5, NEUTRUAL_SEG_LENGTH=8):
    """
    Extract original features, including the neutral features
    """
    startFrame = gesture[1]
    endFrame = gesture[2]
    Pose, corrupt = Extract_feature_UNnormalized(gestures, used_joints, startFrame, endFrame)           
 
    #####################################################################
    # Articulated poses extraction
    #####################################################################
    """
    Feature design: (1) relational pairwise (2) velocity (3) acceleration
    Reference:
    The Moving Pose: An Efficient 3D Kinematics Descriptor for Low-Latency Action Recognition and Detection
    ICCV 2013
    """
    njoints = len(used_joints)
    Feature_gesture = Extract_feature_Accelerate(Pose, njoints)      
    #label information
    gestureID = gesture[0]
    Targets_gesture = numpy.zeros( shape=(Feature_gesture.shape[0], STATE_NO*20+1))
    fr_no = Feature_gesture.shape[0]
    count = 0
    for i in range(STATE_NO):  #HMM states force alignment
        begin_fr = numpy.round(fr_no* i /STATE_NO) + 1
        end_fr = numpy.round( fr_no*(i+1) /STATE_NO) 
        #print "begin: %d, end: %d"%(begin_fr-1, end_fr)
        seg_length=end_fr-begin_fr + 1
        targets = numpy.zeros( shape =(STATE_NO*20+1,1))
        targets[ i + STATE_NO*(gestureID-1)] = 1
        begin_frame = count
        end_frame = count+seg_length
        Targets_gesture[begin_frame:end_frame, :]= numpy.tile(targets.T,(seg_length, 1))
        count=count+seg_length

    #####################################################################
    # Articulated poses extraction--neutral Poses
    #####################################################################
    # pre-allocating the memory
    Feature_neutral = numpy.zeros(shape=(8, Feature_gesture.shape[1]),dtype=numpy.float32)
    Targets_neutral = numpy.zeros( shape=(8, STATE_NO*20+1))
    count = 0
    if startFrame-NEUTRUAL_SEG_LENGTH-1> 0:
        Skeleton_matrix, c= Extract_feature_UNnormalized(gestures, used_joints, startFrame-NEUTRUAL_SEG_LENGTH, startFrame-1)              
        # if corrupt, we don't need it in the main loop
        Feature = Extract_feature_Accelerate(Skeleton_matrix, njoints)
        begin_frame = count
        end_frame = count+NEUTRUAL_SEG_LENGTH-4 # in effect it only generate 4 frames because acceleration requires 5 frames
        Feature_neutral[begin_frame:end_frame,:] = Feature
        Targets_neutral[begin_frame:end_frame, -1] = 1
        count=end_frame

    ## extract last 5 frames
    if endFrame+NEUTRUAL_SEG_LENGTH+1 < gestures.getNumFrames():
        Skeleton_matrix, c= Extract_feature_UNnormalized(gestures, used_joints, endFrame+1, endFrame+NEUTRUAL_SEG_LENGTH)              
        Feature = Extract_feature_Accelerate(Skeleton_matrix, njoints)
        begin_frame = count
        end_frame = count+NEUTRUAL_SEG_LENGTH-4 # in effect it only generate 4 frames because acceleration requires 5 frames
        Feature_neutral[begin_frame:end_frame,:] = Feature
        Targets_neutral[begin_frame:end_frame, -1] = 1

    Feature = numpy.concatenate((Feature_gesture, Feature_neutral), axis=0)
    Targets = numpy.concatenate((Targets_gesture, Targets_neutral), axis=0)
    return Feature, Targets, corrupt


def Extract_feature_UNnormalized(smp, used_joints, startFrame, endFrame):
    """
    Extract original features
    """
    frame_num = 0 
    Skeleton_matrix  = numpy.zeros(shape=(endFrame-startFrame+1, len(used_joints)*3))
    for numFrame in range(startFrame,endFrame+1):                    
        # Get the Skeleton object for this frame
        skel=smp.getSkeleton(numFrame)
        for joints in range(len(used_joints)):
            Skeleton_matrix[frame_num, joints*3: (joints+1)*3] = skel.joins[used_joints[joints]][0]
        frame_num += 1
    
    if numpy.allclose(sum(sum(numpy.abs(Skeleton_matrix))),0):
        corrupt = True
    else:
        corrupt = False

    return Skeleton_matrix, corrupt

def Extract_feature_Accelerate(Pose, njoints):
    #Fcc
    FeatureNum = 0
    Fcc =  numpy.zeros(shape=(Pose.shape[0], njoints * (njoints-1)/2*3))
    for joints1 in range(njoints-1):
        for joints2 in range(joints1+1,njoints):
            Fcc[:, FeatureNum*3:(FeatureNum+1)*3] = Pose[:, joints1*3:(joints1+1)*3]-Pose[:, joints2*3:(joints2+1)*3];
            FeatureNum += 1
            
    #F_cp --joint velocities
    FeatureNum = 0
    Fcp = numpy.zeros(shape=(Pose.shape[0]-2, njoints **2*3))
    for joints1 in range(njoints):
        for joints2 in range(njoints):
            Fcp[:, FeatureNum*3: (FeatureNum+1)*3] = Pose[2:,joints1*3:(joints1+1)*3]-Pose[0:-2,joints2*3:(joints2+1)*3]
            FeatureNum += 1
    
    #F_ca --joint accelerations
    FeatureNum = 0
    Fca= numpy.zeros(shape=(Pose.shape[0]-4, njoints **2*3))
    for joints1 in range(njoints):
        for joints2 in range(njoints):
            Fca[:, FeatureNum*3: (FeatureNum+1)*3] = Pose[4:,joints1*3:(joints1+1)*3] + Pose[0:-4,joints2*3:(joints2+1)*3] - 2 * Pose[2:-2,joints2*3:(joints2+1)*3]
            FeatureNum += 1
              
    #Instead of initial frame as in the paper Eigenjoints-based action recognition using
    #naive-bayes-nearest-neighbor, we use final frame because it's better initiated
    # F_cf

    Features = numpy.concatenate( (Fcc[0:-4, :], Fcp[0:-2,:], Fca), axis = 1)
    return Features


def proc_user(user,krn=5):
    import cv2
    user[user==1]=255
    for i,u in enumerate(user):
        u = cv2.medianBlur(u, krn)
        user[i] = u
    user[user>0] = 1
    return user

def proc_skelet(skelet, _3D=True):
    corrupt = False
    traj2D,traj3D,ori,pheight,hand,center = None,None,None,None,None,None
    l = len(skelet)
    phl, phr, ph, pc = [empty((2,l)) for _ in range(4)]
    if _3D:
        whl, whr, wh, wc = [empty((3,l)) for _ in range(4)]
        ohl, ohr = [empty((4,l)) for _ in range(2)]

    for i,skel in enumerate(skelet):
        pix = skel.getPixelCoordinates()
        if _3D:
            world = skel.getWorldCoordinates()
            ori = skel.getJoinOrientations()
            whl[:,i] = array(world['HandLeft'])
            whr[:,i] = array(world['HandRight'])
            ohl[:,i] = array(ori['HandLeft'])
            ohr[:,i] = array(ori['HandRight'])
            wh[:,i] = array(world['Head'])
            wc[:,i] = array(world['HipCenter'])
        phl[:,i] = array(pix['HandLeft'])
        phr[:,i] = array(pix['HandRight'])
        ph[:,i] = array(pix['Head'])
        pc[:,i] = array(pix['HipCenter'])

    if count_nonzero(phl) < 10*2: 
        corrupt = True
    elif _3D:
        phl,phr,ph,pc,whl,whr,wh,wc = [smooth(s) for s in \
                                                phl,phr,ph,pc,whl,whr,wh,wc]
        ohl,ohr = [smooth(s,3) for s in ohl,ohr]
    else:
        phl,phr,ph,pc = [smooth(s) for s in phl,phr,ph,pc]

    phl_y = phl[1][phl[1].nonzero()]
    phr_y = phr[1][phr[1].nonzero()]

    hand = "left" if phl_y.mean() < phr_y.mean() else "right"

    if hand=="left": 
        traj2D=phl
        if _3D: traj3D,ori = whl, ohl
    else: 
        traj2D = phr
        if _3D: traj3D,ori = whr, ohr

    pheight = empty((l,),dtype="float32")
    # wheight = empty((l,),dtype="float32")
    for i in xrange(l): pheight[i] = linalg.norm(pc[:,i]-ph[:,i])
    #     wheight[i] = linalg.norm(wc[:,i]-wh[:,i])
    pheight = pheight.mean()
    # wheight = wheight.mean()
    # pheight = array([linalg.norm(pc[:,i]-ph[:,i]) for i in range(l)]).mean()
    if _3D:
        wheight = array([linalg.norm(wc[:,i]-wh[:,i]) for i in range(l)]).mean()
        traj3D = (wh-traj3D)/wheight
        if hand=="left": traj3D[0] *= -1
        traj3D[0]    = normalize(traj3D[0],-0.06244 , 0.61260)
        traj3D[1]    = normalize(traj3D[1], 0.10840 , 1.60145)
        traj3D[2]    = normalize(traj3D[2],-0.09836 , 0.76818)
        ori[0]       = normalize(ori[0]   , 0.30971 , 1.00535)
        ori[1]       = normalize(ori[1]   ,-0.27595 , 0.19067)
        ori[2]       = normalize(ori[2]   ,-0.10569 , 0.12660)
        ori[3]       = normalize(ori[3]   ,-0.32591 , 0.48749)

        traj3D,ori = [d.astype("uint8") for d in traj3D,ori]
    
    center = pc.mean(1)

    return (traj2D,traj3D,ori,pheight,hand,center), corrupt

def smooth(x,window_len=5,window='flat'):
    if x.ndim > 2:
            raise ValueError, "smooth only accepts 1 or 2 dimension arrays."
    if x.ndim ==2 :
        for i in range(x.shape[0]):
            x[i] = smooth(x[i],window_len,window)
        return x
    if x.shape[0] < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."
    if window_len<3:
            return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s=r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    # if window == 'flat': #moving average
    #         w=ones(window_len,'d')
    # else:  
    #         w=eval(window+'(window_len)')
    w_smooth=ones(5,'d')/5.
    y=convolve(w_smooth,s,mode='same')
    return y[window_len:-window_len+1]

def proc_depth_wudi(depth, user, user_o, skelet, NEUTRUAL_SEG_LENGTH=8, vid_shape_hand = (128, 128)):
    # settings
    thresh_noise = 200
    scaler = 4
    corrupt = False
    traj2D,traj3D,ori,pheight,hand,center = skelet
    #stats
    depth = 255 - depth
    user_depth = depth[user_o==1]
    #import matplotlib.pyplot as plt
    #for f in range(user_o.shape[0]):
    #    cv2.imshow("user", user_o[f,:,:])
    #    cv2.waitKey(200)
    #print histogram(user_depth,100)
    # thresh_noise = user_depth.max()
    med = average(user_depth)
    # med = 255 - med
    std = user_depth.std()

    depth_b = cut_body(depth.copy(), center, pheight, hand)
    user_b = cut_body(user.copy(), center, pheight, hand)

    depth_h = cut_hand(depth.copy(), traj2D, hand)
    user_h = cut_hand(user.copy(), traj2D, hand)

    new_depth = empty((2,)+(user.shape[0], vid_shape_hand[0], vid_shape_hand[1]), dtype="uint8")

    for i,(depth,user) in enumerate(((depth_b,user_b),(depth_h,user_h))):
        depth[depth>thresh_noise] = 0
        thresh_depth = med-3*std        
        # print med-3*std
        # thresh_depth = 100
        #depth[depth<thresh_depth] = thresh_depth
        depth = depth-thresh_depth
        depth = clip(depth*scaler, 0, 255)
        # depth = depth - depth.mean()
        # depth = norm_vid(depth)
        depth = depth.astype("uint8")
        depth = medianblur(depth)
        new_depth[i] = depth

    depth = new_depth

    new_user = empty((2,)+(user.shape[0], vid_shape_hand[0], vid_shape_hand[1]), dtype="uint8")
    new_user[0] = user_b
    new_user[1] = user_h

    # wudi added the following lines to extract 5 continuous frames together
    TOTAL_CUDBOID = (depth.shape[1]-4) + (4+4 - 2*NEUTRUAL_SEG_LENGTH)
    Feature = empty(shape=(depth.shape[0],TOTAL_CUDBOID, 5, depth.shape[2], depth.shape[3]))
    #gesture cuboid
    frame_count = 0
    for frame_no in range( depth.shape[1]-2*NEUTRUAL_SEG_LENGTH -4 ):
        Feature[:, frame_count, : , :, :] = depth[:, frame_no:frame_no+5, :, :]
        frame_count += 1
    # neutral frames cuboid --before
    for frame_no in range( depth.shape[1]-2*NEUTRUAL_SEG_LENGTH -4, depth.shape[1]-NEUTRUAL_SEG_LENGTH -8):
        Feature[:, frame_count, : , :, :] = depth[:, frame_no:frame_no+5, :, :]
        frame_count += 1
    # neutral frames cuboid --after
    for frame_no in range(depth.shape[1]-NEUTRUAL_SEG_LENGTH -8 ,depth.shape[1] - 12):
        Feature[:, frame_count, : , :, :] = depth[:, frame_no:frame_no+5, :, :]
        frame_count += 1

    return new_user, Feature.astype("uint8"), corrupt

def proc_depth_test_wudi_lio(depth, user, user_o, skelet,  vid_shape_hand = (128, 128)):
    # settings
    thresh_noise = 200
    scaler = 4
    corrupt = False
    traj2D,traj3D,ori,pheight,hand,center = skelet
    #stats
    depth = 255 - depth
    user_depth = depth[user_o==1]

    #import matplotlib.pyplot as plt
    #for f in range(user_o.shape[0]):
    #    cv2.imshow("user", depth[f,:,:])
    #    cv2.waitKey(200)
    #print histogram(user_depth,100)

    med = average(user_depth)
    # med = 255 - med
    std = user_depth.std()

    depth_b = cut_body(depth.copy(), center, pheight, hand)
    user_b = cut_body(user.copy(), center, pheight, hand)

    depth_h = cut_hand(depth.copy(), traj2D, hand)
    user_h = cut_hand(user.copy(), traj2D, hand)

    new_depth = empty((2,)+(user.shape[0], vid_shape_hand[0], vid_shape_hand[1]), dtype="uint8")

    for i,(depth,user) in enumerate(((depth_b,user_b),(depth_h,user_h))):
        depth[depth>thresh_noise] = 0
        thresh_depth = med-3*std        
        # print med-3*std
        # thresh_depth = 100
        #depth[depth<thresh_depth] = thresh_depth
        depth = depth-thresh_depth
        depth = clip(depth*scaler, 0, 255)
        # depth = depth - depth.mean()
        # depth = norm_vid(depth)
        depth = depth.astype("uint8")
        depth = medianblur(depth)
        new_depth[i] = depth

    depth = new_depth

    new_user = empty((2,)+(user.shape[0], vid_shape_hand[0], vid_shape_hand[1]), dtype="uint8")
    new_user[0] = user_b
    new_user[1] = user_h

    #cv2.destroyAllWindows()
    #for f in range(depth.shape[1]):
    #    cv2.imshow("user", depth[0,f,:,:])
    #    cv2.waitKey(200)
    #for f in range(depth.shape[1]):
    #    cv2.imshow("user", depth[1,f,:,:])
    #    cv2.waitKey(200)


    TOTAL_CUDBOID = depth.shape[1]-3
    Feature = empty(shape=(depth.shape[0],TOTAL_CUDBOID, 4, depth.shape[2], depth.shape[3]))
    #gesture cuboid
    frame_count = 0
    for frame_no in range( depth.shape[1]-3):
        Feature[:, frame_no, : , :, :] = depth[:, frame_no:frame_no+4, :, :]

    return new_user, Feature.astype("uint8"), corrupt


def proc_gray_wudi(gray, user, skelet, NEUTRUAL_SEG_LENGTH=8, vid_shape_hand = (128, 128)):
    corrupt = False
    traj2D,traj3D,ori,pheight,hand,center = skelet

    gray_b = cut_body(gray.copy(), center, pheight, hand)
    gray_h = cut_hand(gray.copy(), traj2D, hand)
    new_gray = empty((2,)+(user.shape[0], vid_shape_hand[0], vid_shape_hand[1]), dtype="uint8")
    new_gray[0] = gray_b
    new_gray[1] = gray_h
    gray = new_gray
    # wudi added the following lines to extract 5 continuous frames together

    TOTAL_CUDBOID = (gray.shape[1]-4) + (4+4 - 2*NEUTRUAL_SEG_LENGTH)
    Feature = empty(shape=(gray.shape[0], TOTAL_CUDBOID,5, gray.shape[2], gray.shape[3]))
    #gesture cuboid
    frame_count = 0
    for frame_no in range( gray.shape[1]-2*NEUTRUAL_SEG_LENGTH -4 ):
        Feature[:, frame_count, : , :, :] = gray[:, frame_no:frame_no+5, :, :]
        frame_count += 1
    # neutral frames cuboid --before
    for frame_no in range( gray.shape[1]-2*NEUTRUAL_SEG_LENGTH -4, gray.shape[1]-NEUTRUAL_SEG_LENGTH -8):
        Feature[:, frame_count, : , :, :] = gray[:, frame_no:frame_no+5, :, :]
        frame_count += 1
    # neutral frames cuboid --after
    for frame_no in range(gray.shape[1]-NEUTRUAL_SEG_LENGTH -8 ,gray.shape[1] - 12):
        Feature[:, frame_count, : , :, :] = gray[:, frame_no:frame_no+5, :, :]
        frame_count += 1

    return Feature.astype("uint8"), corrupt

def proc_gray_test_wudi_lio(gray, user, skelet,  vid_shape_hand = (128, 128)):
    corrupt = False
    traj2D,traj3D,ori,pheight,hand,center = skelet

    gray_b = cut_body(gray.copy(), center, pheight, hand)
    gray_h = cut_hand(gray.copy(), traj2D, hand)
    new_gray = empty((2,)+(user.shape[0], vid_shape_hand[0], vid_shape_hand[1]), dtype="uint8")
    new_gray[0] = gray_b
    new_gray[1] = gray_h
    gray = new_gray
    # wudi added the following lines to extract 5 continuous frames together

    TOTAL_CUDBOID = gray.shape[1]-3
    Feature = empty(shape=(gray.shape[0], TOTAL_CUDBOID, 4, gray.shape[2], gray.shape[3]))
    #gesture cuboid
    frame_count = 0
    for frame_no in range( gray.shape[1]-3 ):
        Feature[:, frame_no, : , :, :] = gray[:, frame_no:frame_no+4, :, :]

    return Feature.astype("uint8"), corrupt

def cut_body(vid, center, height, hand,shape=None, vid_shape_body = (128, 128)):
    c = center
    h = height
    if c[0]==0 or c[1]==0 or h==0:
        c = array([320, 240])
        h = 150
    y = int(round(c[1]*1.1))
    l = int(round(h*1.3 + (y-c[1])))
    x = int(round(c[0]-l/2))
    y = (y-l,y)
    x = (x,x+l)
    x,y = fit_screen(x,y)

    vid = vid[:,y[0]:y[1],x[0]:x[1]]
    if hand == "left" and shape==None: 
        for i,img in enumerate(vid): vid[i] = cv2.flip(img,1)

    if shape: body = empty(shape, dtype=vid.dtype)
    else: body = empty((vid.shape[0], vid_shape_body[0], vid_shape_body[1]), dtype=vid.dtype)   #Wudi change this line!!!
    for i,u in enumerate(vid):
        body[i] = cv2.resize(u,vid_shape_body,
            interpolation=cv2.INTER_LINEAR)

    return body

def cut_hand(vid, traj, hand,shape=None, vid_shape_hand = (128, 128),offset = 128/2):
    # new_vid = empty((2,vid.shape[0],offset*2,offset*2), "uint8")
    if shape:new_vid = empty(shape, "uint8")
    else: new_vid = empty((vid.shape[0], vid_shape_hand[0], vid_shape_hand[1]), "uint8")  #Wudi change this line!!!
    for i,img in enumerate(vid):
        img = cut_hand_img(img, traj[:,i])
        if hand == "left" and shape==None: 
            # print "left"
            # if random.randint(1,100)==1: print traj[:,i]
            img = cv2.flip(img,1)
        new_vid[i] = img

    return new_vid

def cut_hand_img(img, center, offset = 128/2):
    c = center.round().astype("int")

    x = (c[0]-offset, c[0]+offset)
    y = (c[1]-offset, c[1]+offset)
    x,y = fit_screen(x,y)

    # cut out hand    
    img = img[y[0]:y[1],x[0]:x[1]]
    return img

def fit_screen(x,y):
    l = x[1]-x[0]
    r = (480, 640) # 640 x 480 video resolution

    if not l == y[1]-y[0]:
        print l, x, y
        raise Exception, "l != y[1]-y[0]"

    if y[0] < 0: 
        y=(0,l)
    elif y[1] > r[0]: 
        y = (r[0]-l,r[0])

    if x[0] < 0: 
        x=(0,l)
    elif x[1] > r[1]: 
        x = (r[1]-l,r[1])

    return x,y

def normalize(x, old_min, old_max, new_min=0, new_max=255):
    """ Normalize numpy array """
    x = clip(x,old_min, old_max)
    return 1.*(x-old_min)*(new_max-new_min)/(old_max-old_min)+new_min

def medianblur(vid, ksize=3):
    for i,img in enumerate(vid): vid[i] = cv2.medianBlur(img, ksize)
    return vid

def play_vid_wudi(vid, Targets,  wait=1000/10, norm=True):
    cv2.destroyAllWindows()
    label_temp = Targets.argmax(axis=1)
    if isinstance(vid, list):
        for i in range(vid.shape[1]):
            for img0,img1 in zip(vid[0, i],vid[1, i]): 
                show_imgs([img0,img1], wait, norm)
    elif vid.ndim == 4:
        vid = vid.swapaxes(0,1)
        for imgs in vid: show_imgs(list(imgs), wait, norm)
    elif vid.ndim == 5:
        vid = vid.swapaxes(0,1)
        for t in range(vid.shape[0]):
            vid_temp = vid[t, :, :, :]
            vid_temp = vid_temp.swapaxes(0,1)
            for i, imgs in enumerate(vid_temp): show_imgs(list(imgs), str(label_temp[t]), wait, norm)
    else: 
        for img in vid: show_img(img, wait, norm)

def show_imgs(imgs, label, wait=0, norm=True):
    if norm:
        for i, img in enumerate(imgs):
            imgs[i] = norm_img(img)

    img = hstack(imgs)
    show_img(img, label,wait, False)

def show_img(img, label, wait=0, norm=True):
    ratio = 1.*img.shape[1]/img.shape[0]
    if norm: img = norm_img(img)
    img = img.astype("uint8")
    size=200 if img.shape[0]<200 else 400
    img = cv2.resize(img, (int(size*ratio), size))
    cv2.imshow(label, img)
    #cv2.namedWindow(
    cv2.waitKey(wait)

def norm_img(img):
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    return img



def make_sure_path_exists(path):
    """Try to create the directory, but if it already exist we ignore the error"""
    try: os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST: raise
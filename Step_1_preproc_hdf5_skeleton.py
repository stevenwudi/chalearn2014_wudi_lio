from numpy import *
import h5py
import gzip
from glob import glob
from scipy import misc
from numpy.random import permutation

import cv2

raw_input("Press Enter to continue...")



pc = "wudi"
pc = "wudi_linux"
if pc=="wudi":
    src = r"D:\Chalearn2014\Data_processed/"
elif pc=="wudi_linux":
    src = '/idiap/temp/dwu/chalearn2014_data/Data_processed/'
elif pc=="lio":
    src = "/mnt/wd/chalearn/preproc/"


def main():
    if pc=="wudi":
        files_train = glob(src+"train/batch_*.zip")
        files_valid = glob(src+"valid/batch_*.zip")
    elif pc=="wudi_linux":
        files_train = glob(src+"train/batch_*.zip")
        files_valid = glob(src+"valid/batch_*.zip")
    elif pc=="lio":
        files_train = glob(src+"train/batch_*.zip")
        files_valid = glob(src+"valid/batch_*.zip")


    print len(files_train), "train files"
    print len(files_valid), "valid files"

    

    f = h5py.File(src+"data%d.hdf5", "w", driver="family", memb_size=2**32-1)
    x_train = f.create_dataset("x_train", (400*1050,2,2,4,64,64), dtype='uint8', chunks=True)
    x_valid = f.create_dataset("x_valid", (29*1050,2,2,4,64,64), dtype='uint8', chunks=True)
    y_train = f.create_dataset("y_train", (400*1050,), dtype='uint8', chunks=True)
    y_valid = f.create_dataset("y_valid", (29*1050,), dtype='uint8', chunks=True)

    x_train_skeleton_feature = f.create_dataset("x_train_skeleton_feature", (400*1050,891), dtype='uint8', chunks=True)
    x_valid_skeleton_feature = f.create_dataset("x_valid_skeleton_feature", (29*1050,891), dtype='uint8', chunks=True)




    l = 0
    pos = 0
    for path in files_train:
        vid, lbl, skeleton_feature = load_data(path)
        print path
        print vid.shape[0], lbl.shape[0], skeleton_feature.shape[0]
        assert vid.shape[0] == lbl.shape[0] == skeleton_feature.shape[0]
        l += vid.shape[0]
        x_train.resize(l, axis=0)
        y_train.resize(l, axis=0)
        x_train_skeleton_feature.resize(l, axis=0)
        x_train[pos:pos+vid.shape[0]] = vid
        y_train[pos:pos+vid.shape[0]] = lbl
        x_train_skeleton_feature[pos:pos+vid.shape[0]] = skeleton_feature
        pos += vid.shape[0]
        print x_train.shape

    l = 0
    pos = 0
    for path in files_valid:
        vid, lbl, skeleton_feature = load_data(path)
        print path
        print vid.shape[0], lbl.shape[0]
        assert vid.shape[0] == lbl.shape[0]
        l += vid.shape[0]
        x_valid.resize(l, axis=0)
        y_valid.resize(l, axis=0)
        x_valid_skeleton_feature.resize(l, axis=0)
        x_valid[pos:pos+vid.shape[0]] = vid
        y_valid[pos:pos+vid.shape[0]] = lbl
        x_valid_skeleton_feature[pos:pos+vid.shape[0]] = skeleton_feature
        pos += vid.shape[0]
        print x_valid.shape

    f.close()
    print "done"

def load_gzip(path):
    file = gzip.GzipFile(path, 'rb')
    video, skeleton_feature, label, sk_trajectory = load(file)
    file.close()
    #wudi made an error in Step 1 extracting skelet, so
    # the following interwined, gesture frames, before frames, after frames
    pheight = empty(shape=(video.shape[2]))
    count = 0
    for i in range(len(sk_trajectory)):
        pheight[count: count+sk_trajectory[i][0].shape[1]-12] \
             = ones(shape=(sk_trajectory[i][0].shape[1]-12)) \
                *sk_trajectory[i][2]
        count = count+sk_trajectory[i][0].shape[1]-12
    #label start from 0 REMEMBER!!!
    video = video.swapaxes(0,2)
    video = video.swapaxes(1,2)

    return video,pheight,skeleton_feature,label


def cut_img(img,s):
    if s==0: return img 
    return img[s:-s,s:-s]


ratio = 0.25
res_shape=[100,2,2,5,64,64]
h = res_shape[-1]
def load_data(path): 
    """ load data into shared variables """
    v,p,skeleton_feature,l = load_gzip(path)
    v = v[:,:,:2]
    # print v.shape[0]
    res_shape[0] = v.shape[0]
    v_new = empty(res_shape,dtype="uint8")

    for i in xrange(v.shape[0]): #batch
        if p[i] < 10: p[i] = 100
        ofs = p[i]*ratio
        mid =  v.shape[-1]/2.
        sli = None
        if ofs < mid:
            start = int(round(mid-ofs))
            end = int(round(mid+ofs))
            sli = slice(start,end)

        for j in xrange(v.shape[2]): #maps
            for k in xrange(v.shape[3]): #frames
                #body
                img = v[i,0,j,k]
                img = cut_img(img,5)
                img = misc.imresize(img,(h,h))
                # if j==0: img = 255-misc.imfilter(img,"contour")
                v_new[i,0,j,k] = img

                #hand
                img = v[i,1,j,k]
                img = img[sli,sli]
                img = misc.imresize(img,(h,h))
                v_new[i,1,j,k] = img

    vid, lbl = v_new,l

    # shuffle data
    ind = permutation(l.shape[0])
    # ind = ind[:1000]
    vid = vid[:,:,:,:4,:,:]
    vid, skeleton_feature, lbl = vid[ind].astype("float32"), skeleton_feature[ind].astype("float32"),lbl[ind].astype("float32")

    # set value
    # x_.set_value(vid, borrow=True)
    # y_.set_value(lbl, borrow=True)

    return vid, lbl, skeleton_feature


if __name__ == '__main__':
    main()

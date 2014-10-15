from multiprocessing import Process, Queue, queues
from glob import glob
from time import time, sleep
import gzip,traceback,sys
from numpy import *
import numpy
from numpy.random import RandomState
from scipy import ndimage
from scipy import misc
from scipy.linalg import norm
from cPickle import loads, dumps, dump, load

#"blur",+ "contour", "detail", "edge_enhance", "edge_enhance_more", 
#"emboss", "find_edges", "smooth", "smooth_more", "sharpen"

q = Queue(1)
class Queue2(queues.Queue):       
    def puts(self,ob):
        global q
        # self.put(dumps(ob,protocol=-1))
        self.get()
        with open("augm.p",'wb') as file:
            dump(ob,file,protocol=-1)
        q.put(1)

    def gets(self):
        global q
        q.get()
        with open("augm.p",'rb') as file:
            r = load(file)
        self.put(1)
        return r


en = enumerate


# load_normal_q = Queue(2)
n_proc = 3
res_shape=[100,2,2,5,64,64]

h = res_shape[-1]
zoom_order = 1
ratio = 0.25
ri = random.randint(9999)
print ri
rng = RandomState(ri)
t_ofs = (32-res_shape[3])/2

def load_normal(path, cut=0):
    v,t,o,p,l = load_gzip(path)
    v = v[:,:,:res_shape[2]]
    if res_shape[2]==3:
        v[:,:,2] *= 255
    v_new = empty(res_shape,dtype="uint8")

    # t = concatenate([t,o],axis=1)


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
            for k in xrange(res_shape[3]): #frames
                #body
                img = v[i,0,j,k+t_ofs]
                img = cut_img(img,cut)
                img = misc.imresize(img,(h,h))
                # if j==0: img = 255-misc.imfilter(img,"contour")
                v_new[i,0,j,k] = img

                #hand
                img = v[i,1,j,k+t_ofs]
                img = img[sli,sli]
                img = misc.imresize(img,(h,h))
                v_new[i,1,j,k] = img

    return v_new,(t,o,p),l

def augment(path):
    v,t,o,p,l = load_gzip(path)

    v = v[:,:,:res_shape[2]]
    if res_shape[2]==3:
        v[:,:,2] *= 255

    v_new = empty(res_shape,dtype="uint8")

    # t = concatenate([t,o],axis=1)

    for i in xrange(v.shape[0]): #batch

        # body
        # cut = 0
        cut = rng.randint(7)
        # rot = 0 
        rot = randi(20)/10.
        trans_t = 0
        # trans_t = randi(2)
        trans_b = (0,trans_t,randi(3),randi(3))
        trans_h = (0,trans_t,randi(3),randi(3))
        # trans_b = (0,trans_t,0,0)
        # trans_h = (0,trans_t,0,0)
        # scale = ratio 
        scale = ratio+randi(10)/1000.

        if p[i] < 10: p[i] = 100
        ofs = p[i]*scale
        mid =  v.shape[-1]/2.
        sli = None
        if ofs < mid:
            start = int(round(mid-ofs))
            end = int(round(mid+ofs))
            sli = slice(start,end)
        t_ofs_ = rng.randint(32-res_shape[3]+1)
        # t_ofs_ = 32-res_shape[3]

        for j in xrange(v.shape[2]): #maps
            for k in xrange(res_shape[3]): #frames

                #body
                img = v[i,0,j,k+t_ofs_]
                img = cut_img(img,cut)
                img = misc.imresize(img,(h,h))
                # if j==0: img = misc.imfilter(img,"find_edges")
                    # img = lms(img)
                # img = cut_img(img,1)
                # img = misc.imrotate(img,rot)
                v_new[i,0,j,k] = img

                #hand
                img = v[i,1,j,k+t_ofs_]
                img = misc.imrotate(img,rot)
                img = img[sli,sli]
                img = misc.imresize(img,(h,h))
                # if j==0: img = lms(img)
                v_new[i,1,j,k] = img

        v_new[i,0] = ndimage.shift(v_new[i,0], trans_b, order=0, mode='nearest', prefilter=False)
        if rot !=0:
            v_new[i,0] = ndimage.rotate(v_new[i,0], rot, axes=(-2,-1), order=1, reshape=False, mode='nearest',)
        v_new[i,1] = ndimage.shift(v_new[i,1], trans_h, order=0, mode='nearest', prefilter=False)
        # t[i] = ndimage.shift(t[i],(0,trans_t), order=0, mode='nearest', prefilter=False)

    # v_new = add_noise(v_new)
    # t = add_noise_traj(t)
    return v_new,(t,o,p),l

paths_,results_ = None,None
def start_load(files,jobs=None,augm=True,start=False):
    # paths.cancel_join_thread()
    # results.cancel_join_thread()
    # print "starting loop"


    # # size = paths.qsize()
    # for _ in xrange(7500):
    #     try:
    #         paths.get_nowait()
    #     except: pass
    # # print "paths empty"
    # # size = results.qsize()
    # for _ in xrange(10):
    #     try:
    #         results.get_nowait()
    #     except: pass
    #     sleep(0.1)
    # # print "results empty"

    # if jobs:
    #     for job in jobs: job.terminate()
    # print "jobs terminated"
    global paths_, results_

    if start:
        paths_ = Queue()
        results_ = Queue2(2)
        jobs = [start_proc(worker,(paths_,results_)) for _ in range(n_proc)]

    while not paths_.empty(): 
        paths_.get()
        sleep(0.1)
    while not q.empty(): 
        q.get()
        sleep(0.1)
    while not results_.empty(): 
        try:
            results_.get_nowait()
        except: pass
        sleep(0.2)

    results_.put(1)

    for file in files: paths_.put((file,augm))

    # print "paths put"


    return jobs, results_

from copy import deepcopy
def worker(paths,results):
    try:
        while True:
            path, augm = paths.get()
            if augm:
                r = augment(path)
            else:
                r = load_normal(path)
            results.puts((deepcopy(r),deepcopy(augm)))
            del r
    except:
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def loop(files,paths):
    print 'start loop'
    while True:
        for file in files: paths.put(file)

def start_proc(target, args=(), daemon=True):
    p_ = Process(target=target, args=args)
    p_.daemon=daemon
    p_.start()
    return p_

def add_noise_traj(t):
    wnoisesize=5
    noisesize=20

    noise = random.randint(0,255,t.shape).astype("float32")
    for i in xrange(t.shape[0]):
        # noise[i,j,k,l] = ndimage.uniform_filter(noise[i,j,k,l],size=fsize)
        noise[i] = ndimage.gaussian_filter(noise[i],3, order=0, mode='reflect')
    wnoise = random.randint(-wnoisesize,wnoisesize,t.shape)
    noise = normalize_full(noise,-noisesize,noisesize)
    t = clip(t+noise+wnoise,0,255).astype(uint8)
    return t

def add_noise(v):
    # fsize=11
    wnoisesize=1
    noisesize=60

    noise = random.randint(0,255,v.shape).astype("float32")
    for i in xrange(v.shape[0]):
        for j in xrange(v.shape[1]):
            for k in xrange(v.shape[2]):
                for l in xrange(v.shape[3]):
                    # noise[i,j,k,l] = ndimage.uniform_filter(noise[i,j,k,l],size=fsize)
                    noise[i,j,k,l] = ndimage.gaussian_filter(noise[i,j,k,l],2, order=0, mode='reflect')
    wnoise = random.randint(-wnoisesize,wnoisesize,v.shape)
    noise = normalize_full(noise,-noisesize,noisesize)
    v = clip(v+noise+wnoise,0,255).astype(uint8)
    return v

def lms(img):
    img = img.astype("float32")
    # img /= 255.
    img -= ndimage.gaussian_filter(img, 2, order=0, mode='reflect')
    img = normalize(img) 
    return img.astype("uint8")

def normalize(x):
    _min = x.min()
    return 1.*(x-_min)*255./(x.max()-_min)

def normalize_full(x, new_min=0, new_max=255):
    old_min = x.min()
    old_max = x.max()
    return 1.*(x-old_min)*(new_max-new_min)/(old_max-old_min)+new_min

def cut_img(img,s):
    if s==0: return img 
    return img[s:-s,s:-s]

def randi(i): return rng.randint(-i,i)

def load_gzip(path):
    file = gzip.GzipFile(path, 'rb')
    video, skeleton_feature, label, sk_trajectory = load(file)
    file.close()
    #wudi made an error in Step 1 extracting skelet, so
    # the following interwined, gesture frames, before frames, after frames
    pheight = numpy.empty(shape=(video.shape[2]))
    count = 0
    for i in range(len(sk_trajectory)):
        pheight[count: count+sk_trajectory[i][0].shape[1]-12] \
             = numpy.ones(shape=(sk_trajectory[i][0].shape[1]-12)) \
                *sk_trajectory[i][2]
        count = count+sk_trajectory[i][0].shape[1]-12
    #label start from 0 REMEMBER!!!
    video = video.swapaxes(0,2)
    video = video.swapaxes(1,2)

    return video,pheight,skeleton_feature,label

def play_vid(vid, wait=50, norm=False):
    import cv2
    for i,img in en(vid):
        if norm: cv2.normalize(img,img, 0, 255, cv2.NORM_MINMAX)
        img = cv2.resize(img.astype("uint8"), (200,200), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Gesture", img)
        cv2.waitKey(wait)

def elastic_distortion(image, sigma=6, alpha=36):
    """https://github.com/nanaya-tachibana/handwritten"""
    def delta():
        d = ndimage.gaussian_filter(random.uniform(-1, 1, size=image.shape), sigma)
        return (d / norm(d)) * alpha
    assert image.ndim == 2
    return bilinear_interpolate(image, delta(), delta())

def bilinear_interpolate(values, dx, dy):
    """Interpolating with given dx and dy"""
    assert values.shape == dx.shape == dy.shape
    A = zeros(values.shape)
    for i in xrange(values.shape[0]):
        for j in xrange(values.shape[1]):
            x,y = i + dx[i, j],j + dy[i, j]
            if x < 0: x += int(1 + 0 - x)
            if x >= values.shape[0] - 1: x -= int(1 + x - (values.shape[0] - 1))
            if y < 0: y += int(1 + 0 - y)
            if y >= values.shape[1]-1: y -= int(1 + y - (values.shape[1] - 1))
            x1,y1 = int(x),int(y)
            x2,y2 = x1 + 1,y1 + 1
            f11,f12,f21,f22 = values[x1, y1],values[x1, y2],values[x2, y1],values[x2, y2]
            A[i,j] = (f11*(x2-x)*(y2-y) + f12*(x2-x)*(y-y1)+ f21*(x-x1)*(y2-y) + f22*(x-x1)*(y-y1))
    return A

if __name__ == '__main__':
    # src = '/media/Data/mp/chalearn2014/20lbl_32x128x128/train'
    # src = "E:/mp/chalearn2014/20lbl_32x128x128/train"
    src = '/media/Data/mp/chalearn2014/20lbl_32x128x128/train'
    files = glob(src+'/batch_100_*.zip')
    print len(files),"files"
    random.shuffle(files)
    test_aug = True

    if not test_aug:
        #--------------TEST MULTIPROCESSING-----------------------------
        jobs, queue = start_load(files)
        # q_paths(files,False)

        for _ in xrange(100):
            print "waiting for result"
            start_time = time()

            (v,s,l),augm = queue.gets()
            jobs, _ = start_load(files,jobs,False)

            print "result get:", time()-start_time
            for i in xrange(v.shape[0]):
                for j in xrange(v.shape[1]):
                    for k in xrange(v.shape[2]):
                        play_vid(v[i,j,k],wait=1)

        for job in jobs: job.join()
        #--------------TEST MULTIPROCESSING-----------------------------

    else:
        #--------------TEST AUGMENTATION-----------------------------
        for path in files:
            start_time = time()
            v,s,l = augment(path)
            # v,s,l = load_normal(path)

            print s[0].shape

            print path[-7:],"in",time()-start_time,"s",v.shape, v.dtype
            for i in xrange(v.shape[0]):
                for j in xrange(v.shape[1]):
                    for k in xrange(v.shape[2]):
                        play_vid(v[i,j,k],wait=0, norm=False)


        #--------------TEST AUGMENTATION-----------------------------


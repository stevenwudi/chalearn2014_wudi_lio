"""
Di Wu transfer some functions from traindebug.py to this
modules to make the file more modular

"""

from numpy import array, empty, float32
from time import time
import string
import shutil
from gzip import GzipFile
from cPickle import dump, load
from data_aug import load_gzip, res_shape, ratio, cut_img, misc, h

from convnet3d import relu, leaky_relu
# theano imports
import theano.tensor as T
from theano.tensor import TensorType
from theano import config, shared
floatX = config.floatX


#  helper functions
# ------------------------------------------------------------------------------
def normalize(input,newmin=-1,newmax=1):
    mini = T.min(input)
    maxi = T.max(input)
    return (input-mini)*(newmax-newmin)/(maxi-mini)+newmin

def _shared(val, borrow=True):
    return shared(array(val, dtype=floatX), borrow=borrow)

def _avg(_list): return list(array(_list).mean(axis=0))

def write(_s, res_dir): 
    with open(res_dir+"/output.txt","a") as f: f.write(_s+"\n")
    print _s

def ndtensor(n): return TensorType(floatX, (False,)*n) # n-dimensional tensor




def load_data(path, rng, epoch, batch_size, x_,y_): 
    """ load data into shared variables """
    #global x_,t_,y_,
    #global first_report2 
    #first_report2 = True
    start_time = time()
    v,p,skeleton_feature,l = load_gzip(path)
    v = v[:,:,:res_shape[2]]
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

    #if epoch==0: print "get in",str(time()-start_time)[:3]+"s",
    # shuffle data
    ind = rng.permutation(l.shape[0])
    ind = ind[:batch_size]
    vid = vid[:,:,:,:4,:,:]
    vid, skeleton_feature, lbl = vid[ind].astype(floatX), skeleton_feature[ind].astype(floatX),lbl[ind].astype(floatX)
    #vid, skeleton_feature, lbl = vid.astype(floatX), skeleton_feature.astype(floatX),lbl.astype(floatX)

    # vid = vid/(255./(scaler*2.))-scaler
    #traj = traj/(255./(scaler_traj*2.))-scaler_traj
    # traj = traj/(255./5.)

    # Wudi already made labels start from 0
    #lbl -= 1 

    #if first_report2:
    #    print "data range:",vid.min(),vid.max()
    #    print "traj range:",skeleton_feature.min(),skeleton_feature.max()
    #    print "lbl range:",lbl.min(),lbl.max()
    #    first_report2 = False

    # set value
    x_.set_value(vid, borrow=True)
    #t_.set_value(skeleton_feature, borrow=True)
    y_.set_value(lbl, borrow=True)


# def load_params():
#     global load_params_pos
#     par = load(open("params.p", "rb"))
#     W = par[load_params_pos]
#     b = par[load_params_pos+1]
#     load_params_pos +=2
#     return W,b



def conv_args(stage, i, batch, net, use, rng, video_shapes, load_path=""):
    """ ConvLayer arguments, i: stage index """
    args = {
        'batch_size':batch.micro, 
        'activation':net.activation, 
        'rng':rng,
        'n_in_maps':net.maps[stage],
        'n_out_maps':net.maps[stage+1], 
        'kernel_shape':net.kernels[stage], 
        'video_shape':video_shapes[stage],
        "fast_conv":use.fast_conv,
        "layer_name":"Conv"+str(stage)+str(i),
        "W_scale":net.W_scale[stage][i],
        "b_scale":net.b_scale[stage][i],
        "stride":net.stride[stage]
    }
    if stage in net.shared_stages and i in net.shared_convnets:
        print "sharing weights!"
        args["W"], args["b"] = layers[-1].params # shared weights
    elif use.load:
        args["W"], args["b"] = load_params(use,load_path) # load stored parameters
    return args


def var_norm(_x,imgs=True,axis=[-3,-2,-1]):
    if imgs:
        return (_x-T.mean(_x,axis=axis,keepdims=True))/T.maximum(1e-4,T.sqrt(T.var(_x,axis=axis,keepdims=True) + 1e-9) )
    return (_x-T.mean(_x))/T.maximum(1e-4,T.sqrt(T.var(_x,axis=axis,keepdims=True) + 1e-9) )

def std_norm(_x,axis=[-3,-2,-1]):
    return _x/T.maximum(1e-4,T.sqrt(T.var(_x,axis=axis,keepdims=True) + 1e-9))                  

def pool_time(X,shape):
    shape_o = shape
    shape = (prod(shape[:-2]),)+shape[-2:]
    X_ = X.reshape(shape)
    print shape
    frames = []
    for i in range(shape[0])[::2]:
        fr1 = X_[i]
        fr2 = X_[i+1]
        m1 = fr1.mean()
        m2 = fr2.mean()
        # fr = ifelse(T.lt(m1,m2),fr2,fr1)
        fr = ifelse(T.lt(m1,m2),i+1,i)
        frames.append(fr)
    ind = T.stack(frames)
    # ind = ind.reshape((shape[0],shape[1]/2))
    new_X = X_[ind]
    # new_X = T.concatenate(frames,axis=0)
    shape = shape_o[:-3]+(shape_o[-3]/2,)+shape_o[-2:]
    new_X = new_X.reshape(shape)

    return new_X

def lin(X): return X


def maxout(X,X_shape):
    shape = X_shape[:-1]+(X_shape[-1]/2,)+(2,)
    out = X.reshape(shape)
    return T.max(out, axis=-1)    


def print_params(params): 
    for param in params[::2]:
        p = param.get_value(borrow=True)
        print param.name+" %.4f %.4f %.4f %.4f %i"%(p.min(),p.mean(),p.max(),p.std(),len(p[p==0]))



def _mini_batch(model, mini_batch, batch, is_train, apply_updates =None ):
    ce = []
    for i in xrange(batch.mini/batch.micro):
        c_,e_, out_mean, out_std = model(mini_batch, i) 
        ce.append([c_,e_])
    if is_train: 
        apply_updates()
    #print ce, out_mean
    #print type(ce), type(out_mean), type(out_std)
    #return _avg(ce), _avg(out_mean), _avg(out_std)
    return _avg(ce), out_mean, out_std

def _batch(model, batch_size, batch, is_train=True, apply_updates=None):
    ce_all = []
    out_mean_all = []
    out_std_all = []
    for i in xrange(batch_size/batch.mini): 
        ce, out_mean, out_std = (_mini_batch(model, i, batch, is_train, apply_updates))
        ce_all.append(ce)
        out_mean_all.append(out_mean)
        out_std_all.append(out_std)
    return _avg(ce_all), _avg(out_mean_all), _avg(out_std_all)


def training_report(train_ce):
    return "%5.3f %5.2f" % (train_ce[0], train_ce[1]*100.)

def epoch_report(epoch, best_valid, time_used, lr, train_ce, valid_ce, res_dir):
    result_string = """ 
    epoch %i: %.2f m since start, LR %.2e
    train_cost: %.3f, train_err: %.3f
    val_cost: %.3f, val_err: %.3f, best: %.3f""" % \
    (epoch, time_used / 60., lr, 
        train_ce[0], train_ce[1]*100., valid_ce[0], valid_ce[1]*100.,best_valid*100.)

    write(result_string, res_dir)

def test_lio_skel(use, test_model, batch, drop, rng, epoch, batch_size, x_, y_, loader, x_skeleton_):
    global jobs
    if use.drop: # dont use dropout when testing
        #drop.p_traj.set_value(float32(0.)) 
        drop.p_vid.set_value(float32(0.)) 
        drop.p_hidden.set_value(float32(0.)) 
    ce = []
    first_test_file = True
    for i in range(loader.n_iter_valid):
        if first_test_file:
            augm = False
            first_test_file = False
        else: augm = True
        # load_data(file, rng, epoch, batch_size, x_, y_)
        loader.next_valid_batch(x_, y_, x_skeleton_)
        #load_data(file,  rng, epoch)
        ce_temp, out_mean_temp, out_std_temp = _batch(test_model, batch_size, batch, is_train=False)
        ce.append(ce_temp)
    if use.drop: # reset dropout
        #drop.p_traj.set_value(drop.p_traj_val) 
        drop.p_vid.set_value(drop.p_vid_val) 
        drop.p_hidden.set_value(drop.p_hidden_val)
    # start_load(files.train,augm=use.aug)
    return _avg(ce)


def save_results(train_ce, valid_ce, res_dir, params=None, out_mean_train=None, out_std_train=None):
    if len(valid_ce)==0: rate = 0
    else: rate = valid_ce[-1][1]
    dst = res_dir.split("/")
    if dst[-1].find("%")>=0:
        d = dst[-1].split("%")
        d[0] = str(rate*100)[:4]
        dst[-1] = string.join(d,"%")
    else:
        dst[-1] = str(rate*100)[:4]+"% "+dst[-1]
    dst = string.join(dst,"/") 
    shutil.move(res_dir, dst)
    res_dir = dst
    file = GzipFile(res_dir+"/params.zip", 'wb')
    dump(params, file, -1)
    file.close()
    ce = (train_ce, valid_ce)
    with open(res_dir+"/cost_error.txt","wb") as f: f.write(str(ce)+"\n")
    dump(ce, open(res_dir+"/cost_error.p", "wb"), -1)

    if out_std_train is not None:
    	inspection = (out_mean_train, out_std_train)
    	with open(res_dir+"/inspection.txt","wb") as f: f.write(str(inspection)+"\n")
    	dump(inspection, open(res_dir+"/inspection.p", "wb"), -1)
    return res_dir

def move_results(res_dir):
    global moved
    dst = res_dir.split("/")
    #dst = dst[:-2]  + [dst[-1]] 
    dst = string.join(dst,"/") 
    shutil.move(res_dir, dst)
    res_dir = dst
    moved = True
    shutil.copy(__file__, res_dir)
    # file_aug = string.join(__file__.split("/")[:-1],"/")+"/data_aug.py"
    try:
        file_aug = "data_aug.py"
        shutil.copy(file_aug, res_dir)
    except: pass
    return res_dir

def save_params(params, res_dir, s=""):
    # global res_dir
    print "Saving params"
    l = []
    for p in params:
        l.append(p.get_value(borrow=True))
    if s=="": file = GzipFile(res_dir+"/params.zip", 'wb')
    else: file = GzipFile(res_dir+"/params"+s+".zip", 'wb')
    dump(l, file, -1)
    file.close()

def load_params(use, load_path=""):
    import os
    if os.path.exists(load_path):
        path = load_path
    else:
        path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))

    if os.path.isfile(path+'paramsbest.zip'):
        file = GzipFile(path+"paramsbest.zip", "rb")
    else:
        file = GzipFile("params.zip", "rb")
    par = load(file)
    file.close()
    print 'load parameters'
    W = par[use.load_params_pos]
    
    if not use.fast_conv:
        # we need to flip here because the best parameter by
        # Lio was using cudaconv, different from Theano's conv op
        # see: http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html
        if len(W.shape) >4:
            W = W[:, :, :, ::-1, ::-1]
    b = par[use.load_params_pos+1]
    use.load_params_pos +=2
    
    return W,b

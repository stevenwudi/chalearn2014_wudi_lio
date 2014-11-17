"""
Video classifier using a 3D deep convolutional neural network

Data: ChaLearn 2014 gesture challenge: gesture recognition

original code by: Lionel Pigou

Code modulated by: Di Wu

"""
# various imports
from cPickle import dump, load
from glob import glob
from time import time, localtime
from gzip import GzipFile
import os
import shutil
import string
from scipy import ndimage

# numpy imports
from numpy import ones, array, prod, zeros, empty, inf, float32, random

# theano imports
from theano import function, config, shared
from theano.ifelse import ifelse
from theano.tensor.nnet import conv2d
from theano.tensor import TensorType
from theano.sandbox.cuda import CudaNdarrayType, CudaNdarray  #----wudi comment: why comment this line?
import theano.tensor as T

# customized imports
#data_aug
from functions.data_aug import start_load, load_normal, load_gzip, res_shape, ratio, cut_img, misc, h
from convnet3d import ConvLayer, NormLayer, PoolLayer, LogRegr, HiddenLayer, \
                    DropoutLayer, relu, tanh

# wudi's modular imports
# the hyperparameter set the data dir, use etc classes, it's important to modify it according to your need
from classes.hyperparameters import use, lr, batch, reg, mom, tr, drop, net , files,  DataLoader
from functions.train_functions import normalize, _shared, _avg, write, ndtensor, \
                                  conv_args, var_norm, std_norm, lin,\
                                  print_params, load_data, _mini_batch, _batch,\
                                  timing_report, training_report, epoch_report, \
                                  test, test_lio, save_results, move_results, save_params

####################################################################
####################################################################
print "\n%s\n\t initializing \n%s"%(('-'*30,)*2)
####################################################################
####################################################################
# source and result directory
pc = "wudi"
if pc=="wudi":
    src = r"I:\Kaggle_multimodal\Training_prepro"
    res_dir_ = r"I:\Kaggle_multimodal\result"# dir of original data -- note that wudi has decompressed it!!!
elif pc=="lio":
    src = "/mnt/wd/chalearn/preproc"
    res_dir_ = "/home/lpigou/chalearn_wudi/try"



lt = localtime()
res_dir = res_dir_+"/try/"+str(lt.tm_year)+"."+str(lt.tm_mon).zfill(2)+"." \
            +str(lt.tm_mday).zfill(2)+"."+str(lt.tm_hour).zfill(2)+"."\
            +str(lt.tm_min).zfill(2)+"."+str(lt.tm_sec).zfill(2)
os.makedirs(res_dir)
#  global variables/constants
# ------------------------------------------------------------------------------
if False:
    import theano 
    theano.config.compute_test_value = 'warn' #debug mode
params = [] # all neural network parameters
layers = [] # all architecture layers
mini_updates = []
micro_updates = []
last_upd = []
update = []

# shared variables
learning_rate = shared(float32(lr.init))
if use.mom: 
    momentum = shared(float32(mom.momentum))
    drop.p_vid = shared(float32(drop.p_vid_val) )
    drop.p_hidden = shared(float32(drop.p_hidden_val))

# symbolic variables
# in shape: #frames * gray/depth * body/hand * 4 maps
x = ndtensor(len(tr.in_shape))(name = 'x') # video input
# x = T.TensorVariable(CudaNdarrayType([False] * len(in_shape))) # video input
y = T.ivector(name = 'y') # labels
idx_mini = T.lscalar(name="idx_mini") # minibatch index
idx_micro = T.lscalar(name="idx_micro") # microbatch index
x_ = _shared(empty(tr.in_shape))
y_ = _shared(empty((tr.batch_size,)))
y_int32 = T.cast(y_,'int32')

# print parameters
# ------------------------------------------------------------------------------
for c in (use, lr, batch, net, reg, drop, mom, tr):
    write(c.__name__+":", res_dir)
    _s = c.__dict__
    del _s['__module__'], _s['__doc__']
    for key in _s.keys(): 
        val = str(_s[key])
        if val.startswith("<static"): val = str(_s[key].__func__.__name__)
        if val.startswith("<Cuda"): continue
        if val.startswith("<Tensor"): continue
        write("  "+key+": "+val, res_dir)

####################################################################
####################################################################
print "\n%s\n\t preparing data \n%s"%(('-'*30,)*2)
####################################################################
####################################################################

# print data sizes
file_info = files(src)
if use.valid2: file_info.n_test = len(file_info.valid2)
else: file_info.n_test = 0
write('data: total: %i train: %i valid: %i test: %i' % \
    ((file_info.n_test+file_info.n_train+file_info.n_valid), 
        file_info.n_train, 
        file_info.n_valid, 
        file_info.n_test), res_dir)

first_report2 = True
epoch = 0


loader = DataLoader(src, tr.batch_size) # Lio changed it to read from HDF5 files

####################################################################
####################################################################
print "\n%s\n\tbuilding\n%s"%(('-'*30,)*2)
####################################################################
#################################################################### 
# ConvNet
# ------------------------------------------------------------------------------
# calculate resulting video shapes for all stages
conv_shapes = []
for i in xrange(net.n_stages):
    k,p,v = array(net.kernels[i]), array(net.pools[i]), array(tr.video_shapes[i])
    conv_s = tuple(v-k+1)
    conv_shapes.append(conv_s)
    tr.video_shapes.append(tuple((v-k+1)/p))
    print "stage", i
    print "  conv",tr.video_shapes[i],"->",conv_s
    print "  pool",conv_s,"->",tr.video_shapes[i+1],"x",net.maps[i+1]

# number of inputs for MLP = (# maps last stage)*(# convnets)*(resulting video shape) + trajectory size
n_in_MLP = net.maps[-1]*net.n_convnets*prod(tr.video_shapes[-1]) 
print 'MLP:', n_in_MLP, "->", net.hidden, "->", net.n_class, ""

if use.depth:
    if net.n_convnets==2: 
        out = [x[:,:,0,:,:,:], x[:,:,1,:,:,:]] # 2 nets: body and hand

# build 3D ConvNet
insp = []
for stage in xrange(net.n_stages):
    for i in xrange(len(out)): # for body and hand
        # normalization
        if use.norm and stage==0: 
            gray_norm = NormLayer(out[i][:,0:1], method="lcn",
                use_divisor=use.norm_div).output
            gray_norm = std_norm(gray_norm,axis=[-3,-2,-1])
            depth_norm = var_norm(out[i][:,1:])
            out[i]  = T.concatenate([gray_norm,depth_norm],axis=1)
        elif use.norm:
            out[i] = NormLayer(out[i], method="lcn",use_divisor=use.norm_div).output
            out[i] = std_norm(out[i],axis=[-3,-2,-1])
        # convolutions  
        out[i] *= net.scaler[stage][i]
        layers.append(ConvLayer(out[i], **conv_args(stage, i, batch, net, use, tr.rng, tr.video_shapes)))
        out[i] = layers[-1].output
        out[i] = PoolLayer(out[i], net.pools[stage], method=net.pool_method).output
        if tr.inspect: insp.append(T.mean(out[i]))

# flatten all convnets outputs
for i in xrange(len(out)): out[i] = std_norm(out[i],axis=[-3,-2,-1])
out = [out[i].flatten(2) for i in range(len(out))]
vid_ = T.concatenate(out, axis=1)

# dropout
if use.drop: 
    vid_ = DropoutLayer(vid_, rng=tr.rng, p=drop.p_vid).output

#maxout
if use.maxout:
    vid_ = maxout(vid_, (batch.micro,n_in_MLP))
    net.activation = lin
    n_in_MLP /= 2
    # net.hidden *= 2

# MLP
# ------------------------------------------------------------------------------
# fusion
if net.fusion == "early":
    out = vid_
    # hidden layer
    layers.append(HiddenLayer(out, n_in=n_in_MLP, n_out=net.hidden, rng=tr.rng, 
        W_scale=net.W_scale[-2], b_scale=net.b_scale[-2], activation=relu))
    out = layers[-1].output


if tr.inspect: insp = T.stack(insp[0],insp[1],insp[2],insp[3],insp[4],insp[5], T.mean(out))
else: insp =  T.stack(0,0)
# out = normalize(out)
if use.drop: out = DropoutLayer(out, rng=tr.rng, p=drop.p_hidden).output
#maxout
if use.maxout:
    out = maxout(out, (batch.micro,net.hidden))
    net.hidden /= 2

# softmax layer
layers.append(LogRegr(out, rng=tr.rng, activation=lin, n_in=net.hidden, 
    W_scale=net.W_scale[-1], b_scale=net.b_scale[-1], n_out=net.n_class))

"""
layers[-1] : softmax layer
layers[-2] : hidden layer (video if late fusion)
layers[-3] : hidden layer (trajectory, only if late fusion)
"""
# cost function
cost = layers[-1].negative_log_likelihood(y)

if reg.L1_vid > 0 or reg.L2_vid > 0:
    # L1 and L2 regularization
    L1 = T.abs_(layers[-2].W).sum() + T.abs_(layers[-1].W).sum()
    L2 = (layers[-2].W ** 2).sum() + (layers[-1].W ** 2).sum()

    cost += reg.L1_vid*L1 + reg.L2_vid*L2 

# function computing the number of errors
errors = layers[-1].errors(y)

# gradient descent
# ------------------------------------------------------------------------------
# parameter list
for layer in layers: params.extend(layer.params)
# gradient list
gparams = T.grad(cost, params)
assert len(gparams)==len(params)

def get_update(i): return update[i]/(batch.mini/batch.micro)

for i, (param, gparam) in enumerate(zip(params, gparams)):
    # shape of the parameters
    shape = param.get_value(borrow=True).shape
    # init updates := zeros
    update.append(_shared(zeros(shape, dtype=config.floatX)))
    # micro_updates: sum of lr*grad
    micro_updates.append((update[i], update[i] + learning_rate*gparam))
    # re-init updates to zeros
    mini_updates.append((update[i], zeros(shape, dtype=config.floatX)))

    if use.mom:
        last_upd.append(_shared(zeros(shape, dtype=config.floatX)))
        v = momentum * last_upd[i] - get_update(i)
        mini_updates.append((last_upd[i], v))
        if mom.nag: # nesterov momentum
            mini_updates.append((param, param + momentum*v - get_update(i)))
        else:
            mini_updates.append((param, param + v))
    else:    
        mini_updates.append((param, param - get_update(i)))

####################################################################
####################################################################
print "\n%s\n\tcompiling\n%s"%(('-'*30,)*2)
####################################################################
#################################################################### 
# compile functions
# ------------------------------------------------------------------------------
if True:
    def get_batch(_data): 
        pos_mini = idx_mini*batch.mini
        idx1 = pos_mini + idx_micro*batch.micro
        idx2 = pos_mini + (idx_micro+1)*batch.micro
        return _data[idx1:idx2]

    def givens(dataset_):
        return {x: get_batch(dataset_[0]),
                y: get_batch(dataset_[1])}

    print 'compiling apply_updates'
    apply_updates = function([], 
        updates=mini_updates, 
        on_unused_input='ignore')

    print 'compiling train_model'
    train_model = function([idx_mini, idx_micro], [cost, errors, insp], 
        updates=micro_updates, 
        givens=givens((x_,y_int32)), 
        on_unused_input='ignore')

    print 'compiling test_model'
    test_model = function([idx_mini, idx_micro], [cost, errors], 
        givens=givens((x_,y_int32)),
        on_unused_input='ignore')

####################################################################
####################################################################
print "\n%s\n\ttraining\n%s"%(('-'*30,)*2)
####################################################################
#################################################################### 
time_start = 0
best_valid = inf
# main loop
# ------------------------------------------------------------------------------
lr_decay_epoch = 0
n_lr_decays = 0
train_ce, valid_ce, valid2_ce = [], [], []
flag=True
global insp_
insp_ = None

res_dir = save_results(train_ce, valid_ce, res_dir, params=params)
if not tr.moved: res_dir = move_results(res_dir)
tr.moved = True
save_params(params, res_dir)

for epoch in xrange(tr.n_epochs):
    ce = []
    print_params(params) 
    ####################################################################
    ####################################################################
    print "\n%s\n\t epoch %d \n%s"%('-'*30, epoch, '-'*30)
    ####################################################################
    ####################################################################
    for i in range(loader.n_iter_train):
        time_start = time()
        #load
        # load_data(train_file, tr.rng, epoch, tr.batch_size, x_, y_)
        loader.next_train_batch(x_, y_)
        # print "loading time", time()-time_start
        # train
        tr.batch_size = y_.get_value(borrow=True).shape[0]
        ce.append(_batch(train_model, tr.batch_size, batch, True, apply_updates))
       
        if epoch==0: timing_report(i, time()-time_start, tr.batch_size, res_dir)
        print "\t| "+ training_report(ce[-1])
    # End of Epoch
    #-------------------------------
    ####################################################################
    ####################################################################
    print "\n%s\n\t End of epoch %d, \n printing some debug info.\n%s" \
        %('-'*30, epoch, '-'*30)
    ####################################################################
    ####################################################################
    # print insp_
    train_ce.append(_avg(ce))
    # validate
    valid_ce.append(test_lio(file_info.valid, use, test_model, batch, drop, tr.rng, epoch, tr.batch_size, x_, y_,loader))

    # save best params
    # if valid_ce[-1][1] < 0.25:
    res_dir = save_results(train_ce, valid_ce, res_dir, params=params)
    if not tr.moved: res_dir = move_results(res_dir)

    if valid_ce[-1][1] < best_valid:
        save_params(params, res_dir, "best")
    save_params(params, res_dir)

    if valid_ce[-1][1] < best_valid:
        best_valid = valid_ce[-1][1]

    # epoch report
    epoch_report(epoch, best_valid, time()-time_start, learning_rate.get_value(borrow=True),\
        train_ce[-1], valid_ce[-1], res_dir)
    # make_plot(train_ce, valid_ce)

    if lr.decay_each_epoch:
        learning_rate.set_value(float32(learning_rate.get_value(borrow=True)*lr.decay))
    # elif lr.decay_if_plateau:
    #     if epoch - lr_decay_epoch > tr.patience \
    #         and valid_ce[-1-tr.patience][1] <= valid_ce[-1][1]:

    #         write("Learning rate decay: validation error stopped improving")
    #         lr_decay_epoch = epoch
    #         n_lr_decays +=1
    #         learning_rate.set_value(float32(learning_rate.get_value(borrow=True)*lr.decay_big))
    # if epoch == 0: 
        # learning_rate.set_value(float32(3e-4))
    # else:
        # learning_rate.set_value(float32(learning_rate.get_value(borrow=True)*lr.decay))
    tr.rng.shuffle(file_info.train)

#if use.aug: 
#    for job in jobs: job.join()
"""
import matplotlib.pyplot as plt
def make_plot(train_ce, valid_ce): 
    tr = array(train_ce)[:,1]*100.
    va = array(valid_ce)[:,1]*100.
    x = range(1,tr.shape[0]+1)

    plt.plot(x, tr, 'rs--', label='train')
    plt.plot(x, va, 'bo-', label='valid')
    plt.ylabel('Error (%)')
    plt.xlabel('Epoch')
    plt.xlim([0,tr.shape[0]+1])
    plt.ylim([0,95])
    plt.legend()
    plt.savefig(res_dir+'/plot.pdf', bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.cla()
"""
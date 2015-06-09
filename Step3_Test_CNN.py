from numpy import ones, array, prod, zeros, empty, inf, float32, random
import theano.tensor as T
from theano import function, config, shared

from classes.hyperparameters import use, lr, batch, reg, mom, tr, drop, net , files,  DataLoader
from convnet3d import ConvLayer, NormLayer, PoolLayer, LogRegr, HiddenLayer, \
                    DropoutLayer, relu, tanh
from functions.train_functions import normalize, _shared, _avg, write, ndtensor, \
                                  conv_args, var_norm, std_norm, lin,\
                                  print_params, load_data, _mini_batch, _batch,\
                                  timing_report, training_report, epoch_report, \
                                  test, test_lio, save_results, move_results, save_params, load_params


use.load = True  # we load the CNN parameteres here
####################################################################
####################################################################
print "\n%s\n\t preparing data \n%s"%(('-'*30,)*2)
####################################################################
####################################################################
# source and result directory
pc = "wudi"
if pc=="wudi":
    src = r"D:\Chalearn2014\Data_processed"
    res_dir_ = r"D:\Chalearn2014\result"# dir of original data -- note that wudi has decompressed it!!!
elif pc=="lio":
    src = "/mnt/wd/chalearn/preproc"
    res_dir_ = "/home/lpigou/chalearn_wudi/try"


loader = DataLoader(src, tr.batch_size) # Lio changed it to read from HDF5 files

####################################################################
####################################################################
print "\n%s\n\tbuilding\n%s"%(('-'*30,)*2)
####################################################################
#################################################################### 

idx_mini = T.lscalar(name="idx_mini") # minibatch index
idx_micro = T.lscalar(name="idx_micro") # microbatch index
x = ndtensor(len(tr.in_shape))(name = 'x') # video input
x_ = _shared(empty(tr.in_shape))
y_ = _shared(empty((tr.batch_size,)))
y_int32 = T.cast(y_,'int32')
y = T.ivector(name = 'y') # labels

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
layers = [] # all architecture layers
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
    drop.p_vid = shared(float32(drop.p_vid_val) )
    drop.p_hidden = shared(float32(drop.p_hidden_val))
    drop.p_vid.set_value(float32(0.))  # dont use dropout when testing
    drop.p_hidden.set_value(float32(0.))  # dont use dropout when testing
    vid_ = DropoutLayer(vid_, rng=tr.rng, p=drop.p_vid).output

# MLP
# ------------------------------------------------------------------------------
# fusion
if net.fusion == "early":
    out = vid_
    # hidden layer
    Wh, bh = load_params(use)  # This is test, wudi added this!
    layers.append(HiddenLayer(out, W = Wh, b =bh, n_in=n_in_MLP, n_out=net.hidden, rng=tr.rng, 
        W_scale=net.W_scale[-2], b_scale=net.b_scale[-2], activation=relu))
    out = layers[-1].output

if tr.inspect: insp = T.stack(insp[0],insp[1],insp[2],insp[3],insp[4],insp[5], T.mean(out))
else: insp =  T.stack(0,0)

if use.drop: out = DropoutLayer(out, rng=tr.rng, p=drop.p_hidden).output
#maxout
# softmax layer
Ws, bs = load_params(use) # This is test, wudi added this!
layers.append(LogRegr(out, W = Ws, b = bs, rng=tr.rng, activation=lin, n_in=net.hidden, 
    W_scale=net.W_scale[-1], b_scale=net.b_scale[-1], n_out=net.n_class))
"""
layers[-1] : softmax layer
layers[-2] : hidden layer (video if late fusion)
layers[-3] : hidden layer (trajectory, only if late fusion)
"""
# cost function
cost = layers[-1].negative_log_likelihood(y)
y_pred = layers[-1].y_pred;

if reg.L1_vid > 0 or reg.L2_vid > 0:
    # L1 and L2 regularization
    L1 = T.abs_(layers[-2].W).sum() + T.abs_(layers[-1].W).sum()
    L2 = (layers[-2].W ** 2).sum() + (layers[-1].W ** 2).sum()

    cost += reg.L1_vid*L1 + reg.L2_vid*L2 

# function computing the number of errors
errors = layers[-1].errors(y)
####################################################################
####################################################################
print "\n%s\n\tcompiling\n%s"%(('-'*30,)*2)
####################################################################
#################################################################### 
# compile functions
# ------------------------------------------------------------------------------
print 'compiling test_model'

def get_batch(_data): 
    pos_mini = idx_mini*batch.mini
    idx1 = pos_mini + idx_micro*batch.micro
    idx2 = pos_mini + (idx_micro+1)*batch.micro
    return _data[idx1:idx2]

def givens(dataset_):
    return {x: get_batch(dataset_[0]),
            y: get_batch(dataset_[1])}

test_model = function([idx_mini, idx_micro], [cost, errors, y_pred], 
    givens=givens((x_,y_int32)),
    on_unused_input='ignore')

####################################################################
####################################################################
print "\n%s\n\t testing \n%s"%(('-'*30,)*2)
####################################################################
#################################################################### 




def _avg(_list): return list(array(_list).mean(axis=0))

def _mini_batch(model, mini_batch, batch, is_train, apply_updates =None ):
    global insp_
    ce = []
    for i in xrange(batch.mini/batch.micro):
        if not is_train:
            output = model(mini_batch, i)
            ce.append([output[0], output[1]])
    return _avg(ce)

def _batch(model, batch_size, batch, is_train=True, apply_updates=None):
    ce = []
    for i in xrange(batch_size/batch.mini): ce.append(_mini_batch(model, i, batch, is_train, apply_updates))
    return _avg(ce)



ce = []
first_test_file = True
for i in range(loader.n_iter_valid):
    if first_test_file:
        augm = False
        first_test_file = False
    else: augm = True
    # load_data(file, rng, epoch, batch_size, x_, y_)
    loader.next_train_batch(x_, y_)
    #load_data(file,  rng, epoch)
    ce.append(_batch(test_model, tr.batch_size, batch, is_train=False))
    print ce


from numpy import ones, array, prod, zeros, empty, inf, float32, random
import theano.tensor as T
from theano import function, config, shared
import numpy

from classes.hyperparameters import use, lr, batch, reg, mom, tr, drop, net , files,  DataLoader
from convnet3d import ConvLayer, NormLayer, PoolLayer, LogRegr, HiddenLayer, \
                    DropoutLayer, relu, tanh
from functions.train_functions import normalize, _shared, _avg, write, ndtensor, \
                                  conv_args, var_norm, std_norm, lin,\
                                  print_params, load_data, _mini_batch, _batch,\
                                  timing_report, training_report, epoch_report, \
                                  test, test_lio, save_results, move_results, save_params, load_params

def build():
    use.load = True  # we load the CNN parameteres here
    x = ndtensor(len(tr.in_shape))(name = 'x') # video input
    x_ = _shared(empty(tr.in_shape))

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
    # prediction
    y_pred = layers[-1].y_pred
    p_y_given_x = layers[-1].p_y_given_x
    ####################################################################
    ####################################################################
    print "\n%s\n\tcompiling\n%s"%(('-'*30,)*2)
    ####################################################################
    #################################################################### 
    # compile functions
    # ------------------------------------------------------------------------------
    print 'compiling test_model'

    eval_model = function([], [y_pred, p_y_given_x], 
        givens={x:x_},
        on_unused_input='ignore')

    return eval_model, x_
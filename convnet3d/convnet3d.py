"""
3D ConvNet layers using Theano, Pylearn and Numpy

ConvLayer: convolutions, filter bank
NormLayer: normalization (LCN, GCN, local mean subtraction)
PoolLayer: pooling, subsampling
RectLayer: rectification (absolute value)

"""


from conv3d2d_new import conv3d
from maxpool3d import max_pool_3d
from activations import relu, softplus, leaky_relu

from numpy import array, sqrt, prod, ones, floor, repeat, pi, exp, zeros, sum
from numpy.random import RandomState

from theano.tensor.nnet import conv2d
from theano import shared, config, _asarray, clone
import theano.tensor as T
floatX = config.floatX


class ConvLayer(object):
    """ Convolutional layer, Filter Bank Layer """

    def __init__(self, input, n_in_maps, n_out_maps, kernel_shape, video_shape, 
        batch_size, activation, layer_name="Conv", rng=RandomState(1234), 
        borrow=True, stride=1, W=None, b=None, b_scale=0.1, W_scale=0.01, fast_conv=False):

        """
        video_shape: (frames, height, width)
        kernel_shape: (frames, height, width)

        W_shape: (out, in, kern_frames, kern_height, kern_width)
        """

        self.__dict__.update(locals())
        del self.self
        
        # init W
        if W != None: self.W = shared(array(W, dtype=floatX), borrow=borrow, name=layer_name+'_W') # wudi made it shared
        else: 
            # fan in: filter time x filter height x filter width x input maps
            fan_in = prod(kernel_shape)*n_in_maps
            norm_scale = 2. * sqrt( 1. / fan_in )
            if activation in ('relu', 'softplus', 'leaky_relu'): 
                print activation
                norm_scale = W_scale
            W_shape = (n_out_maps, n_in_maps)+kernel_shape
            W_val = _asarray(rng.normal(loc=0, scale=norm_scale, size=W_shape),\
                        dtype=floatX)
            # W_val = ones(W_shape, dtype=floatX)*W_scale
            self.W = shared(value=W_val, borrow=borrow, name=layer_name+'_W')
        self.params = [self.W]

        # init bias
        if b != None: 
            self.b = shared(array(b, dtype=floatX), name=layer_name+"_b", borrow=borrow) # wudi made it shared
        elif activation in ('relu', 'softplus', 'leaky_relu'): 
            # print b_scale
            b_val = (ones((n_out_maps,), dtype=floatX)*b_scale).astype(floatX)
            self.b = shared(b_val, name=layer_name+"_b", borrow=borrow)
        else: 
            b_val = zeros((n_out_maps,), dtype=floatX)
            self.b = shared(b_val, name=layer_name+"_b", borrow=borrow)
        self.params.append(self.b)

        # 3D convolution; dimshuffle: last 3 dimensions must be (in, h, w)
        n_fr, h, w = video_shape
        n_fr_k, h_k, w_k = kernel_shape
        out = conv3d(
                signals=input.dimshuffle([0,2,1,3,4]), 
                filters=self.W.dimshuffle([0,2,1,3,4]), 
                signals_shape=(batch_size, n_fr, n_in_maps, h, w), 
                filters_shape=(n_out_maps, n_fr_k, n_in_maps, h_k, w_k),         
                border_mode='valid',
                fast_conv=fast_conv,
                stride=stride
                ).dimshuffle([0,2,1,3,4])

        out += self.b.dimshuffle('x',0,'x','x','x')

        self.output = eval(activation)(out)


class NormLayer(object):
    """ Normalization layer """

    def __init__(self, input, method="lcn", **kwargs):
        """
        method: "lcn", "gcn", "mean"

        LCN: local contrast normalization
            kwargs: 
                kernel_size=9, threshold=1e-4, use_divisor=True

        GCN: global contrast normalization
            kwargs:
                scale=1., subtract_mean=True, use_std=False, sqrt_bias=0., 
                min_divisor=1e-8

        MEAN: local mean subtraction
            kwargs:
                kernel_size=5
        """
        if method=="ccn":
            self.output = self.cross_channel_norm(input, **kwargs)
        else:
            input_shape = input.shape

            # make 4D tensor out of 5D tensor -> (n_images, 1, height, width)
            input_shape_4D = (input_shape[0]*input_shape[1]*input_shape[2], 1,
                                input_shape[3], input_shape[4])
            input_4D = input.reshape(input_shape_4D, ndim=4)
            if method=="lcn":
                out = self.lecun_lcn(input_4D, **kwargs)
            elif method=="lcn2":
                out = self.lecun_lcn_2(input_4D, input_shape, **kwargs)
            elif method=="gcn":
                out = self.global_contrast_normalize(input_4D,**kwargs)
            elif method=="mean":
                out = self.local_mean_subtraction(input_4D, **kwargs)
            else:
                raise NotImplementedError()

            self.output = out.reshape(input_shape)

    def cross_channel_norm(self, X, shape=None, alpha = 1e-4, beta=0.75, n=5):
        from pylearn2.sandbox.cuda_convnet.response_norm import CrossMapNorm
        from theano.sandbox.cuda.basic_ops import gpu_contiguous
        # from theano.sandbox.cuda import gpu_from_host, host_from_gpu

        #shape: BCF01
        input_C01BF = X.dimshuffle(1,3,4,0,2) #C01BF
        # s = input_C01BF.shape
        # shape_C01B = T.cast(s[:-2]+(s[-2]*s[-1],), 'int32')

        shape_C01B = (
            shape[1],
            shape[3],
            shape[4],
            shape[0]*shape[2],
            )
        shape_C01BF = (
            shape[1],
            shape[3],
            shape[4],
            shape[0],
            shape[2],
            )

        input_C01B = input_C01BF.reshape(shape_C01B) #C01B

        op = CrossMapNorm(size_f=n,add_scale=alpha,pow_scale=beta,blocked=True)

        # return host_from_gpu(op(gpu_from_host(X))[0])
        out = op(gpu_contiguous(input_C01B))[0]
        # out = host_from_gpu(op(gpu_from_host(input_C01B))[0])

        out = out.reshape(shape_C01BF) #C01BF
        out = out.dimshuffle(3,0,4,1,2) #BCF01

        return out


    def lecun_lcn_2(self, X, input_shape, kernel_size=7, threshold = 1e-8, use_divisor=False):
        """
        Yann LeCun's local contrast normalization
        Orginal code in Theano by: Guillaume Desjardins
        """

        filter_shape = (1, 1, kernel_size, kernel_size)
        filters = gaussian_filter(kernel_size).reshape(filter_shape)
        filters = shared(_asarray(filters, dtype=floatX), borrow=True)

        convout = conv2d(X, filters=filters, filter_shape=filter_shape, 
                            border_mode='full')

        # For each pixel, remove mean of kernel_sizexkernel_size neighborhood
        mid = int(floor(kernel_size/2.))
        mean_X = convout[:,:,mid:-mid,mid:-mid]
        mean_X = mean_X.reshape(input_shape) #(batch, maps, fr, h, w)
        mean_X = T.mean(mean_X, 1).dimshuffle(0,'x',1,2,3)
        mean_X = T.repeat(mean_X, input_shape[1], 1)
        mean_X = mean_X.reshape(X.shape)
        new_X = X - mean_X

        if use_divisor:
            # Scale down norm of kernel_sizexkernel_size patch
            sum_sqr_XX = conv2d(T.sqr(T.abs_(X)), filters=filters, 
                                filter_shape=filter_shape, border_mode='full')

            denom = T.sqrt(sum_sqr_XX[:,:,mid:-mid,mid:-mid])
            per_img_mean = denom.mean(axis=[2,3])
            divisor = T.largest(per_img_mean.dimshuffle(0,1,'x','x'), denom)
            divisor = T.maximum(divisor, threshold)

            new_X /= divisor

        return new_X

    def lecun_lcn(self, X, kernel_size=9, threshold = 1e-4, use_divisor=True, border=False):
        """
        Yann LeCun's local contrast normalization
        Orginal code in Theano by: Guillaume Desjardins
        """

        filter_shape = (1, 1, kernel_size, kernel_size)
        filters = gaussian_filter(kernel_size).reshape(filter_shape)
        filters = shared(_asarray(filters, dtype=floatX), borrow=True)
        mid = int(floor(kernel_size/2.))

        if border:
            r = (kernel_size-1)/2
            up = X[:,:,0:1,:].repeat(r,axis=2)
            down = X[:,:,-1:,:].repeat(r,axis=2)
            X_ = T.concatenate([up,X,down],axis=2)
            left = X_[:,:,:,0:1].repeat(r,axis=3)
            right = X_[:,:,:,-1:].repeat(r,axis=3)
            X_ = T.concatenate([left,X_,right],axis=3)

            convout = conv2d(X_, filters=filters, filter_shape=filter_shape, 
                                border_mode='valid')
            centered_X = X - convout

        else:
            convout = conv2d(X, filters=filters, filter_shape=filter_shape, 
                                border_mode='full')

            # For each pixel, remove mean of kernel_sizexkernel_size neighborhood
            centered_X = X - convout[:,:,mid:-mid,mid:-mid]

        if use_divisor:
            # Scale down norm of kernel_sizexkernel_size patch
            sum_sqr_XX = conv2d(T.sqr(X), filters=filters, 
                                filter_shape=filter_shape, border_mode='full')

            sum_sqr_XX = sum_sqr_XX[:,:,mid:-mid,mid:-mid]
            sum_sqr_XX = T.maximum(sum_sqr_XX, threshold)
            denom = T.sqrt(sum_sqr_XX)
            # denom = abs(centered_X)
            per_img_mean = denom.mean(axis=[2,3])
            divisor = T.largest(per_img_mean.dimshuffle(0,1,'x','x'), denom)
            divisor = T.maximum(divisor, threshold)

            new_X = centered_X / divisor
            return new_X

        else: 
            return centered_X


    def local_mean_subtraction(self, X, kernel_size=5):
         
        filter_shape = (1, 1, kernel_size, kernel_size)
        filters = mean_filter(kernel_size).reshape(filter_shape)
        filters = shared(_asarray(filters, dtype=floatX), borrow=True)

        mean = conv2d(X, filters=filters, filter_shape=filter_shape, 
                        border_mode='full')
        mid = int(floor(kernel_size/2.))

        return X - mean[:,:,mid:-mid,mid:-mid]

    def global_contrast_normalize(self, X, scale=1., subtract_mean=True, 
        use_std=False, sqrt_bias=0., min_divisor=1e-8):

        ndim = X.ndim
        if not ndim in [3,4]: raise NotImplementedError("X.dim>4 or X.ndim<3")

        scale = float(scale)
        mean = X.mean(axis=ndim-1)
        new_X = X.copy()

        if subtract_mean:
            if ndim==3:
                new_X = X - mean[:,:,None]
            else: new_X = X - mean[:,:,:,None]

        if use_std:
            normalizers = T.sqrt(sqrt_bias + X.var(axis=ndim-1)) / scale
        else:
            normalizers = T.sqrt(sqrt_bias + (new_X ** 2).sum(axis=ndim-1)) / scale

        # Don't normalize by anything too small.
        T.set_subtensor(normalizers[(normalizers < min_divisor).nonzero()], 1.)

        if ndim==3: new_X /= normalizers[:,:,None]
        else: new_X /= normalizers[:,:,:,None]

        return new_X


class PoolLayer(object):
    """ Subsampling and pooling layer """

    def __init__(self, input, pool_shape, method="max"):
        """
        method: "max", "avg", "L2", "L4", ...
        """

        self.__dict__.update(locals())
        del self.self

        if method=="max":
            out = max_pool_3d(input,pool_shape)
        else:
            raise NotImplementedError()

        self.output = out


class RectLayer(object):
    """  Rectification layer """

    def __init__(self, input):
        self.output = T.abs_(input)


def gaussian_filter(kernel_shape):

    x = zeros((kernel_shape, kernel_shape), dtype='float32')

    def gauss(x, y, sigma=2.0):
        Z = 2 * pi * sigma**2
        return  1./Z * exp(-(x**2 + y**2) / (2. * sigma**2))

    mid = floor(kernel_shape/ 2.)
    for i in xrange(0,kernel_shape):
        for j in xrange(0,kernel_shape):
            x[i,j] = gauss(i-mid, j-mid)

    return x / sum(x)


def mean_filter(kernel_size):
    s = kernel_size**2
    x = repeat(1./s, s).reshape((kernel_size, kernel_size))
    return x
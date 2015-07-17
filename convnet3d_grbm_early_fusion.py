# numpy imports
from numpy import zeros, empty, inf, float32, random
import cPickle, os

# modular imports
# the hyperparameter set the data dir, use etc classes, it's important to modify it according to your need
from classes.hyperparameters import use, lr, batch, reg, mom, tr, drop,\
                                     net,  DataLoader_with_skeleton_normalisation

from functions.train_functions import _shared, _avg, write, ndtensor, print_params, lin,\
                                      training_report, epoch_report, _batch,\
                                      save_results, move_results, save_params, test_lio_skel

# theano imports
from theano import function, config, shared
import theano.tensor as T
# customized imports
from dbn.GRBM_DBN import GRBM_DBN
from conv3d_chalearn import conv3d_chalearn
from convnet3d import LogRegr, HiddenLayer, DropoutLayer

class convnet3d_grbm_early_fusion():

    def __init__(self, res_dir, load_path):
        
        self.layers = [] # only contain the layers from fusion
        self.insp_mean = [] # inspection for each layer mean activation
        self.insp_std = []  # inspection for each layer std activation
        self.params = [] # parameter list
        self.idx_mini = T.lscalar(name="idx_mini") # minibatch index
        self.idx_micro = T.lscalar(name="idx_micro") # microbatch index

        # symbolic variables
        self.x = ndtensor(len(tr.in_shape))(name = 'x') # video input
        self.y = T.ivector(name = 'y') # labels
        # symbolic variables
        self.x_skeleton = ndtensor(len(tr._skeleon_in_shape))(name = 'x_skeleton') # video input

        if use.drop: 
            drop.p_vid = shared(float32(drop.p_vid_val) )
            drop.p_hidden = shared(float32(drop.p_hidden_val))
        video_cnn = conv3d_chalearn(self.x, use, lr, batch, net, reg, drop, mom, \
                                             tr, res_dir, load_path)

        dbn = GRBM_DBN(numpy_rng=random.RandomState(123), n_ins=891, \
                hidden_layers_sizes=[2000, 2000, 1000], n_outs=101, input_x=self.x_skeleton, label=self.y ) 
        # we load the pretrained DBN skeleton parameteres here
        if use.load == True: dbn.load(os.path.join(load_path,'dbn_2015-06-19-11-34-24.npy'))


        #####################################################################
        # fuse the ConvNet output with skeleton output  -- need to change here
        ######################################################################  
        out = T.concatenate([video_cnn.out, dbn.sigmoid_layers[-1].output], axis=1)

        #####################################################################
        # wudi add the mean and standard deviation of the activation values to exam the neural net
        # Reference: Understanding the difficulty of training deep feedforward neural networks, Xavier Glorot, Yoshua Bengio
        #####################################################################
        insp_mean_list = []
        insp_std_list = []
        insp_mean_list.extend(dbn.out_mean)
        insp_mean_list.extend(video_cnn.insp_mean)
        insp_std_list.extend(dbn.out_std)
        insp_std_list.extend(video_cnn.insp_std)

        ######################################################################
        #MLP layer                
        self.layers.append(HiddenLayer(out, n_in=net.hidden, n_out=net.hidden, rng=tr.rng, 
            W_scale=net.W_scale[-1], b_scale=net.b_scale[-1], activation=net.activation))
        out = self.layers[-1].output

        if tr.inspect: 
            insp_mean_list.extend([T.mean(out)])
            insp_std_list.extend([T.std(out)])
        self.insp_mean = T.stacklists(insp_mean_list)
        self.insp_std = T.stacklists(insp_std_list)


        if use.drop: out = DropoutLayer(out, rng=tr.rng, p=drop.p_hidden).output

        ######################################################################
        # softmax layer
        self.layers.append(LogRegr(out, rng=tr.rng, n_in=net.hidden, 
            W_scale=net.W_scale[-1], b_scale=net.b_scale[-1], n_out=net.n_class))

        self.p_y_given_x = self.layers[-1].p_y_given_x
        ######################################################################
        # cost function
        self.cost = self.layers[-1].negative_log_likelihood(self.y)

        # function computing the number of errors
        self.errors = self.layers[-1].errors(self.y)

        # parameter list
        for layer in video_cnn.layers: 
            self.params.extend(layer.params)

        # pre-trained dbn parameter last layer  (W, b) doesn't need to incorporate into the params
        # for calculating the gradient
        self.params.extend(dbn.params[:-2])

        # MLP hidden layer params
        self.params.extend(self.layers[-2].params)
        # softmax layer params
        self.params.extend(self.layers[-1].params)
        # number of inputs for MLP = (# maps last stage)*(# convnets)*(resulting video shape) + trajectory size
        print 'MLP:', video_cnn.n_in_MLP, "->", net.hidden_penultimate, "+", net.hidden_traj, '->', \
           net.hidden, '->', net.hidden, '->', net.n_class, ""

        return 

    def load_normalisation_constant(self, load_path):
        # load the normalisation for the skeleton       
        f = open(os.path.join(load_path, 'SK_normalization.pkl'),'r')
        SK_normalization = cPickle.load(f)
        Mean_skel = SK_normalization ['Mean1']
        Std_skel = SK_normalization['Std1']

        # load the normalisation for the 3dcnn
        f = open(os.path.join(load_path, 'CNN_normalization.pkl'),'r')
        CNN_normalization = cPickle.load(f)
        Mean_CNN = CNN_normalization ['Mean_CNN']
        Std_CNN = CNN_normalization['Std_CNN']

        return  Mean_skel, Std_skel, Mean_CNN, Std_CNN

    def build_finetune_functions(self, x_, y_int32, x_skeleton_, learning_rate):
        '''
        This is used to fine tune the network
        '''
         # compute the gradients with respect to the model parameters
        gparams = T.grad(self.cost, self.params)
        # compute list of fine-tuning updates
        mini_updates = []
        micro_updates = []
        update = []
        last_upd = []

        # shared variables
        
        if use.mom: momentum = shared(float32(mom.momentum))
        def get_update(i): return update[i]/(batch.mini/batch.micro)

        for i, (param, gparam) in enumerate(zip(self.params, gparams)):
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


        def get_batch(_data): 
            pos_mini = self.idx_mini*batch.mini
            idx1 = pos_mini + self.idx_micro*batch.micro
            idx2 = pos_mini + (self.idx_micro+1)*batch.micro
            return _data[idx1:idx2]

        def givens(dataset_):
            return {self.x: get_batch(dataset_[0]),
                    self.y: get_batch(dataset_[1]),
                    self.x_skeleton: get_batch(dataset_[2])}

        print 'compiling apply_updates'
        apply_updates = function([], 
            updates=mini_updates, 
            on_unused_input='ignore')

        print 'compiling train_model'
        train_model = function([self.idx_mini, self.idx_micro], [self.cost, self.errors, self.insp_mean, self.insp_std], 
            updates=micro_updates, 
            givens=givens((x_, y_int32, x_skeleton_)), 
            on_unused_input='ignore')

        print 'compiling test_model'
        test_model = function([self.idx_mini, self.idx_micro], [self.cost, self.errors, self.insp_mean, self.insp_std], 
            givens=givens((x_, y_int32, x_skeleton_)),
            on_unused_input='ignore')

        return apply_updates, train_model, test_model

    def load_params(self, load_file=""):
        import os
        from gzip import GzipFile
        from cPickle import dump, load
        if os.path.isfile(load_file):
            file = GzipFile(load_file, "rb")
        param_load = load(file)
        file.close()
        load_params_pos = 0
        for p in self.params:
            #print p.get_value().shape
            #print param_load[load_params_pos].shape
            p.set_value(param_load[load_params_pos], borrow=True)
            load_params_pos += 1 
        print "finish loading parameters"

    def prediction_function(self, x_, x_skeleton_):
        '''
        This is used to have the output function using video and skeleton
        '''

        print 'compiling test_model'

        test_model = function([], self.p_y_given_x, 
            givens={self.x: x_,
                    self.x_skeleton: x_skeleton_}, 
            on_unused_input='ignore')

        return test_model
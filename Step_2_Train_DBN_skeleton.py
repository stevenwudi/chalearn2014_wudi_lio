import cPickle
import gzip
import os
import sys
import time
import numpy
import sys

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from dbn.logistic_sgd import LogisticRegression
from dbn.mlp import HiddenLayer
from dbn.rbm import RBM
from dbn.grbm import GBRBM
from dbn.utils import zero_mean_unit_variance
from dbn.utils import normalize
from dbn.GRBM_DBN import GRBM_DBN

from sklearn import preprocessing

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                            dtype=theano.config.floatX),
                                borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                            dtype=theano.config.floatX),
                                borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')

def load_CodaLab_skel():
    print '... loading data'

    # Wudi hard coded this part.... sorry....
    #src = r"D:\Chalearn2014\Data_processed"
    src = "/idiap/user/dwu/chalearn"
    import h5py
    file = h5py.File(src+"/data%d.hdf5", "r", driver="family", memb_size=2**32-1)
  
    Feature_train = file["x_train_skeleton_feature"]
    Target_all = file["y_train"]
    #print Feature_train[0,:100]
    #print Target_all[200:300]
    
    valid_set_feature = file["x_valid_skeleton_feature"]
    valid_set_new_target = file["y_valid"]
    # Wudi added normalized data for GRBM
    import cPickle as pickle
    """
    [train_set_feature_normalized, Mean1, Std1]  = zero_mean_unit_variance(Feature_train)
    f = open('SK_normalization.pkl','wb')
    pickle.dump( {"Mean1": Mean1, "Std1": Std1 },f)
    f.close()
    train_set_x, train_set_y = shared_dataset( (train_set_feature_normalized, Target_all))
"""
    f = open('SK_normalization.pkl','rb')
    SK_normalisation = cPickle.load(f)
    Mean1 = SK_normalisation['Mean1']
    Std1 = SK_normalisation['Std1']

    #Feature_train = numpy.array(Feature_train)
    Feature_train -= Mean1
    Feature_train /= Std1
    
    train_set_x, train_set_y = shared_dataset( (Feature_train, Target_all))
    valid_set_feature = normalize(valid_set_feature, Mean1, Std1)
    valid_set_x, valid_set_y = shared_dataset((valid_set_feature,valid_set_new_target))
    print "finish loading"

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
    return rval

def test_GRBM_DBN(finetune_lr=1, pretraining_epochs=2,
             pretrain_lr=0.01, k=1, training_epochs=2,
             batch_size=200, annealing_learning_rate=0.99999):
    """
    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training
    :type k: int
    :param k: number of Gibbs steps in CD/PCD
    :type training_epochs: int
    :param training_epochs: maximal number of iterations ot run the optimizer
    :type dataset: string
    :param dataset: path the the pickled dataset
    :type batch_size: int
    :param batch_size: the size of a minibatch
    """

    datasets = load_CodaLab_skel()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'
    # construct the Deep Belief Network
    dbn = GRBM_DBN(numpy_rng=numpy_rng, n_ins=891,
              hidden_layers_sizes=[2000, 2000, 1000],
              n_outs=101, finetune_lr=finetune_lr)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    for i in xrange(dbn.n_layers):      
        if i==0:
            # for GRBM, the The learning rate needs to be about one or 
            #two orders of magnitude smaller than when using
            #binary visible units and some of the failures reported in the 
            # literature are probably due to using a
            pretrain_lr_new = pretrain_lr*0.1 
        else:
            pretrain_lr_new = pretrain_lr
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            start_time_temp = time.clock()
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr_new))
            end_time_temp = time.clock()
            print 'Pre-training layer %i, epoch %d, cost %f ' % (i, epoch, numpy.mean(c)) + ' ran for %d sec' % ((end_time_temp - start_time_temp) )

    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                annealing_learning_rate=annealing_learning_rate)

    print '... finetunning the model'
    # early-stopping parameters
    patience = 4 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is
                              # found
    improvement_threshold = 0.999  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        start_time_temp = time.clock()
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                import warnings
                warnings.filterwarnings("ignore")
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss



                    end_time_temp = time.clock()
                    print(('epoch %i, minibatch %i/%i, validation error %f %%, used time %d sec') %
                          (epoch, minibatch_index + 1, n_train_batches,this_validation_loss * 100.,
                            (end_time_temp - start_time_temp)))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))
    from time import gmtime, strftime
    filename = 'dbn_'+strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    dbn.save(filename)






if __name__ == '__main__':
    test_GRBM_DBN()

#if 1: # here for testing, where we never used
### Now for testing
#dbn = GRBM_DBN(numpy_rng=numpy_rng, n_ins=891,
#hidden_layers_sizes=[2000, 2000, 1000],
#n_outs=101)

    
#dbn.load('dbn_2015-01-01-18-01-07.npy')
## compiling a Theano function that computes the mistakes that are made by
## the model on a minibatch
#index = T.lscalar('index') 
#validate_model = theano.function(inputs=[index],
#    outputs=dbn.logLayer.p_y_given_x,
#    givens={
#        dbn.x: valid_set_x[index * batch_size:(index + 1) * batch_size]})

#validate_model = theano.function([index], outputs=dbn.errors,
#              givens={dbn.x: valid_set_x[index * batch_size:
#                                          (index + 1) * batch_size],
#                      dbn.y: valid_set_y[index * batch_size:
#                                          (index + 1) * batch_size]})

#validate_model = theano.function(inputs=[index],
#    outputs=dbn.logLayer.y_pred,
#    givens={
#        dbn.x: valid_set_x[index * batch_size:(index + 1) * batch_size]})

#n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
#n_valid_batches /= batch_size
#temp = [validate_model(i)  for i in xrange(n_valid_batches)]
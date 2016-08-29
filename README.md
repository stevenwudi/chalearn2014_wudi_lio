chalearn2014_wudi_lio 
=====================


Citation
-------
If you use this toolbox as part of a research project, please consider citing the corresponding paper
******************************************************************************************************

@inproceedings{IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI)},

  title={Deep Dynamic Neural Networks for Multimodal Gesture Segmentation and Recognition},
  
  author={Di Wu, Lionel Pigou, Pieter-Jan Kindermans, Nam LE, Ling Shao, Joni Dambr},
  
  year={2016}
}
******************************************************************************************************


Dependency: Theano
-------

To train the network, you first need to run the following code:
This is the very first file that you should run to extract training data (skeleton data and the depth and rgb data).

(1) `Step1_preproc.py`

Note I used first 650 examples for training and 50 for validation with 1000 frames per storage(line 87 and 95).

-  Change input directory: line 34-39
-  Change destination directory:  lin 85-101



(2) `Step_1_preproc_hdf5_skeleton.py`:

Save the file into hdf5 file for easy read.


(3) `Step_2_DBN_train_small_batch.py`:

To train the skelenton module used the pre-trained RBM weights.


(4) `Step_3_train_CNN_normalisation.py`:

To train the rgb and depth module using CNN.

In the file: classes/hyperparameters.py you will have all the specs, e.g., train, valid dir,line 14-19:
Note: line 27: use.fast_conv

(5) `Step_4_Train_CNN_DBN_argparser.py`:

To train the early fusion network using pre-trained weights.

Contact
-------
If you read the code and find it really hard to understand, please send feedback to: stevenwudi@gmail.com
Thank you!

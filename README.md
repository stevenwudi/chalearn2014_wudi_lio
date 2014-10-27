chalearn2014_wudi_lio 
=====================


Citation
-------
If you use this toolbox as part of a research project, please consider citing the corresponding paper
******************************************************************************************************
@inproceedings{..,
  title={},
  author={},
  booktitle={},
  year={}
}
******************************************************************************************************


Dependency: Theano
-------

	
To Lio:
-------
To train the network, you first need to run and change the following code:

(1) Step1_preproc.py:

Note I used first 650 examples for training and 50 for validation with 1000 frames per storage. (line 85-91)

change input directory: line 37: raise NotImplementedError("TODO: implement this function.")-->set to  data = r"I:\Kaggle_multimodal\Training"
change destination directory:  lin 87-90: dest = r"I:\Kaggle_multimodal\Training_prepro\train_wudi" # dir to  destination processed data


(2) Step2_Train_CNN.py:
in the file: classes/hyperparameters.py you will have all the specs, e.g., train, valid dir,line 14-19:
line 27: use.fast_conv

It takes about 600 second for each example file . (I use Theano GPU model, but I reckon CPU model should almost of the same speed)

Train
-------
Voila, here you go.


Contact
-------
If you read the code and find it really hard to understand, please send feedback to: stevenwudi@gmail.com
Thank you!

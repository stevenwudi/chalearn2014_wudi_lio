from functions.test_functions import *


outPred=r'C:\Users\PC-User\Documents\GitHub\chalearn2014_wudi_lio\CNN_valid_pred'

TruthDir=r'I:\Kaggle_multimodal\Code_for_submission\Final_project\training\gt'


#outPred=r'C:\Users\PC-User\Documents\GitHub\chalearn2014_wudi_lio\CNN_test_pred_combine_sk_cnn'
#TruthDir=r'I:\Kaggle_multimodal\Test_label'
final_score = evalGesture(outPred,TruthDir, begin_add=0, end_add=-1) 


print("The score for this prediction is " + "{:.12f}".format(final_score))

#The score for this prediction is 0.816150551922--combined
#The score for this prediction is 0.787309630209--skeleton

#begin +1, end: 0 --0.747930669661
#begin 0, end: +1 --0.748291985589
#begin -1, end: 0  --0.743814904906
#begin 0, end -1:  --0.742372259401
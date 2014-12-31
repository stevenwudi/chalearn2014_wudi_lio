from functions.visualize_functions import *
from functions.train_functions import load_params
from classes.hyperparameters import use, DataLoader, tr

# eps. file save path can be changed inside the function
options = {0: gray_body_conv1, 
           1: gray_hand_conv1, 
           2: depth_body_conv1,
           3: depth_hand_conv1,
           10: draw_original, # which frame to plot can be changed inside the function
                            # here we have chosen a top 1000 random frame to plot
           30: 'hand_original',

           100: plot_confusion_matrix,
           200: plot_cnn_error_rate,
           }

#options[0]()
#options[1]()
#options[2]()
#options[3]()
#options[10]()

options[200]()
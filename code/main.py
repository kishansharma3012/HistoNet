import sys,os,time,random
import os
from config import *
from utils import *

## HyperParameter Tuning
#lr = [ 5e-5, 5e-6]
#wd = [ 5e-6, 5e-7]
#lr = [5e-5, 1e-5, 1e-6, 1e-7]
#wd = [ 1e-4, 1e-5, 1e-6, 1e-7]
lr = [8e-3]
wd = [1e-4]
lr_decay = 0.95

root_result = os.getcwd() + "/Results_CountHistNet_larvae_DSN_8_5/"

if not os.path.exists(root_result):
    os.mkdir(root_result)

Model_no = 1
best_val_err_l1 = 10000
best_Model_number = 100000
Hist_wt = [0.5, 0.2, 0.3] #Total, Dsn_1, Dsn_2
for learning_rate in lr:
    for reg in wd:
        ## Setup Model
        print("\n \n \n Preparing Model......", Model_no)
        net, net_hist, dsn_1, dsn_2, input_var, input_var_ex = model()

        loss_name = 'w_L1'
        ## Compile Theano Loss Functions
        loss_list, variables, classify, classify_test = loss_func_1(net, net_hist, dsn_1, dsn_2, input_var, input_var_ex, reg, loss_name= loss_name, sum_flag = False, Hist_wt = Hist_wt)
        
        ## Train Model
        epochs = 40
        epochs_dsn1 = 20
        epochs_dsn2 = 20
        batch_size = 4
        
        Experiment_name = 'Model_exp' + str(Model_no) + '_net1_' + loss_name + '_lr_' + str(learning_rate) + '_wd_' + str(reg)
        directory = root_result + Experiment_name + '/'

        if not os.path.exists(directory):
            os.mkdir(directory)
        #val_err_l1, Training_info, Best_info = trainer(net, net_hist, dsn_1, dsn_2, loss_list, variables, epochs, learning_rate, lr_decay, batch_size, reg, root_result, directory, Experiment_name, classify, classify_test, loss_name = loss_name, sum_flag = False, Hist_wt = Hist_wt)
        val_err_l1, Training_info, Best_info = trainer_1(net, net_hist, dsn_1, dsn_2, loss_list, variables, epochs, epochs_dsn1, epochs_dsn2, learning_rate, lr_decay, batch_size, reg, root_result, directory, Experiment_name, classify, classify_test, loss_name = loss_name, sum_flag = False, Hist_wt = Hist_wt)
        
        if val_err_l1 < best_val_err_l1 :
            best_valid_err_l1 = val_err_l1
            best_Model_number = Model_no

        Model_no += 1
        #Test_directory = directory + '/Test_results/'
        Result_file  = open(root_result + '/Summary.txt', "a+")
        Result_file.write("Best Model Number is : " + str(best_Model_number))
        Result_file.close()
        print("Best Model Number is : ", best_Model_number)

import sys,os,time,random
import os
from data_utils import *
from net import *
from train import *

lr = 8e-3
reg = 1e-4
lr_decay = 0.95
Loss_wt = [0.5, 0.5]
num_bins = 8
loss_name = 'w_L1'
epochs = 100
batch_size = 4
root_result = os.getcwd() + "/Results_HistoNet/"
Experiment_name = "HistoNet_lr_" + str(lr) + "wd_" + str(reg) 
dataset_path =  #TODO: Insert dataset_path

## Setup Network
print("\n \n \n Preparing Network......")
net_count, net_hist, input_var, input_var_ex = HistoNet(num_bins = num_bins)

## Compile Theano Loss Functions
loss_list, placeholder_list, train_op, test_op = loss_func_histonet(net_count, net_hist, input_var, input_var_ex, reg, loss_name= loss_name, Loss_wt = Loss_wt)

## Train and Evaluate Network
trainer_histonet(net_count, net_hist, dataset_path, loss_list, placeholder_list, epochs, lr, \
                    lr_decay, batch_size, reg, root_result, Experiment_name,\
                    train_op, test_op, loss_name = loss_name,  Loss_wt = Loss_wt, num_bins = num_bins, print_every=50):
   

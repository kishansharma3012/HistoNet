import sys,os,time,random
import os
from data_utils import *
from net import *
from train import *
import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="HistoNet")
    parser.add_argument("-d", "--dataset_path", type=str, metavar="N", help="Dataset directory")
    parser.add_argument("--experiment_name", type=str, metavar="N", help="name of experiment")
    parser.add_argument("--output_dir", type=str, metavar="N", help="output_dir")
    parser.add_argument("--loss_name", type=str,  default="w_L1", metavar="N", help="Loss for HistoNet L1 or w_L1")
    parser.add_argument("--num_epochs", type=int,  default=100, metavar="N", help="Number of epochs")
    parser.add_argument("--batch_size", type=int,  default=4, metavar="N", help="batch size")
    parser.add_argument("--num_bins", type=str, default="8", metavar="N", help="Num of bins for Histonet 8 or HistoNet_DSN [2,4,8]")
    parser.add_argument("--Loss_wt",  type=str, default="0.5, 0.5", metavar="N", help="Loss wt. for countception and histogram prediction")
    parser.add_argument("--lr_decay", type=float,  default=0.95, metavar="N", help="Learning rate decay per epoch")
    parser.add_argument("--lr", type=float,  default=8e-3, metavar="N", help="Learning rate")
    parser.add_argument("--reg", type=float,  default=1e-4, metavar="N", help="Regularization strength")
    parser.add_argument("--DSN", type=str,  default="False", metavar="N", help="Network type HistoNet or HistoNet-DSN")

    args = parser.parse_args()

    lr = args.lr
    reg = args.reg
    lr_decay = args.lr_decay
    Loss_wt = [float(x) for x in args.Loss_wt.split(",")]
    loss_name = args.loss_name
    epochs = args.num_epochs
    batch_size = args.batch_size
    root_result = args.output_dir
    Experiment_name = args.experiment_name
    dataset_path = args.dataset_path

    if  args.DSN == "False":
        num_bins = int(args.num_bins[0])
        ## Setup Network
        print("\n \n \n Preparing Network......")
        net_count, net_hist, input_var, input_var_ex = HistoNet(num_bins = num_bins)
        ## Compile Theano Loss Functions
        loss_list, placeholder_list, train_op, test_op = loss_func_histonet(net_count, net_hist, input_var, input_var_ex, reg, loss_name= loss_name, Loss_wt = Loss_wt)
        
        ## Train and Evaluate Network
        trainer_histonet(net_count, net_hist, dataset_path, loss_list, placeholder_list, epochs, lr, \
                            lr_decay, batch_size, reg, root_result, Experiment_name,\
                            train_op, test_op, loss_name = loss_name,  Loss_wt = Loss_wt, num_bins = num_bins, print_every=50)
        

    else:
        num_bins = [int(x) for x in args.num_bins.split(",")]
        ## Setup Network
        print("\n \n \n Preparing Network......")
        net_count, net_hist, net_hist_dsn1, net_hist_dsn2, input_var, input_var_ex = HistoNet_DSN(num_bins = num_bins)
        ## Compile Theano Loss Functions
        loss_list, placeholder_list, train_op, test_op = loss_func_histonet_dsn(net_count, net_hist, net_hist_dsn1, net_hist_dsn2, input_var, input_var_ex, reg, loss_name= loss_name, Loss_wt = Loss_wt)

        ## Train and Evaluate Network
        trainer_histonet_dsn(net_count, net_hist, net_hist_dsn1, net_hist_dsn2, dataset_path, loss_list, placeholder_list, epochs, lr, \
                        lr_decay, batch_size, reg, root_result, Experiment_name,\
                        train_op, test_op, loss_name = loss_name,  Loss_wt = Loss_wt, num_bins = num_bins, print_every=50)
    

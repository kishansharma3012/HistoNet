import lasagne
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.set_cmap('jet')
import sys,os,time,random
import numpy as np
import theano
import theano.tensor as T 
from data_utils import import_data, import_data_dsn

seed = 0 
random.seed(seed)
np.random.seed(seed)
lasagne.random.set_rng(np.random.RandomState(seed))

def evaluate_histonet(dataset_image, dataset_countmap, dataset_count, dataset_hist, Hist_wt, test_op, data_mean, loss_list,\
                visualize = False, path = 'Results/', loss_name = 'w_L1',  Loss_wt= [0.5, 0.5,], num_bins = 8):
    """
    Function: Evaluate HistoNet network performance on a dataset (For validation and Testing)
    
    Input:
    dataset_image : Input images
    dataset_countmap: Gt Redundant countmaps
    dataset_count: Gt object counts
    dataset_hist: Gt object size distribution histograms
    Hist_wt: Weights for L1 loss between predicted and target size histogram
    test_op: Operation for extracting output from network for a given input during test and validation(deterministic)
    data_mean: Mean image of training dataset 
    loss_list: list of different losses (only for initializing the test loss history list)
    visualize: Boolean (to visualize the test results)
    path : path of the result folder
    loss_name: Name of the L1 loss for histogram (Weighted L1 loss - w_L1 or L1 loss - L1)
    Loss_wt: Weights for KL divergence loss and L1 loss for histogram
    num_bins: number of the bins of histogram
    
    Output:
    test_loss_history: list of lists of different losses [loss_total, loss_count, loss_pix, loss_kl, loss_l1, loss_l1_temp, loss_reg (0 for the test)]
    gt_count: list of object count in the test samples
    pred_count: list of predicted count in the test samples
    """
    
    test_loss_history = [[]] * len(loss_list)
    gt_count = []
    pred_count = []
    batchsize = 1

    for i in range(0,len(dataset_image), batchsize):
        pred_countmap, pred_hist = test_op(dataset_image, range(i,i+batchsize))

        if visualize:
            visualize_HistoNet_result(path, i, dataset_image[i], dataset_countmap[i], dataset_count[i], dataset_hist[i], \
                        pred_countmap, pred_hist, Hist_wt, data_mean, num_bins)
        
        err_pix= np.abs(pred_countmap - dataset_countmap[i:i+bs]).mean(axis=(2,3))[0][0]
        
        pred_count = (pred_countmap/(ef)).sum(axis=(1,2,3))
        err_count = np.abs((dataset_count[i:i+bs]/ef).sum()-pred_count)[0]
        
        y_shape = pred_hist.shape[0]
        gt_hist = dataset_s[i:i+bs]
        p_prob = pred_hist/pred_hist.sum(axis = 1, keepdims=True) + (1e-6)
        p_prob1 = p_prob/p_prob.sum(axis =1, keepdims = True)
        t_prob = gt_hist/gt_hist.sum(axis = 1, keepdims=True) + (1e-6)
        t_prob1 = t_prob/t_prob.sum(axis = 1, keepdims =True)
        kl = (t_prob1*np.log((t_prob1)/(p_prob1)))
        err_kl = kl.sum()/y_shape
        
        if loss_name == 'w_L1':
            err_l1 = (Hist_wt*np.abs(pred_hist - gt_hist)).sum()/y_shape 
            err_l1_temp = np.abs(pred_hist - gt_hist).sum()/y_shape 
    
        elif loss_name == 'L1':
            err_l1 = np.abs(pred_hist - gt_hist).sum()/y_shape 
            loss_l1_temp = (Hist_wt*np.abs(pred_hist - gt_hist)).sum()/y_shape 
            
        
        err_total = Loss_wt[0]*err_kl + Loss_wt[1]*err_l1 + err_pix 
        test_loss_history = update_loss_history(test_loss_history, [err_total, err_count, err_pix, err_kl, err_l1, err_l1_temp, 0.0])
        gt_count.append((dataset_countmap[i:i+bs]/ef).sum())
        pred_count.append(pred_count)

    return test_loss_history, gt_count, pred_count 

def evaluate_histonet_dsn(dataset_image, dataset_countmap, dataset_count, dataset_hist, dataset_hist_dsn1, Hist_wt, Hist_wt_dsn1, Hist_wt_dsn2, test_op, data_mean, loss_list,\
                visualize = False, path = 'Results/', loss_name = 'w_L1',  Loss_wt= [0.5, 0.5,], num_bins = [2,4,8]):
    """
    Function: Evaluate HistoNet network performance on a dataset (For validation and Testing)
    
    Input:
    dataset_image : Input images
    dataset_countmap: Gt Redundant countmaps
    dataset_count: Gt object counts
    dataset_hist: Gt object size distribution histograms
    dataset_hist_dsn1: Gt object size distribution histograms dsn1
    dataset_hist_dsn2: Gt object size distribution histograms dsn2
    Hist_wt: Weights for L1 loss between predicted and target size histogram
    Hist_wt_dsn1: Weights for L1 loss between predicted and target size histogram dsn1
    Hist_wt_dsn2: Weights for L1 loss between predicted and target size histogram dsn2
    test_op: Operation for extracting output from network for a given input during test and validation(deterministic)
    data_mean: Mean image of training dataset 
    loss_list: list of different losses (only for initializing the test loss history list)
    visualize: Boolean (to visualize the test results)
    path : path of the result folder
    loss_name: Name of the L1 loss for histogram (Weighted L1 loss - w_L1 or L1 loss - L1)
    Loss_wt: Weights for KL divergence loss and L1 loss for histogram
    num_bins: list of number of bins of histogram [early layer output hist size, middle layer output hist size, final output hist size]
    
    Output:
    test_loss_history: list of lists of different losses [loss_total, loss_count, loss_pix, loss_kl, loss_l1, loss_l1_temp, loss_reg (0 for the test)]
    gt_count: list of object count in the test samples
    pred_count: list of predicted count in the test samples
    """
    
    test_loss_history = [[]] * len(loss_list)
    gt_count = []
    pred_count = []
    batchsize = 1

    for i in range(0,len(dataset_image), batchsize):
        pred_countmap, pred_hist, pred_hist_dsn1, pred_hist_dsn2 = test_op(dataset_image, range(i,i+batchsize))

        if visualize:
            visualize_HistoNet_DSN_result(path, i, dataset_image[i], dataset_countmap[i], dataset_count[i], dataset_hist[i], dataset_hist_dsn1[i], dataset_hist_dsn2[i] \
                        pred_countmap, pred_hist, pred_hist_dsn1, pred_hist_dsn2, Hist_wt, Hist_wt_dsn1, Hist_wt_dsn2, data_mean, num_bins)
        
        err_pix= np.abs(pred_countmap - dataset_countmap[i:i+bs]).mean(axis=(2,3))[0][0]
        
        pred_count = (pred_countmap/(ef)).sum(axis=(1,2,3))
        err_count = np.abs((dataset_count[i:i+bs]/ef).sum()-pred_count)[0]
        
        y_shape = pred_hist.shape[0]
        gt_hist = dataset_hist[i:i+bs]
        p_prob = pred_hist/pred_hist.sum(axis = 1, keepdims=True) + (1e-6)
        p_prob1 = p_prob/p_prob.sum(axis =1, keepdims = True)
        t_prob = gt_hist/gt_hist.sum(axis = 1, keepdims=True) + (1e-6)
        t_prob1 = t_prob/t_prob.sum(axis = 1, keepdims =True)
        kl = (t_prob1*np.log((t_prob1)/(p_prob1)))
        err_kl = kl.sum()/y_shape

        # KL Div loss - DSN1
        y_shape_dsn1 = pred_hist_dsn1.shape[0]
        gt_hist_dsn1 = dataset_hist_dsn1[i:i+bs]
        p_prob_dsn1 = pred_hist_dsn1/(pred_hist_dsn1.sum(axis = 1, keepdims=True) + 1e-6) + (1e-6)
        p_prob1_dsn1 = p_prob_dsn1/p_prob_dsn1.sum(axis =1, keepdims = True)
        t_prob_dsn1 = gt_hist_dsn1/gt_hist_dsn1.sum(axis = 1, keepdims=True) + (1e-6)
        t_prob1_dsn1 = t_prob_dsn1/t_prob_dsn1.sum(axis = 1, keepdims =True)
        kl_dsn1 = (t_prob1_dsn1*np.log((t_prob1_dsn1)/(p_prob1_dsn1)))
        err_kl_dsn1 = kl_dsn1.sum()/y_shape_dsn1

        # KL Div loss - DSN2
        y_shape_dsn2 = pred_hist_dsn2.shape[0]
        gt_hist_dsn2 = dataset_hist_dsn2[i:i+bs]
        p_prob_dsn2 = pred_hist_dsn2/(pred_hist_dsn2.sum(axis = 1, keepdims=True) + 1e-6) + (1e-6)
        p_prob1_dsn2 = p_prob_dsn2/p_prob_dsn2.sum(axis =1, keepdims = True)
        t_prob_dsn2 = gt_hist_dsn2/gt_hist_dsn2.sum(axis = 1, keepdims=True) + (1e-6)
        t_prob1_dsn2 = t_prob_dsn2/t_prob_dsn2.sum(axis = 1, keepdims =True)
        kl_dsn2 = (t_prob1_dsn2*np.log((t_prob1_dsn2)/(p_prob1_dsn2)))
        err_kl_dsn2 = kl_dsn2.sum()/y_shape_dsn2
        
        if loss_name == 'w_L1':
            err_l1 = (Hist_wt*np.abs(pred_hist - gt_hist)).sum()/y_shape 
            err_l1_temp = np.abs(pred_hist - gt_hist).sum()/y_shape 
    
            err_l1_dsn1 = (Hist_wt_dsn1*np.abs(pred_hist_dsn1 - gt_hist_dsn1)).sum()/y_shape_dsn1 
            err_l1_temp_dsn1 = np.abs(pred_hist_dsn1 - gt_hist_dsn1).sum()/y_shape_dsn1 
    
            err_l1_dsn2 = (Hist_wt_dsn2*np.abs(pred_hist_dsn2 - gt_hist_dsn2)).sum()/y_shape_dsn2 
            err_l1_temp_dsn2 = np.abs(pred_hist_dsn2 - gt_hist_dsn2).sum()/y_shape_dsn2 
    
        elif loss_name == 'L1':
            err_l1 = np.abs(pred_hist - gt_hist).sum()/y_shape 
            loss_l1_temp = (Hist_wt*np.abs(pred_hist - gt_hist)).sum()/y_shape 
            
            err_l1_dsn1 = np.abs(pred_hist_dsn1 - gt_hist_dsn1).sum()/y_shape_dsn1 
            loss_l1_temp_dsn1 = (Hist_wt_dsn1*np.abs(pred_hist_dsn1 - gt_hist_dsn1)).sum()/y_shape_dsn1 
            
            err_l1_dsn2 = np.abs(pred_hist_dsn2 - gt_hist_dsn2).sum()/y_shape_dsn2 
            loss_l1_temp_dsn2 = (Hist_wt_dsn2*np.abs(pred_hist_dsn2 - gt_hist_dsn2)).sum()/y_shape_dsn2 
            
        
        err_total = Loss_wt[0]*(err_kl + err_kl_dsn1 + err_kl_dsn2) + Loss_wt[1]*(err_l1 + err_l1_dsn1 + err_l1_dsn2) + err_pix 
        test_loss_history = update_loss_history(test_loss_history, [err_total, err_count, err_pix, err_kl, err_kl_dsn1, err_kl_dsn2,\
            err_l1, err_l1_dsn1, err_l1_dsn2, err_l1_temp, err_l1_temp_dsn1, err_l1_temp_dsn2, 0.0])
        gt_count.append((dataset_countmap[i:i+bs]/ef).sum())
        pred_count.append(pred_count)

    return test_loss_history, gt_count, pred_count 

def save_network(net, file_name, directory):
    """
    Function: save network weights
    
    Input:
    net : network 
    file_name: name of the model file
    directory: Logging directory for saving model weights 
    """
    np.savez( directory +'/' + file_name, *lasagne.layers.get_all_param_values(net))

def load_network(net, model_path):
    """
    Function: load network weights
    
    Input:
    net : network 
    model_path: path of the model file
    """ 
    with np.load(model_path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net, param_values)

def update_loss_history(losses_lists, new_values):

    losses_lists = [losses_lists[i].append(new_values[i]) for i in range(len(losses_lists))]

    return losses_lists

def trainer_histonet(net_count, net_hist, dataset_path, loss_list, placeholder_list, epochs, lr_value , \
                    lr_decay, batch_size, weight_decay_value, root_result, Experiment_name, \
                    train_op, test_op, loss_name = 'w_L1',  Loss_wt = [0.5, 0.5], num_bins = 8, print_every=50):
    """
    Function: Training and evaluating HistoNet network

    Input:
    net_count : count map output from network
    net_hist: histogram vector output from network
    dataset_path: path to the directory containing train, val and test dataset pickle files
    loss_list: list of different losses for count and size histogram prediction
    placeholder_list: list of placeholders for input and target 
    epochs: number of epochs
    lr_value: Learning rate
    lr_decay: decay rate of Learning rate (0 - 1.0)
    batch_size: batch_size for training 
    weight_decay_value: L2 regularization strength
    root_result: directory to training plots, test results 
    Experiment_name: Name of the experiment
    train_op: Operation for extracting output from network for a given input during training(non deterministic)
    test_op: Operation for extracting output from network for a given input during test and validation(deterministic)
    loss_name: Name of the L1 loss for histogram (Weighted L1 loss - w_L1 or L1 loss - L1)
    Loss_wt: Weights for KL divergence loss and L1 loss for histogram
    num_bins: number of bins of histogram
    print_every: number of iteration after which to show the progress of training losses
    
    """    
    
    # Import Dataset
    train_set, val_set, test_set, data_mean = import_data(dataset_path, num_bins)
    np_train_dataset_x, np_train_dataset_y, np_train_dataset_c,  np_train_dataset_s = train_set[0], train_set[1], train_set[2], train_set[3]
    np_val_dataset_x, np_val_dataset_y, np_val_dataset_c,  np_val_dataset_s = val_set[0], val_set[1], val_set[2], val_set[3]
    np_test_dataset_x, np_test_dataset_y, np_test_dataset_c,  np_test_dataset_s = test_set[0], test_set[1], test_set[2], test_set[3]
    print("Imported Data !!....")

    # Unpack losses and placeholders
    lr = theano.shared(np.array(0.0, dtype=theano.config.floatX))
    loss_total, loss_count, loss_pix, loss_kl, loss_l1, loss_l1_temp, loss_reg = loss_list 
    input_var, input_var_ex, target_var, target_var_hist, weight_var_hist = placeholder_list  
    
    # Calculate weights for weighted L1 histogram loss
    Bins_var = np.linspace(0,200, num_bins + 1)
    center_bin_hist = (Bins_var[:-1] + Bins_var[1:])/2
    Hist_wt = center_bin_hist/center_bin_hist.sum()
    Hist_wt = np.tile(Hist_wt, (batch_size,1)).astype(dtype = np.float32)
    
    # Preparing optimizer
    params = lasagne.layers.get_all_params([net_count, net_hist], trainable=True)
    updates = lasagne.updates.adam(loss_total, params, learning_rate=lr)
    train_fn = theano.function([input_var_ex], [loss_total, loss_count, loss_pix, loss_kl, loss_l1, loss_l1_temp, loss_reg], updates=updates,
                        givens={input_var:np_train_dataset_x, target_var:  np_train_dataset_y,  target_var_hist: np_train_dataset_s, weight_var_hist: Hist_wt})
    
    print("Training Function Compiled !!.....")

    lr.set_value(lr_value)
    best_valid_err = np.inf
    dataset_length = len(np_train_dataset_x)
    print("batch_size", batch_size)
    print("lr", lr.eval())
    print("datasetlength",dataset_length)

    training_plot_path = os.path.join(root_result, 'Training_plots')
    model_dir = os.path.join(root_result, Experiment_name)
    if not os.path.exists(root_result):
        os.mkdir(root_result)
        os.mkdir(training_plot_path)
        os.mkdir(model_dir)

    # Resetting Training and Validation loss per epoch history 
    train_loss_epoch_history = [[]]*len(loss_list)
    val_loss_epoch_history = [[]]*len(loss_list)
    
    for epoch in range(epochs):

        train_loss_history = [[]]* len(loss_list)    
        
        todo = range(datasetlength)
        
        for i in range(0,datasetlength, batch_size):
            ex = todo[i:i+batch_size]

            err_total, err_count, err_pix, err_kl, err_l1, err_l1_temp, err_reg = train_fn(ex)
            
            if i%print_every == 0 :
                print("Epoch :", epoch," | Iteration : ", i ,"| Total_Loss :",np.around(err_loss,2), \
                    "| Pix Loss :",np.around(err_pix,2), "| Count_Loss : ",np.around(err_count.mean(),2), \
                    "| kl_Loss:", np.around(err_kl,2), "| l1_Loss:", np.around(err_l1,2), "| l1_Loss_temp:", np.around(err_l1_temp,2), \
                    "| reg_Loss:", np.around(err_reg,2), "| Learning_rate:", np.around(lr.get_value(),5))
    
            # Updating Loss history
            train_loss_history = update_loss_history(train_loss_history, [err_total, err_count, err_pix, err_kl, err_l1, err_l1_temp, err_reg])
            
        # Learning rate decay
        lr.set_value(lasagne.utils.floatX(lr.get_value() * lr_decay))

        val_loss_history, _, _ = evaluate_histonet(np_val_dataset_x, np_val_dataset_y, np_val_dataset_c, np_val_dataset_s, Hist_wt[0,:], test_op, \
                                                    data_mean, loss_list, loss_name = loss_name, Loss_wt = Loss_wt, num_bins= num_bins)
        
        # Updating Loss Epoch history
        train_loss_epoch_history = update_loss_history(train_loss_epoch_history, [np.mean(train_loss_history[i]) for i in range(len(train_loss_history))])
        val_loss_epoch_history = update_loss_history(val_loss_epoch_history,  [np.mean(val_loss_history[i]) for i in range(len(val_loss_history))])

        # plot results   
        plot_results(train_loss_epoch_history[0], val_loss_epoch_history[0], training_plot_path,'Total_loss',1)
        plot_results(train_loss_epoch_history[1], val_loss_epoch_history[1], training_plot_path,'Count_loss',2)
        plot_results(train_loss_epoch_history[2], val_loss_epoch_history[2], training_plot_path,'Pix_loss',3)
        plot_results(train_loss_epoch_history[3], val_loss_epoch_history[3], training_plot_path,'kl_loss',4)
        plot_results(train_loss_epoch_history[4], val_loss_epoch_history[4], training_plot_path,'l1_loss',5)
        plot_results(train_loss_epoch_history[5], val_loss_epoch_history[5], training_plot_path,'l1_loss_temp',6)
        plot_results(train_loss_epoch_history[6], val_loss_epoch_history[6], training_plot_path,'reg_loss',7)

        print("Epoch : ",epoch, "| Train_Total_Loss :", np.around(train_loss_epoch_history[0][-1],2), \
            "| Val_Total_Loss :", np.around(val_loss_epoch_history[0][-1],2), \
            "| Train_Count Loss:",np.around(train_loss_epoch_history[1][-1],2),\
            "| Val_Count Loss:",np.around(val_loss_epoch_history[1][-1],2),\
            "| Train_Pix_loss:",np.around(train_loss_epoch_history[2][-1],2),\
            "| Val_Pix_loss:",np.around(val_loss_epoch_history[2][-1],2),\
            "| Train KL_Loss:",np.around(train_loss_epoch_history[3][-1],2), \
            "| Val KL_Loss:",np.around(val_loss_epoch_history[3][-1],2), \
            "| Train L1_Loss:",np.around(train_loss_epoch_history[4][-1],2), \
            "| Val L1_Loss:",np.around(val_loss_epoch_history[4][-1],2)) 
        
        save_network([net, net_hist], 'model_' +str(epoch) + '.npz', model_dir)
        
        # saving best model
        if (val_loss_epoch_history[0][-1] < best_valid_err):
            best_valid_err = val_loss_epoch_history[0][-1]
            save_network([net, net_hist], 'model_best.npz', model_dir)
            
    Test_directory = root_result + '/Test_results/'
    if not os.path.exists(Test_directory):
        os.mkdir(Test_directory)

    # Loading best model
    load_network([net, net_hist], model_dir + '/model_best.npz')
    test_loss_history, _, _ = evaluate_histonet(np_test_dataset_x, np_test_dataset_y, np_test_dataset_c, np_test_dataset_s, Hist_wt[0,:], \
                                    test_op, data_mean, loss_list, visualize = True, path = Test_directory, loss_name = loss_name, Loss_wt = Loss_wt, num_bins= num_bins)

    # saving results for test dataset
    Error_file  = open(model_dir + '/Test_result_Summary.txt', "w")
    Error_file.write('\n Total_count_Mean abs Error :' + str(np.mean(test_loss_history[1])))
    Error_file.write('\n Total_kl_Mean abs Error :' + str(np.mean(test_loss_history[3])))
    Error_file.write('\n Total_l1_Mean abs Error :' + str(np.mean(test_loss_history[4])))
    Error_file.write('\n Total_l1_ temp Mean abs Error :' + str(np.mean(test_loss_history[5])))
    Error_file.close()

def trainer_histonet_dsn(net_count, net_hist, net_hist_dsn1, net_hist_dsn2, dataset_path, loss_list, placeholder_list, epochs, lr_value , \
                    lr_decay, batch_size, weight_decay_value, root_result, Experiment_name, \
                    train_op, test_op, loss_name = 'w_L1',  Loss_wt = [0.5, 0.5], num_bins = [2,4,8], print_every=50):
    """
    Function: Training and evaluating HistoNet DSN network

    Input:
    net_count : count map output from network
    net_hist: histogram vector output of size num_bins[2]from network
    net_hist_dsn1: histogram vector output of size num_bins[0] from early part of the network
    net_hist_dsn1: histogram vector output of size num_bins[1] from middle part of the network
    dataset_path: path to the directory containing train, val and test dataset pickle files
    loss_list: list of different losses for count and size histogram prediction
    placeholder_list: list of placeholders for input and target 
    epochs: number of epochs
    lr_value: Learning rate
    lr_decay: decay rate of Learning rate (0 - 1.0)
    batch_size: batch_size for training 
    weight_decay_value: L2 regularization strength
    root_result: directory to training plots, test results 
    Experiment_name: Name of the experiment
    train_op: Operation for extracting output from network for a given input during training(non deterministic)
    test_op: Operation for extracting output from network for a given input during test and validation(deterministic)
    loss_name: Name of the L1 loss for histogram (Weighted L1 loss - w_L1 or L1 loss - L1)
    Loss_wt: Weights for KL divergence loss and L1 loss for histogram
    num_bins: list of number of bins of histogram [early layer output hist size, middle layer output hist size, final output hist size]
    print_every: number of iteration after which to show the progress of training losses
    """    
    
    # Import Dataset
    train_set, val_set, test_set, data_mean = import_data_dsn(dataset_path, num_bins)
    np_train_dataset_x, np_train_dataset_y, np_train_dataset_c,  np_train_dataset_s_dsn1, np_train_dataset_s_dsn2, np_train_dataset_s = train_set[0], train_set[1], train_set[2], train_set[3], train_set[4], train_set[5]
    np_val_dataset_x, np_val_dataset_y, np_val_dataset_c,  np_val_dataset_s_dsn1, np_val_dataset_s_dsn2, np_val_dataset_s = val_set[0], val_set[1], val_set[2], val_set[3], val_set[4], val_set[5]
    np_test_dataset_x, np_test_dataset_y, np_test_dataset_c,  np_test_dataset_s_dsn1, np_test_dataset_s_dsn2, np_test_dataset_s = test_set[0], test_set[1], test_set[2], test_set[3], test_set[4], test_set[5]
    print("Imported Data !!....")

    # Unpack losses and placeholders
    lr = theano.shared(np.array(0.0, dtype=theano.config.floatX))
    loss_total, loss_count, loss_pix, loss_kl, loss_kl_dsn1, loss_kl_dsn2, loss_l1, loss_l1_dsn1, loss_l1_dsn2,\
    loss_l1_temp, loss_l1_temp_dsn1, loss_l1_temp_dsn2, loss_reg = loss_list 
    input_var, input_var_ex, target_var, target_var_hist, target_var_hist_dsn1, target_var_hist_dsn2, \
    weight_var_hist, weight_var_hist_dsn1, weight_var_hist_dsn2 = placeholder_list 
    
    # Calculate weights for weighted L1 histogram loss num_bins[0]
    Bins_var = np.linspace(0,200, num_bins[0] + 1)
    center_bin_hist = (Bins_var[:-1] + Bins_var[1:])/2
    Hist_wt = center_bin_hist/center_bin_hist.sum()
    Hist_wt_dsn1 = np.tile(Hist_wt, (batch_size,1)).astype(dtype = np.float32)
    
    # Calculate weights for weighted L1 histogram loss num_bins[1]
    Bins_var = np.linspace(0,200, num_bins[1] + 1)
    center_bin_hist = (Bins_var[:-1] + Bins_var[1:])/2
    Hist_wt = center_bin_hist/center_bin_hist.sum()
    Hist_wt_dsn2 = np.tile(Hist_wt, (batch_size,1)).astype(dtype = np.float32)
    
    # Calculate weights for weighted L1 histogram loss num_bins[2]
    Bins_var = np.linspace(0,200, num_bins[2] + 1)
    center_bin_hist = (Bins_var[:-1] + Bins_var[1:])/2
    Hist_wt = center_bin_hist/center_bin_hist.sum()
    Hist_wt = np.tile(Hist_wt, (batch_size,1)).astype(dtype = np.float32)
    
    # Preparing optimizer
    params = lasagne.layers.get_all_params([net_count, net_hist, net_hist_dsn1, net_hist_dsn2], trainable=True)
    updates = lasagne.updates.adam(loss_total, params, learning_rate=lr)
    train_fn = theano.function([input_var_ex], [loss_total, loss_count, loss_pix, loss_kl, loss_kl_dsn1, loss_kl_dsn2, \
                        loss_l1, loss_l1_dsn1, loss_l1_dsn2, loss_l1_temp, loss_l1_temp_dsn1, loss_l1_temp_dsn2, loss_reg], updates=updates,
                        givens={input_var:np_train_dataset_x, target_var:  np_train_dataset_y,  target_var_hist: np_train_dataset_s, weight_var_hist: Hist_wt, \
                            target_var_hist_dsn1: np_train_dataset_s_dsn1, weight_var_hist_dsn1: Hist_wt_dsn1,\
                            target_var_hist_dsn2: np_train_dataset_s_dsn2, weight_var_hist_dsn2: Hist_wt_dsn2 })
    
    print("Training Function Compiled !!.....")

    lr.set_value(lr_value)
    best_valid_err = np.inf
    dataset_length = len(np_train_dataset_x)
    print("batch_size", batch_size)
    print("lr", lr.eval())
    print("datasetlength",dataset_length)

    training_plot_path = os.path.join(root_result, 'Training_plots')
    model_dir = os.path.join(root_result, Experiment_name)
    if not os.path.exists(root_result):
        os.mkdir(root_result)
        os.mkdir(training_plot_path)
        os.mkdir(model_dir)

    # Resetting Training and Validation loss per epoch history 
    train_loss_epoch_history = [[]]*len(loss_list)
    val_loss_epoch_history = [[]]*len(loss_list)
    
    for epoch in range(epochs):

        train_loss_history = [[]]* len(loss_list)    
        
        todo = range(datasetlength)
        
        for i in range(0,datasetlength, batch_size):
            ex = todo[i:i+batch_size]

            err_total, err_count, err_pix, err_kl, err_kl_dsn1, err_kl_dsn2, err_l1, err_l1_dsn1, err_l1_dsn2,\
                 err_l1_temp, err_l1_temp_dsn1, err_l1_temp_dsn2, err_reg = train_fn(ex)
            
            if i%print_every == 0 :
                print("Epoch :", epoch," | Iteration : ", i ,"| Total_Loss :",np.around(err_loss,2), \
                    "| Pix Loss :",np.around(err_pix,2), "| Count_Loss : ",np.around(err_count.mean(),2), \
                    "| kl_Loss:", np.around(err_kl,2), "| l1_Loss:", np.around(err_l1,2), "| l1_Loss_temp:", np.around(err_l1_temp,2), \
                    "| kl_Loss dsn1:", np.around(err_kl_dsn1,2), "| l1_Loss dsn1:", np.around(err_l1_dsn1,2), "| l1_Loss_temp dsn1:", np.around(err_l1_temp_dsn1,2), \
                    "| kl_Loss dsn2:", np.around(err_kl_dsn2,2), "| l1_Loss dsn2:", np.around(err_l1_dsn2,2), "| l1_Loss_temp dsn2:", np.around(err_l1_temp_dsn2,2), \
                    "| reg_Loss:", np.around(err_reg,2), "| Learning_rate:", np.around(lr.get_value(),5))
    
            # Updating Loss history
            train_loss_history = update_loss_history(train_loss_history, [err_total, err_count, err_pix, err_kl, err_kl_dsn1, err_kl_dsn2,\
                             err_l1, err_l1_dsn1, err_l1_dsn2, err_l1_temp, err_l1_temp_dsn1, err_l1_temp_dsn2, err_reg])
            
        # Learning rate decay
        lr.set_value(lasagne.utils.floatX(lr.get_value() * lr_decay))

        val_loss_history, _, _ = evaluate_histonet_dsn(np_val_dataset_x, np_val_dataset_y, np_val_dataset_c, np_val_dataset_s, np_val_dataset_s_dsn1, np_test_dataset_s_dsn2, Hist_wt[0,:], Hist_wt_dsn1[0,:], Hist_wt_dsn2[0,:], test_op, \
                                                    data_mean, loss_name = loss_name, Loss_wt = Loss_wt, num_bins= num_bins)
        
        # Updating Loss Epoch history
        train_loss_epoch_history = update_loss_history(train_loss_epoch_history, [np.mean(train_loss_history[i]) for i in range(len(train_loss_history))])
        val_loss_epoch_history = update_loss_history(val_loss_epoch_history,  [np.mean(val_loss_history[i]) for i in range(len(val_loss_history))])

        # plot results   
        plot_results(train_loss_epoch_history[0], val_loss_epoch_history[0], training_plot_path,'Total_loss',1)
        plot_results(train_loss_epoch_history[1], val_loss_epoch_history[1], training_plot_path,'Count_loss',2)
        plot_results(train_loss_epoch_history[2], val_loss_epoch_history[2], training_plot_path,'Pix_loss',3)
        plot_results(train_loss_epoch_history[3], val_loss_epoch_history[3], training_plot_path,'kl_loss',4)
        plot_results(train_loss_epoch_history[4], val_loss_epoch_history[4], training_plot_path,'kl_loss dsn1',5)
        plot_results(train_loss_epoch_history[5], val_loss_epoch_history[5], training_plot_path,'kl_loss dsn2',6)
        plot_results(train_loss_epoch_history[6], val_loss_epoch_history[6], training_plot_path,'l1_loss',7)
        plot_results(train_loss_epoch_history[7], val_loss_epoch_history[7], training_plot_path,'l1_loss dsn1',8)
        plot_results(train_loss_epoch_history[8], val_loss_epoch_history[8], training_plot_path,'l1_loss dsn2',9)
        plot_results(train_loss_epoch_history[9], val_loss_epoch_history[9], training_plot_path,'l1_loss_temp',10)
        plot_results(train_loss_epoch_history[10], val_loss_epoch_history[10], training_plot_path,'l1_loss_temp dsn1',11)
        plot_results(train_loss_epoch_history[11], val_loss_epoch_history[11], training_plot_path,'l1_loss_temp dsn2',12)
        plot_results(train_loss_epoch_history[12], val_loss_epoch_history[12], training_plot_path,'reg_loss',13)

        print("Epoch : ",epoch, "| Train_Total_Loss :", np.around(train_loss_epoch_history[0][-1],2), \
            "| Val_Total_Loss :", np.around(val_loss_epoch_history[0][-1],2), \
            "| Train_Count Loss:",np.around(train_loss_epoch_history[1][-1],2),\
            "| Val_Count Loss:",np.around(val_loss_epoch_history[1][-1],2),\
            "| Train_Pix_loss:",np.around(train_loss_epoch_history[2][-1],2),\
            "| Val_Pix_loss:",np.around(val_loss_epoch_history[2][-1],2),\
            "| Train KL_Loss:",np.around(train_loss_epoch_history[3][-1],2), \
            "| Val KL_Loss:",np.around(val_loss_epoch_history[3][-1],2), \
            "| Train L1_Loss:",np.around(train_loss_epoch_history[6][-1],2), \
            "| Val L1_Loss:",np.around(val_loss_epoch_history[6][-1],2), \
            "| Train KL dsn1_Loss:",np.around(train_loss_epoch_history[4][-1],2), \
            "| Val KL_dsn1 Loss:",np.around(val_loss_epoch_history[4][-1],2), \
            "| Train L1 dsn1_Loss:",np.around(train_loss_epoch_history[7][-1],2), \
            "| Val L1_dsn1 Loss:",np.around(val_loss_epoch_history[7][-1],2)) 
            "| Train KL_dsn2_Loss:",np.around(train_loss_epoch_history[5][-1],2), \
            "| Val KL_dsn2_Loss:",np.around(val_loss_epoch_history[5][-1],2), \
            "| Train L1_dsn2_Loss:",np.around(train_loss_epoch_history[8][-1],2), \
            "| Val L1_dsn2_Loss:",np.around(val_loss_epoch_history[8][-1],2))) 
        
        save_network([net, net_hist, net_hist_dsn1, net_hist_dsn2], 'model_' +str(epoch) + '.npz', model_dir)
        
        # saving best model
        if (val_loss_epoch_history[0][-1] < best_valid_err):
            best_valid_err = val_loss_epoch_history[0][-1]
            save_network([net, net_hist, net_hist_dsn1, net_hist_dsn2], 'model_best.npz', model_dir)
            
    Test_directory = root_result + '/Test_results/'
    if not os.path.exists(Test_directory):
        os.mkdir(Test_directory)

    # Loading best model
    load_network([net, net_hist, net_hist_dsn1, net_hist_dsn2], model_dir + '/model_best.npz')
    test_loss_history, _, _ = evaluate_histonet(np_test_dataset_x, np_test_dataset_y, np_test_dataset_c, np_test_dataset_s, np_test_dataset_s_dsn1, np_test_dataset_s_dsn2,\
                                    Hist_wt[0,:], Hist_wt_dsn1[0,:], Hist_wt_dsn2[0,:], test_op, data_mean, visualize = True, path = Test_directory, loss_name = loss_name, Loss_wt = Loss_wt, num_bins= num_bins)

    # saving results for test dataset
    Error_file  = open(model_dir + '/Test_result_Summary.txt', "w")
    Error_file.write('\n Total_count_Mean abs Error :' + str(np.mean(test_loss_history[1])))
    Error_file.write('\n Total_kl_Mean abs Error :' + str(np.mean(test_loss_history[3])))
    Error_file.write('\n Total_l1_Mean abs Error :' + str(np.mean(test_loss_history[6])))
    Error_file.write('\n Total_l1_ temp Mean abs Error :' + str(np.mean(test_loss_history[9])))
    Error_file.close()

def loss_func_histonet(net_count, net_hist, input_var, input_var_ex, reg, loss_name = 'w_L1',  Loss_wt = [0.5, 0.5]):
    """
    Function: Defining Loss functions for training HistoNet network
    
    Input:
    net_count : count map output from network
    net_hist: histogram vector output from network
    input_var: Place holder for input to the network
    input_var_ex: Place holder for index for data (input, target)
    reg: Regularization strength
    loss_name: Name of the L1 loss for histogram (Weighted L1 loss - w_L1 or L1 loss - L1)
    Loss_wt: Weights for KL divergence loss and L1 loss for histogram

    Output:
    loss_list: list of different losses for count and size histogram prediction
    placeholder_list: list of placeholders for input and target 
    train_op: Operation for extracting output from network for a given input during training(non deterministic)
    test_op: Operation for extracting output from network for a given input during test and validation(deterministic)
    """
    
    # Training forward pass
    prediction_count_map, prediction_hist = lasagne.layers.get_output([net_count, net_hist], deterministic=False)
    prediction_count = (prediction_count_map/ef).sum(axis=(2,3))
    train_op = theano.function([input_var, input_var_ex], [prediction_count, prediction_hist])

    # Val/Test forward pass
    prediction_count_map_t, prediction_hist_t = lasagne.layers.get_output([net_count, net_hist], deterministic=True)
    prediction_count_t = (prediction_count_map_t/ef).sum(axis=(2,3))
    test_op = theano.function([input_var, input_var_ex], [prediction_count_t, prediction_hist_t])

    # Placeholders for target and weights for histogram weighted L1 loss
    target_var = T.tensor4('target')
    target_var_hist = T.matrix('target_hist')
    weight_var_hist = T.matrix('weight_hist')
    
    #Mean Absolute Error is computed between each count of the count map, pixel wise
    l1_loss = T.abs_(prediction_count_map - target_var[input_var_ex])
    loss_pix = l1_loss.mean()
    
    #Mean Absolute Error is computed for the overall count loss 
    loss_count = T.abs_(prediction_count - (target_var[input_var_ex]/ef).sum(axis=(2,3))).mean()
    
    #KL DIV LOSS between probability distribution of target and predicted histogram to capture shape of histogram
    y_shape = prediction_hist.shape[0]
    target_hist = target_var_hist[input_var_ex]
    p_prob_1 = prediction_hist/prediction_hist.sum(axis = 1, keepdims=True) + (1e-6)
    p_prob = p_prob_1/p_prob_1.sum(axis=1, keepdims = True)
    t_prob_1 = target_hist/target_hist.sum(axis = 1, keepdims=True) + (1e-6)
    t_prob = t_prob_1/t_prob_1.sum(axis = 1, keepdims = True)
    kl= (t_prob*T.log((t_prob)/(p_prob)))
    loss_kl = kl.sum()/y_shape

    # weighted L1  or L1 loss between predicted and target histogram to capture scale of size histogram
    if loss_name == 'L1':
        loss_l1 = (T.abs_(prediction_hist - target_hist)).sum()/y_shape 
        loss_l1_temp = (weight_var_hist*T.abs_(prediction_hist - target_hist)).sum()/y_shape
    elif loss_name == 'w_L1':
        loss_l1 = (weight_var_hist*T.abs_(prediction_hist - target_hist)).sum()/y_shape
        loss_l1_temp = (T.abs_(prediction_hist - target_hist)).sum()/y_shape 

    # Regularization loss
    loss_reg = 0.5*reg*regularize_network_params([net_count, net_hist], l2)

    # Total Loss
    loss_total = loss_pix + Loss_wt[0]*loss_kl + Loss_wt[1]*loss_l1 + loss_reg
    
    loss_list = [loss_total, loss_count, loss_pix, loss_kl, loss_l1, loss_l1_temp, loss_reg] 
    placeholder_list = [input_var, input_var_ex, target_var, target_var_hist, weight_var_hist] 
    return loss_list, placeholder_list, train_op, test_op
    
def loss_func_histonet_dsn(net_count, net_hist, net_hist_dsn1, net_hist_dsn2, input_var, input_var_ex, reg, loss_name = 'w_L1', Loss_wt = [0.5, 0.5]):
    """
    Function: Defining Loss functions for training HistoNet network
    
    Input:
    net_count : count map output from network
    net_hist: histogram vector output from network
    net_hist_dsn1: histogram vector output from early part of the network
    net_hist_dsn2: histogram vector output from middle part of the network 
    input_var: Place holder for input to the network
    input_var_ex: Place holder for index for data (input, target)
    reg: Regularization strength
    loss_name: Name of the L1 loss for histogram (Weighted L1 loss - w_L1 or L1 loss - L1)
    Loss_wt: Weights for KL divergence loss and L1 loss for histogram

    Output:
    loss_list: list of different losses for count and size histogram prediction
    placeholder_list: list of placeholders for input and target 
    train_op: Operation for extracting output from network for a given input during training(non deterministic)
    test_op: Operation for extracting output from network for a given input during test and validation(deterministic)
    """
    
    # Training forward pass
    prediction_count_map, prediction_hist, prediction_hist_dsn1, prediction_hist_dsn2 = lasagne.layers.get_output([net_count, net_hist, net_hist_dsn1, net_hist_dsn2], deterministic=False)
    prediction_count = (prediction_count_map/ef).sum(axis=(2,3))
    train_op = theano.function([input_var, input_var_ex], [prediction_count, prediction_hist, prediction_hist_dsn1, prediction_hist_dsn2])

    # Val/Test forward pass
    prediction_count_map_t, prediction_hist_t, prediction_hist_dsn1_t, prediction_hist_dsn2_t = lasagne.layers.get_output([net_count, net_hist, net_hist_dsn1, net_hist_dsn2], deterministic=True)
    prediction_count_t = (prediction_count_map_t/ef).sum(axis=(2,3))
    test_op = theano.function([input_var, input_var_ex], [prediction_count_t, prediction_hist_t, prediction_hist_dsn1_t, prediction_hist_dsn2_t])

    # Placeholders for target and weights for histogram weighted L1 loss
    target_var = T.tensor4('target')
    target_var_hist = T.matrix('target_hist')
    weight_var_hist = T.matrix('weight_hist')
    target_var_hist_dsn1 = T.matrix('target_hist_dsn1')
    weight_var_hist_dsn1 = T.matrix('weight_hist_dsn1')
    target_var_hist_dsn2 = T.matrix('target_hist_dsn2')
    weight_var_hist_dsn2 = T.matrix('weight_hist_dsn2')
    
    #Mean Absolute Error is computed between each count of the count map, pixel wise
    l1_loss = T.abs_(prediction_count_map - target_var[input_var_ex])
    loss_pix = l1_loss.mean()
    
    #Mean Absolute Error is computed for the overall count loss 
    loss_count = T.abs_(prediction_count - (target_var[input_var_ex]/ef).sum(axis=(2,3))).mean()
    
    #KL DIV LOSS between probability distribution of target and predicted histogram (main objective histogram) to capture shape of histogram
    y_shape = prediction_hist.shape[0]
    target_hist = target_var_hist[input_var_ex]
    p_prob_1 = prediction_hist/(prediction_hist.sum(axis = 1, keepdims=True) + 1e-6) + (1e-6)
    p_prob = p_prob_1/p_prob_1.sum(axis=1, keepdims = True)
    t_prob_1 = target_hist/target_hist.sum(axis = 1, keepdims=True) + (1e-6)
    t_prob = t_prob_1/t_prob_1.sum(axis = 1, keepdims = True)
    kl= (t_prob*T.log((t_prob)/(p_prob)))
    loss_kl = kl.sum()/y_shape

    #KL DIV LOSS - DSN1 between probability distribution of target and predicted histogram DSN1 to capture shape of histogram
    target_hist_dsn1 = target_var_hist_dsn1[input_var_ex]
    p_prob_1_dsn1 = prediction_hist_dsn1/(prediction_hist_dsn1.sum(axis = 1, keepdims=True) + 1e-6) + (1e-6)
    p_prob_dsn1 = p_prob_1_dsn1/p_prob_1_dsn1.sum(axis=1, keepdims = True)
    t_prob_1_dsn1 = target_hist_dsn1/target_hist_dsn1.sum(axis = 1, keepdims=True) + (1e-6)
    t_prob_dsn1 = t_prob_1_dsn1/t_prob_1_dsn1.sum(axis = 1, keepdims = True)
    kl_dsn1= (t_prob_dsn1*T.log((t_prob_dsn1)/(p_prob_dsn1)))
    loss_kl_dsn1 = kl_dsn1.sum()/y_shape

    #KL DIV LOSS - DSN2between probability distribution of target and predicted histogram DSN2 to capture shape of histogram
    target_hist_dsn2 = target_var_hist_dsn2[input_var_ex]
    p_prob_1_dsn2 = prediction_hist_dsn2/(prediction_hist_dsn2.sum(axis = 1, keepdims=True) + 1e-6) + (1e-6)
    p_prob_dsn2 = p_prob_1_dsn2/p_prob_1_dsn2.sum(axis=1, keepdims = True)
    t_prob_1_dsn2 = target_hist_dsn2/target_hist_dsn2.sum(axis = 1, keepdims=True) + (1e-6)
    t_prob_dsn2 = t_prob_1_dsn2/t_prob_1_dsn2.sum(axis = 1, keepdims = True)
    kl_dsn2= (t_prob_dsn2*T.log((t_prob_dsn2)/(p_prob_dsn2)))
    loss_kl_dsn2 = kl_dsn2.sum()/y_shape

    # weighted L1  or L1 loss between predicted and target histogram to capture scale of size histogram
    if loss_name == 'L1':
        loss_l1 = (T.abs_(prediction_hist - target_hist)).sum()/y_shape 
        loss_l1_temp = (weight_var_hist*T.abs_(prediction_hist - target_hist)).sum()/y_shape
        loss_l1_dsn1 = (T.abs_(prediction_hist_dsn1 - target_hist_dsn1)).sum()/y_shape 
        loss_l1_temp_dsn1 = (weight_var_hist_dsn1*T.abs_(prediction_hist_dsn1 - target_hist_dsn1)).sum()/y_shape
        loss_l1_dsn2 = (T.abs_(prediction_hist_dsn2 - target_hist_dsn2)).sum()/y_shape 
        loss_l1_temp_dsn2 = (weight_var_hist_dsn2*T.abs_(prediction_hist_dsn2 - target_hist_dsn2)).sum()/y_shape
    
    elif loss_name == 'w_L1':
        loss_l1 = (weight_var_hist*T.abs_(prediction_hist - target_hist)).sum()/y_shape
        loss_l1_temp = (T.abs_(prediction_hist - target_hist)).sum()/y_shape 
        loss_l1_dsn1 = (weight_var_hist_dsn1*T.abs_(prediction_hist_dsn1 - target_hist_dsn1)).sum()/y_shape
        loss_l1_temp_dsn1 = (T.abs_(prediction_hist_dsn1 - target_hist_dsn1)).sum()/y_shape 
        loss_l1_dsn2 = (weight_var_hist_dsn2*T.abs_(prediction_hist_dsn2 - target_hist_dsn2)).sum()/y_shape
        loss_l1_temp_dsn2 = (T.abs_(prediction_hist_dsn2 - target_hist_dsn2)).sum()/y_shape 
    
    # Regularization loss
    loss_reg = 0.5*reg*regularize_network_params([net_count, net_hist, net_hist_dsn1, net_hist_dsn2], l2)

    # Total Loss
    loss_total = loss_pix + Loss_wt[0]*(loss_kl + loss_kl_dsn1 + loss_kl_dsn2) + loss_wt[1]*(loss_l1 + loss_l1_dsn1 + loss_l1_dsn2) + loss_reg
    
    loss_list = [loss_total, loss_count, loss_pix, loss_kl, loss_kl_dsn1, loss_kl_dsn2, loss_l1, loss_l1_dsn1, loss_l1_dsn2,\
                loss_l1_temp, loss_l1_temp_dsn1, loss_l1_temp_dsn2, loss_reg] 
    placeholder_list = [input_var, input_var_ex, target_var, target_var_hist, target_var_hist_dsn1, target_var_hist_dsn2, \
                weight_var_hist, weight_var_hist_dsn1, weight_var_hist_dsn2]  
    return loss_list, placeholder_list, train_op, test_op

def plot_results( train_loss,  val_loss, Log_dir, plotname, i ):
    """
    Function: Plot results training and validation
    
    Input:
    train_loss : List of train loss per epoch
    val_loss: List of val loss per epoch
    Log_dir: Logging directory for saving training plots
    plotname: name of the loss plot 
    i: random integer for different plot
    """

    if not os.path.exists(Log_dir):
        os.mkdir(Log_dir)    
    # Saving the train and val loss as .txt file
    Error_file  = open(Log_dir + '/' + plotname + '_Summary.txt', "w")
    Error_file.write('Train_loss: ' + str(train_loss))
    Error_file.write('\n val_loss: '+  str(val_loss))
    Error_file.close()

    # Plotting training curves
    plt.figure(i+10, figsize=(15, 10))
    plt.plot(range(len(train_loss)), train_loss, 'r', label = 'train')
    plt.plot(range(len(val_loss)), val_loss, 'g', label = 'val')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Train & Validation ' + plotname)
    plt.savefig(Log_dir + '/' + plotname + '.png')
    plt.close()

def visualize_HistoNet_result(path, i, image, gt_countmap, gt_count, gt_hist, pred_countmap, pred_hist, Hist_wt, data_mean, num_bins):
    """
    Function: Visualizing results of HistoNet network
    
    Input:
    path : path of the result folder
    i : image number
    gt_countmap: Ground truth redundant count map 
    gt_count: Ground truth object count  
    gt_hist: Ground truth object size distribution histogram
    pred_countmap: predicted redundant count map 
    pred_hist: predicted object size distribution histogram 
    Hist_wt: Weights for L1 loss between predicted and target size histogram
    data_mean: Mean image of training dataset 
    num_bins: number of the bins of histogram
    """

    fig = plt.Figure(figsize=(18, 9), dpi=160)
    gcf = plt.gcf()
    gcf.set_size_inches(18, 15)
    fig.set_canvas(gcf.canvas)

    pred_count = [(l.sum()/(ef)).astype(np.int) for l in pred_countmap]
    
    img = image.transpose((1,2,0))
    #De-Normalize
    img = img + data_mean
    ax2 = plt.subplot2grid((4,6), (0, 0), colspan=2)
    ax3 = plt.subplot2grid((4,6), (0, 2), colspan=5)
    ax4 = plt.subplot2grid((4,6), (1, 2), colspan=5)
    ax5 = plt.subplot2grid((4,6), (1, 0), rowspan=1)
    ax6 = plt.subplot2grid((4,6), (1, 1), rowspan=1)
    ax7 = plt.subplot2grid((4,6), (2, 0), colspan=5)
    ax8 = plt.subplot2grid((4,6), (3, 0), colspan=5)
    
    ax2.set_title("Input Image")
    ax2.imshow(img, interpolation='none', cmap='Greys_r')
    ax3.set_title("Target Countmap")
    ax3.imshow(np.concatenate((gt_countmap),axis=1), interpolation='none')
    ax4.set_title("Predicted Countmap")
    ax4.imshow(pred_countmap.squeeze(), interpolation='none')
    
    ax5.set_title("Target " + str(gt_count))
    ax5.set_ylim((0, np.max(gt_count)*2))
    ax5.set_xticks(np.arange(0, 1, 1.0))
    ax5.bar(range(1),gt_count, align='center')
    ax6.set_title("Pred " + str(pred_count))
    ax6.set_ylim((0, np.max(gt_count)*2))
    ax6.set_xticks(np.arange(0, 1, 1.0))
    ax6.bar(range(1),pred_count, align='center')

    Bins = np.linspace(0,200, num_bins + 1)
    ax7.set_title("Gt Histogram")
    ax7.hist(Bins[:-1], weights= gt_hist.T , bins = Bins)
    
    ax8.set_title("Pred Histogram")
    ax8.hist(Bins[:-1], weights= pred_hist.T , bins = Bins)
    
    if not os.path.exists(path + '/HistoNet_eval_result'): 
        os.mkdir(path + '/HistoNet_eval_result-cell')
    fig.savefig(path + '/HistoNet_eval_result/image-' + str(i) + '.png', bbox_inches='tight', pad_inches=0)

def visualize_HistoNet_DSN_result(path, i, image, gt_countmap, gt_count, gt_hist, gt_hist_dsn1, gt_hist_dsn2, pred_countmap, pred_hist, pred_hist_dsn1, pred_hist_dsn2,\
                             Hist_wt, Hist_wt_dsn1, Hist_wt_dsn2, data_mean, num_bins):
    """
    Function: Visualizing results of HistoNet network
    
    Input:
    path : path of the result folder
    i : image number
    gt_countmap: Ground truth redundant count map 
    gt_count: Ground truth object count  
    gt_hist: Ground truth object size distribution histogram dsn2
    gt_hist_dsn1: Ground truth object size distribution histogram
    gt_hist_dsn2: Ground truth object size distribution histogram dsn1
    pred_countmap: predicted redundant count map 
    pred_hist: predicted object size distribution histogram 
    pred_hist_dsn1: predicted object size distribution histogram dsn1
    pred_hist_dsn2: predicted object size distribution histogram dsn2
    Hist_wt: Weights for L1 loss between predicted and target size histogram
    Hist_wt_dsn1: Weights for L1 loss between predicted and target size histogram dsn1
    Hist_wt_dsn2: Weights for L1 loss between predicted and target size histogram dsn2
    data_mean: Mean image of training dataset 
    num_bins: list of number of bins of histogram [early layer output hist size, middle layer output hist size, final output hist size]
    """

    fig = plt.Figure(figsize=(18, 9), dpi=160)
    gcf = plt.gcf()
    gcf.set_size_inches(18, 15)
    fig.set_canvas(gcf.canvas)

    pred_count = [(l.sum()/(ef)).astype(np.int) for l in pred_countmap]
    
    img = image.transpose((1,2,0))
    #De-Normalize
    img = img + data_mean
    ax2 = plt.subplot2grid((6,6), (0, 0), colspan=2)
    ax3 = plt.subplot2grid((6,6), (0, 2), colspan=5)
    ax4 = plt.subplot2grid((6,6), (1, 2), colspan=5)
    ax5 = plt.subplot2grid((6,6), (1, 0), rowspan=1)
    ax6 = plt.subplot2grid((6,6), (1, 1), rowspan=1)
    ax7 = plt.subplot2grid((6,6), (2, 0), colspan=5)
    ax8 = plt.subplot2grid((6,6), (3, 0), colspan=5)
    ax9 = plt.subplot2grid((6,6), (4, 0), colspan=3)
    ax10 = plt.subplot2grid((6,6), (4, 3), colspan=5)
    ax11 = plt.subplot2grid((6,6), (5, 0), colspan=3)
    ax12 = plt.subplot2grid((6,6), (5, 3), colspan=5)
    
    ax2.set_title("Input Image")
    ax2.imshow(img, interpolation='none', cmap='Greys_r')
    ax3.set_title("Target Countmap")
    ax3.imshow(np.concatenate((gt_countmap),axis=1), interpolation='none')
    ax4.set_title("Predicted Countmap")
    ax4.imshow(pred_countmap.squeeze(), interpolation='none')
    
    ax5.set_title("Target " + str(gt_count))
    ax5.set_ylim((0, np.max(gt_count)*2))
    ax5.set_xticks(np.arange(0, 1, 1.0))
    ax5.bar(range(1),gt_count, align='center')
    ax6.set_title("Pred " + str(pred_count))
    ax6.set_ylim((0, np.max(gt_count)*2))
    ax6.set_xticks(np.arange(0, 1, 1.0))
    ax6.bar(range(1),pred_count, align='center')

    Bins = np.linspace(0,200, num_bins[2] + 1)
    ax7.set_title("Gt Histogram")
    ax7.hist(Bins[:-1], weights= gt_hist.T , bins = Bins)
    ax8.set_title("Pred Histogram")
    ax8.hist(Bins[:-1], weights= pred_hist.T , bins = Bins)

    Bins2 = np.linspace(0,200, num_bins[0] + 1))
    ax9.set_title("Gt Histogram dsn1")
    ax9.hist(Bins2[:-1], weights= gt_hist_dsn1.T , bins = Bins2)
    ax11.set_title("Pred Histogram dsn1")
    ax11.hist(Bins2[:-1], weights= pred_hist_dsn1.T , bins = Bins2)
    
    Bins4 = np.linspace(0,200,num_bins[1] + 1))
    ax10.set_title("Gt Histogram dsn2")
    ax10.hist(Bins4[:-1], weights= gt_hist_dsn2.T , bins = Bins4)
    ax12.set_title("Pred Histogram dsn2")
    ax12.hist(Bins4[:-1], weights= pred_hist_dsn2.T , bins = Bins4)
    
    if not os.path.exists(path + '/HistoNet_DSN_eval_result'): 
        os.mkdir(path + '/HistoNet_DSN_eval_result-cell')
    fig.savefig(path + '/HistoNet_DSN_eval_result/image-' + str(i) + '.png', bbox_inches='tight', pad_inches=0)

from lasagne.layers.normalization import BatchNormLayer
from lasagne.layers import InputLayer, ConcatLayer, Conv2DLayer
from config import *

import lasagne

from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax
from lasagne.regularization import regularize_network_params, l2
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import DropoutLayer

import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.set_cmap('jet')

import sys,os,time,random
import numpy as np

import theano
import theano.tensor as T 
import lasagne

seed = 0 
random.seed(seed)
np.random.seed(seed)
lasagne.random.set_rng(np.random.RandomState(seed))

def test_perf(dataset_x, dataset_y, dataset_c, dataset_s, center_var, classify, data_mean, plot = False, path = 'Results/', loss_name = '', sum_flag = False, Hist_wt= [0.34, 0.33, 0.33]):

    testpixelerrors = []
    testerrors = []
    test_total_loss = []
    test_loss_kl = []
    test_loss_l1 = []
    test_loss_l1_temp = []
    test_loss_sum = []
    gt_count = []
    pred_count = []
    bs = 1
    for i in range(0,len(dataset_x), bs):
        pcount, pred_hist = classify(dataset_x,range(i,i+bs))

        if plot:
            processImages(path, 'test', i, classify, dataset_x[i], dataset_y[i], dataset_c[i], dataset_s[i], pcount, pred_hist, center_var, data_mean)
            
        pixelerr = np.abs(pcount - dataset_y[i:i+bs]).mean(axis=(2,3))[0][0]
        testpixelerrors.append(pixelerr)
        
        pred_est = (pcount/(ef)).sum(axis=(1,2,3))
        err = np.abs((dataset_y[i:i+bs]/ef).sum()-pred_est)[0]
        testerrors.append(err)

        y_shape = pred_hist.shape[0]
        hist = dataset_s[i:i+bs]
    
        p_prob = pred_hist/pred_hist.sum(axis = 1, keepdims=True) + (1e-6)
        p_prob1 = p_prob/p_prob.sum(axis =1, keepdims = True)
        t_prob = hist/hist.sum(axis = 1, keepdims=True) + (1e-6)
        t_prob1 = t_prob/t_prob.sum(axis = 1, keepdims =True)
        kl = (t_prob1*np.log((t_prob1)/(p_prob1)))
        loss_kl = kl.sum()/y_shape
        test_loss_kl.append(loss_kl)
        
        if loss_name == 'w_L1':
            loss_l1 = (center_var*np.abs(pred_hist - hist)).sum()/y_shape 
            
            loss_l1_temp = np.abs(pred_hist - hist).sum()/y_shape 
            test_loss_l1.append(loss_l1)
            test_loss_l1_temp.append(loss_l1_temp)

        elif loss_name == 'L1':
            loss_l1 = np.abs(pred_hist - hist).sum()/y_shape 
            loss_l1_temp = (center_var*np.abs(pred_hist - hist)).sum()/y_shape 
            test_loss_l1.append(loss_l1)
            test_loss_l1_temp.append(loss_l1_temp)


    
        if not sum_flag:
            loss_sum = 0.0* np.abs(pred_hist.sum(axis = 1) - hist.sum(axis =1)).sum()/y_shape
            test_loss_sum.append(loss_sum) 
        else:
            loss_sum = np.abs(pred_hist.sum(axis = 1) - hist.sum(axis =1)).sum()/y_shape
            test_loss_sum.append(loss_sum) 
            
        gt_count.append((dataset_y[i:i+bs]/ef).sum())
        pred_count.append(pred_est)
        test_total_loss.append(Hist_wt[0]*loss_kl + Hist_wt[1]*loss_l1 + pixelerr + Hist_wt[2]*loss_sum)
    
    return np.abs(testpixelerrors).mean(), np.abs(testerrors).mean(), np.abs(test_total_loss).mean(), np.mean(test_loss_kl), np.mean(test_loss_l1), np.mean(test_loss_l1_temp), np.mean(test_loss_sum), gt_count, pred_count 

def save_network(net,name, directory):
    pkl_params = lasagne.layers.get_all_param_values(net, trainable=True)
    out = open(directory + str(name) + ext, "wb") #bufsize=0
    pickle.dump(pkl_params, out)
    out.close()

def load_network(net,name):
    all_param_values = pickle.load(open(name, "rb" ))
    lasagne.layers.set_all_param_values(net, all_param_values, trainable=True)
    #lasagne.layers.set_all_param_values(net, all_param_values)

def trainer(net, net_hist, loss_list, variables, epochs, lr_value , lr_decay, batch_size, weight_decay_value, root_result, directory, Experiment_name, classify, classify_test, loss_name = '', sum_flag = False, Hist_wt = [0.34, 0.33, 0.33]):

    # Import Dataset
    dataset_path  = "/cluster/scratch/ksharma/dataset/Train_val_test5/"
    #dataset_path  = "/usr/stud/sharmaki/Projects/dataset/Train_val_test5/"
    train_set, val_set, test_set, data_mean = import_data(dataset_path)
    np_train_dataset_x, np_train_dataset_y, np_train_dataset_c,  np_train_dataset_s = train_set[0], train_set[1], train_set[2], train_set[3]
    np_val_dataset_x, np_val_dataset_y, np_val_dataset_c,  np_val_dataset_s = val_set[0], val_set[1], val_set[2], val_set[3]
    np_test_dataset_x, np_test_dataset_y, np_test_dataset_c,  np_test_dataset_s = test_set[0], test_set[1], test_set[2], test_set[3]
    print("Imported Data....")

    #MODEL_PATH = "/usr/stud/sharmaki/Projects/CountHistNet_1/Results_ICCV/Model_exp1_net1_w_L1_lr_0.0008_wd_1e-06/model_55.npz"
    #with np.load(MODEL_PATH) as f:
    #    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    #lasagne.layers.set_all_param_values([net, net_hist], param_values)
    
    lr = theano.shared(np.array(0.0, dtype=theano.config.floatX))
    loss, loss_count, loss_pix, loss_kl, loss_l1, loss_l1_temp, loss_sum, loss_reg = loss_list[0], loss_list[1], loss_list[2], loss_list[3], loss_list[4], loss_list[5], loss_list[6], loss_list[7]
    #if loss_name== 'w_L1':
    input_var, input_var_ex, target_var, target_var_hist, weight_var_hist = variables[0], variables[1], variables[2], variables[3], variables[4]
    #else:
    #    input_var, input_var_ex, target_var, target_var_hist = variables[0], variables[1], variables[2], variables[3]
    
    
    #Bins_var = np.linspace(0,200,17)
    Bins_var = np.linspace(0,200,9)
    
    center_var = (Bins_var[:-1] + Bins_var[1:])/2
    center_var = center_var/center_var.sum()
    center_var = np.tile(center_var, (batch_size,1)).astype(dtype = np.float32)
    
    params = lasagne.layers.get_all_params([net, net_hist], trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=lr)
    
    #if loss_name == 'w_L1':
    train_fn = theano.function([input_var_ex], [loss, loss_count, loss_pix, loss_kl, loss_l1, loss_l1_temp, loss_sum, loss_reg], updates=updates,
                        givens={input_var:np_train_dataset_x, target_var:  np_train_dataset_y,  target_var_hist: np_train_dataset_s, weight_var_hist: center_var})
    #else:
    #    train_fn = theano.function([input_var_ex], [loss, loss_count, loss_pix, loss_kl, loss_l1, loss_sum, loss_reg], updates=updates,
    #                        givens={input_var:np_train_dataset_x, target_var:  np_train_dataset_y,  target_var_hist: np_train_dataset_s})
    
    print("Compiled Training Function.....")

    lr.set_value(lr_value)
    best_valid_err = 99999999

    print("batch_size", batch_size)
    print("lr", lr.eval())

    datasetlength = len(np_train_dataset_x)
    #datasetlength = 20
    print("datasetlength",datasetlength)

    train_err_history_loss = []
    val_err_history_loss = [] 

    val_err_history_count = []
    train_err_history_count = []

    val_err_history_pix = []
    train_err_history_pix = []

    val_err_history_kl = [] 
    train_err_history_kl = []

    val_err_history_l1 = [] 
    train_err_history_l1 = []

    val_err_history_l1_temp = [] 
    train_err_history_l1_temp = []

    val_err_history_sum = [] 
    train_err_history_sum = []

    val_err_history_reg = [] 
    train_err_history_reg = []

    for epoch in range(epochs):
        epoch_err_pix = []
        epoch_err_count = []
        epoch_err_loss = []
        epoch_err_kl = []
        epoch_err_l1 = []
        epoch_err_l1_temp = []
        epoch_err_sum = []
        epoch_err_reg = []

        t_err_pix = 0
        t_err_count = 0
        t_err_loss = 0
        t_err_kl = 0
        t_err_l1 = 0
        t_err_l1_temp = 0
        t_err_sum = 0
        t_err_reg = 0
        
        todo = range(datasetlength)
        #todo = np.arange(datasetlength).astype(np.int32)
        #np.random.shuffle(todo)   
        
        for i in range(0,datasetlength, batch_size):
            ex = todo[i:i+batch_size]

            err_loss, err_count, err_pix, err_kl, err_l1, err_l1_temp, err_sum, err_reg = train_fn(ex)
            
            if i%50 == 0 :
                print("Epoch :", epoch," | Iteration : ", i ,"| Total_Loss :",np.around(err_loss,2), \
                    "| Pix Loss :",np.around(err_pix,2), "| Count_Loss : ",np.around(err_count.mean(),2), \
                    "| kl_Loss:", np.around(err_kl,2), "| l1_Loss:", np.around(err_l1,2), "| l1_Loss_temp:", np.around(err_l1_temp,2), \
                    "| reg_Loss:", np.around(err_reg,2),  \
                    "| hist_sum_Loss:", np.around(err_sum,2), "| Learning_rate:", np.around(lr.get_value(),5))
    
            epoch_err_pix.append(err_pix)
            epoch_err_count.append(err_count)
            epoch_err_loss.append(err_loss)
            epoch_err_kl.append(err_kl)
            epoch_err_l1.append(err_l1)
            epoch_err_l1_temp.append(err_l1_temp)
            epoch_err_sum.append(err_sum)
            epoch_err_reg.append(err_reg)

            t_err_pix += len(ex)*err_pix
            t_err_count += len(ex)*err_count
            t_err_loss += len(ex)*err_loss
            t_err_kl += len(ex)*err_kl
            t_err_l1 += len(ex)*err_l1
            t_err_l1_temp += len(ex)*err_l1_temp
            t_err_sum += len(ex)*err_sum
            t_err_reg += len(ex)*err_reg

        lr.set_value(lasagne.utils.floatX(lr.get_value() * lr_decay))

        #val_err_pix, val_err_count, val_err_loss, val_err_kl, val_err_l1, val_err_l1_temp, val_err_sum, aa,bb = test_perf(np_val_dataset_x[:100], np_val_dataset_y[:100], np_val_dataset_c[:100], np_val_dataset_s[:100], center_var[0,:], classify_test, data_mean, loss_name = loss_name, sum_flag = sum_flag, Hist_wt = Hist_wt)
        val_err_pix, val_err_count, val_err_loss, val_err_kl, val_err_l1, val_err_l1_temp, val_err_sum, aa,bb = test_perf(np_val_dataset_x, np_val_dataset_y, np_val_dataset_c, np_val_dataset_s, center_var[0,:], classify_test, data_mean, loss_name = loss_name, sum_flag = sum_flag, Hist_wt = Hist_wt)
        
        """
        train_err_history_loss.append(np.mean(epoch_err_loss))
        train_err_history_count.append(np.mean(epoch_err_count))
        train_err_history_kl.append(np.mean(epoch_err_kl))
        train_err_history_pix.append(np.mean(epoch_err_pix))
        train_err_history_l1.append(np.mean(epoch_err_l1))
        train_err_history_reg.append(np.mean(epoch_err_reg))
        """
        train_err_history_loss.append(t_err_loss/datasetlength)
        train_err_history_count.append(t_err_count/datasetlength)
        train_err_history_kl.append(t_err_kl/datasetlength)
        train_err_history_pix.append(t_err_pix/datasetlength)
        train_err_history_l1.append(t_err_l1/datasetlength)
        train_err_history_l1_temp.append(t_err_l1_temp/datasetlength)
        train_err_history_sum.append(t_err_sum/datasetlength)
        train_err_history_reg.append(np.mean(epoch_err_reg))
        
        val_err_history_loss.append(np.mean(val_err_loss))
        val_err_history_count.append(np.mean(val_err_count))
        val_err_history_pix.append(np.mean(val_err_pix))
        val_err_history_kl.append(np.mean(val_err_kl))
        val_err_history_l1.append(np.mean(val_err_l1))
        val_err_history_l1_temp.append(np.mean(val_err_l1_temp))
        val_err_history_sum.append(np.mean(val_err_sum))

        log_results(train_err_history_loss, val_err_history_loss, directory,'Total_loss',1)
        log_results(train_err_history_count, val_err_history_count, directory,'Count_loss',2)
        log_results(train_err_history_pix, val_err_history_pix, directory,'Pix_loss',3)
        log_results(train_err_history_kl, val_err_history_kl, directory,'kl_loss',4)
        log_results(train_err_history_l1, val_err_history_l1, directory,'l1_loss',5)
        log_results(train_err_history_l1_temp, val_err_history_l1_temp, directory,'l1_loss_temp',6)
        log_results(train_err_history_sum, val_err_history_sum, directory,'sum_loss',7)
        log_results(train_err_history_reg, val_err_history_reg, directory,'reg_loss',8)

        """
        if epoch % 100 == 0:
            print("Epoch : ",epoch, "| Train_Total_Loss :", np.around(train_err_history_loss[-1],2), \
                "| Val_Total_Loss :", np.around(val_err_history_loss[-1],2), \
                "| Train_Count Loss:",np.around(train_err_history_count[-1],2),\
                "| Val_Count Loss:",np.around(val_err_history_count[-1],2),\
                "| Train_Pix_loss:",np.around(train_err_history_pix[-1],2),\
                "| Val_Pix_loss:",np.around(val_err_history_pix[-1],2),\
                "| Train KL_Loss:",np.around(train_err_history_kl[-1],2), \
                "| Val KL_Loss:",np.around(val_err_history_kl[-1],2), \
                "| Train L1_Loss:",np.around(train_err_history_l1[-1],2), \
                "| Val L1_Loss:",np.around(val_err_history_l1[-1],2),\
                "| Train reg_Loss:",np.around(train_err_history_reg[-1],2) ) 
            save_network([net, net_hist], Experiment_name + "_" +  str(epoch), directory)
        """

                # a threshold is used to reduce processing when we are far from the goal
        if epoch%5==0 & epoch >=75:
            np.savez( directory + '/model_' +str(epoch) + '.npz', *lasagne.layers.get_all_param_values([net, net_hist]))



        if (val_err_l1 < best_valid_err):
            best_valid_err = val_err_l1
            #save_network([net, net_hist], Experiment_name + "_best_valid_err_", directory)
            #np.savez( directory + '/model_best_val.npz', *lasagne.layers.get_all_param_values([net, net_hist]))

            Best_info = " Best Result info last Epoch : " + str(epoch) + " | Train_Total_Loss :" + str( np.around(train_err_history_loss[-1],2)) + \
                "| Val_Total_Loss :"+ str(np.around(val_err_history_loss[-1],2)) + \
                "| Train_Count Loss:"+str(np.around(train_err_history_count[-1],2)) + \
                "| Val_Count Loss:" + str(np.around(val_err_history_count[-1],2)) +\
                "| Train_Pix_loss:"+ str(np.around(train_err_history_pix[-1],2)) +\
                "| Val_Pix_loss:"+ str(np.around(val_err_history_pix[-1],2)) +\
                "| Train KL_Loss:"+ str(np.around(train_err_history_kl[-1],2)) + \
                "| Val KL_Loss:"+ str(np.around(val_err_history_kl[-1],2)) + \
                "| Train L1_Loss:"+ str(np.around(train_err_history_l1[-1],2)) + \
                "| Val L1_Loss:"+ str(np.around(val_err_history_l1[-1],2)) + \
                "| Train L1_Loss Temp:"+ str(np.around(train_err_history_l1_temp[-1],2)) + \
                "| Val L1_Loss Temp:"+ str(np.around(val_err_history_l1_temp[-1],2)) + \
                "| Train sum_Loss:"+ str(np.around(train_err_history_sum[-1],2)) + \
                "| Val sum_Loss:"+ str(np.around(val_err_history_sum[-1],2)) + \
                "| Train reg_Loss:"+ str(np.around(train_err_history_reg[-1],2))     
    
    save_network([net, net_hist], Experiment_name + "_" +  str(epoch), directory)
    np.savez( directory + '/model_' +str(epoch) + '.npz', *lasagne.layers.get_all_param_values([net, net_hist]))


    Training_info = "Training info last Epoch : " + str(epoch) + " | Train_Total_Loss :" + str( np.around(train_err_history_loss[-1],2)) + \
        "| Val_Total_Loss :"+ str(np.around(val_err_history_loss[-1],2)) + \
        "| Train_Count Loss:"+str(np.around(train_err_history_count[-1],2)) + \
        "| Val_Count Loss:" + str(np.around(val_err_history_count[-1],2)) +\
        "| Train_Pix_loss:"+ str(np.around(train_err_history_pix[-1],2)) +\
        "| Val_Pix_loss:"+ str(np.around(val_err_history_pix[-1],2)) +\
        "| Train KL_Loss:"+ str(np.around(train_err_history_kl[-1],2)) + \
        "| Val KL_Loss:"+ str(np.around(val_err_history_kl[-1],2)) + \
        "| Train L1_Loss:"+ str(np.around(train_err_history_l1[-1],2)) + \
        "| Val L1_Loss:"+ str(np.around(val_err_history_l1[-1],2)) + \
        "| Train L1_Loss temp:"+ str(np.around(train_err_history_l1_temp[-1],2)) + \
        "| Val L1_Loss temp:"+ str(np.around(val_err_history_l1_temp[-1],2)) + \
        "| Train sum_Loss:"+ str(np.around(train_err_history_sum[-1],2)) + \
        "| Val sum_Loss:"+ str(np.around(val_err_history_sum[-1],2)) + \
        "| Train reg_Loss:"+ str(np.around(train_err_history_reg[-1],2))      
        
        #visualize training
        #processImages(str(epoch) + '-cell',4)
    
    Result_file  = open(root_result + '/Summary.txt', "a+")
    Result_file.write(Experiment_name)
    Result_file.write('\n' + Training_info)
    Result_file.write('\n' + Best_info)
    Result_file.write('\n \n \n ')
    Result_file.close()

    Test_directory = directory + '/Test_results/'
    if not os.path.exists(Test_directory):
        os.mkdir(Test_directory)

    test_err_pix,test_err_count,test_err_loss, test_err_kl,test_err_l1, test_err_l1_temp, test_err_sum, aa,bb = test_perf(np_test_dataset_x, np_test_dataset_y, np_test_dataset_c, np_test_dataset_s, center_var[0,:], classify_test,  data_mean,  plot = True, path = Test_directory, loss_name = loss_name, sum_flag = sum_flag, Hist_wt = Hist_wt)
    #test_err_pix,test_err_count,test_err_loss, test_err_kl,test_err_l1, test_err_l1_temp, test_err_sum, aa,bb = test_perf(np_test_dataset_x[:20], np_test_dataset_y[:20], np_test_dataset_c[:20], np_test_dataset_s[:20], center_var[0,:], classify_test,  data_mean,  plot = True, path = Test_directory, loss_name = loss_name, sum_flag = sum_flag, Hist_wt = Hist_wt)
    

    Error_file  = open(Test_directory + '/Error_Summary.txt', "w")
    Error_file.write('\n Total_count_Mean abs Error :' + str(test_err_count))
    Error_file.write('\n Total_kl_Mean abs Error :' + str(test_err_kl))
    Error_file.write('\n Total_l1_Mean abs Error :' + str(test_err_l1))
    Error_file.write('\n Total_l1_ temp Mean abs Error :' + str(test_err_l1_temp))
    Error_file.write('\n Total_sum_Mean abs Error :' + str(test_err_sum))
    Error_file.close()
    
    #with np.load(directory + '/model.npz') as f:
    #    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    #lasagne.layers.set_all_param_values([net_1, net_hist_1], param_values)

    return np.min(val_err_l1), Training_info , Best_info

def loss_func(net, net_hist, input_var, input_var_ex, reg, loss_name = '', sum_flag = False, Hist_wt = [0.34, 0.33, 0.33]):
    prediction, prediction_hist = lasagne.layers.get_output([net, net_hist], deterministic=False)
    prediction_count = (prediction/ef).sum(axis=(2,3))
    classify = theano.function([input_var, input_var_ex], [prediction, prediction_hist])

    #Val/Test
    prediction_t, prediction_hist_t = lasagne.layers.get_output([net, net_hist], deterministic=True)
    prediction_count_t = (prediction_t/ef).sum(axis=(2,3))
    classify_test = theano.function([input_var, input_var_ex], [prediction_t, prediction_hist_t])

    ## Comping Theano Training Function
    target_var = T.tensor4('target')
    target_var_hist = T.matrix('target_hist')
    weight_var_hist = T.matrix('weight_hist')
    #Mean Absolute Error is computed between each count of the count map
    l1_loss = T.abs_(prediction - target_var[input_var_ex])

    #Mean Absolute Error is computed for the overall image prediction
    prediction_count2 =(prediction/ef).sum(axis=(2,3))
    loss_count = T.abs_(prediction_count2 - (target_var[input_var_ex]/ef).sum(axis=(2,3))).mean()
    loss_pix = l1_loss.mean()
    
    #KL DIV LOSS
    y_shape = prediction_hist.shape[0]
    target_hist = target_var_hist[input_var_ex]
    p_prob_1 = prediction_hist/prediction_hist.sum(axis = 1, keepdims=True) + (1e-6)
    p_prob = p_prob_1/p_prob_1.sum(axis=1, keepdims = True)
    t_prob_1 = target_hist/target_hist.sum(axis = 1, keepdims=True) + (1e-6)
    t_prob = t_prob_1/t_prob_1.sum(axis = 1, keepdims = True)
    kl= (t_prob*T.log((t_prob)/(p_prob)))
    loss_kl = kl.sum()/y_shape

    if loss_name == 'L1':
        loss_l1 = (T.abs_(prediction_hist - target_hist)).sum()/y_shape 
        loss_l1_temp = (weight_var_hist*T.abs_(prediction_hist - target_hist)).sum()/y_shape
    elif loss_name == 'w_L1':
        loss_l1 = (weight_var_hist*T.abs_(prediction_hist - target_hist)).sum()/y_shape
        loss_l1_temp = (T.abs_(prediction_hist - target_hist)).sum()/y_shape 

    
    if not sum_flag:
        loss_sum =  0.0*(T.abs_(prediction_hist.sum(axis = 1) - target_hist.sum(axis=1))).sum()/y_shape 
    else:   
        loss_sum =  (T.abs_(prediction_hist.sum(axis = 1) - target_hist.sum(axis=1))).sum()/y_shape 
    

    #loss5 = T.abs_((target_var[input_var_ex]/ef).sum(axis=(2,3)) - prediction_hist.sum(axis = 1, keepdims = True)).sum()/y_shape
    #loss_reg = 0.5*reg*regularize_network_params([net, net_hist], l2)
    loss_reg = 0.5*reg*regularize_network_params([net_hist], l2)

    loss = loss_pix + Hist_wt[0]*loss_kl + Hist_wt[1]*loss_l1 + Hist_wt[2]*loss_sum + loss_reg
    
    print("Compiled Loss Functions....")
    #if loss_name == 'w_L1':
    return [loss, loss_count, loss_pix, loss_kl, loss_l1, loss_l1_temp, loss_sum, loss_reg], [input_var, input_var_ex, target_var, target_var_hist, weight_var_hist], classify, classify_test
    #else:
    #    return [loss, loss_count, loss_pix, loss_kl, loss_l1, loss_sum, loss_reg], [input_var, input_var_ex, target_var, target_var_hist], classify

def log_results( train_loss,  val_loss, Log_dir, plotname, i ):

    if not os.path.exists(Log_dir):
        os.mkdir(Log_dir)    
    #plt.Figure()
    Error_file  = open(Log_dir + '/' + plotname + '_Summary.txt', "w")
    Error_file.write('Train_loss: ' + str(train_loss))
    Error_file.write('\n val_loss: '+  str(val_loss))
    Error_file.close()

    plt.figure(i+10, figsize=(15, 10))
    plt.plot(range(len(train_loss)), train_loss, 'r', label = 'train')
    plt.plot(range(len(val_loss)), val_loss, 'g', label = 'val')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Train & Validation ' + plotname)
    plt.savefig(Log_dir + '/' + plotname + '.png')
    #canvas.print_figure('test')
    plt.close()

def processImages(path, name, i, classify, image, label, count_value, histogram, pcount, p_hist, center_var, data_mean):
    fig = plt.Figure(figsize=(18, 9), dpi=160)
    gcf = plt.gcf()
    gcf.set_size_inches(18, 15)
    fig.set_canvas(gcf.canvas)

    img, lab, count, gt_hist = image, label, count_value, histogram
    
    #print str(i),count
    #pcount, p_hist = classify([img], [0])
    lab_est = [(l.sum()/(ef)).astype(np.int) for l in lab]
    
    #print lab_est
    pred_est = [(l.sum()/(ef)).astype(np.int) for l in pcount]
    
    #print(str(i),"label est ",lab_est," --> predicted est ",pred_est)
    img = img.transpose((1,2,0))
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
    ax3.set_title("Regression target, {}x{} sliding window.".format(patch_size, patch_size))
    ax3.imshow(np.concatenate((lab),axis=1), interpolation='none')
    ax4.set_title("Predicted counts")
    ax4.imshow(pcount.squeeze(), interpolation='none')
    
    ax5.set_title("Real " + str(lab_est))
    ax5.set_ylim((0, np.max(lab_est)*2))
    ax5.set_xticks(np.arange(0, noutputs, 1.0))
    ax5.bar(range(noutputs),lab_est, align='center')
    ax6.set_title("Pred " + str(pred_est))
    ax6.set_ylim((0, np.max(lab_est)*2))
    ax6.set_xticks(np.arange(0, noutputs, 1.0))
    ax6.bar(range(noutputs),pred_est, align='center')

    #Bins = np.linspace(0,200,9)
    Bins = np.linspace(0,200,17)
    ax7.set_title("Gt Histogram")
    ax7.hist(Bins[:-1], weights= gt_hist.T , bins = Bins)
    
    ax8.set_title("Pred Histogram")
    ax8.hist(Bins[:-1], weights= p_hist.T , bins = Bins)
    
    if not os.path.exists(path + '/images-cell'): 
        os.mkdir(path + '/images-cell')
    fig.savefig(path + '/images-cell/image-' + str(i) + "-" + name + '.png', bbox_inches='tight', pad_inches=0)

import sys,os,time,random
import numpy as np 
from random import shuffle
import cv2
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
import skimage
from skimage.io import imread, imsave
import pickle
from skimage import measure
from skimage import filters
from tqdm import tqdm

def preprocess_data(ROOT_DIR, TARGET_DIR):
    """
    Function: To prepare train, val and test dataset from raw images (Large size) including sampling 
    patches of size 256 X 256 px --> Data Augmentation 

    Input:
    ROOT_DIR: path of directory containing raw images and labels
    TARGET_DIR: path of preprocessed dataset directory
    """

    print('Preparing Dataset............')
    #Step 1: Prepare Train, Val and Test 
    print('\n Step 1 : Preparing Train, Val and Test datasets')
    TARGET_DIR_1 = os.path.join(TARGET_DIR, 'Sampled_Data')
    Distribution = [0.6, 0.2, 0.2]  #[train, val, test]
    train_val_test(ROOT_DIR, TARGET_DIR_1, Distribution)

    #Step 2: Sample Raw Data Images and labels 
    print('\n Step 2: Sampling Raw Data Images and Labels......')
    ROOT_DIR_2= TARGET_DIR_1
    TARGET_DIR_2 = os.path.join(TARGET_DIR, 'Train_val_test')
    sampler(ROOT_DIR_2, TARGET_DIR_2, 'Train', img_size= 256, stride = 128)
    sampler(ROOT_DIR_2, TARGET_DIR_2, 'Val', img_size= 256, stride = 128)
    sampler(ROOT_DIR_2, TARGET_DIR_2, 'Test', img_size= 256, stride = 128)

    #Step 3: Data Augmentation
    print('\n Step 3: Data Augmentation...........')
    ROOT_DIR_3 = os.path.join(TARGET_DIR_2, 'Train')
    TARGET_DIR_3 = os.path.join(TARGET_DIR_2, 'Train_Aug')
    data_augmentation(ROOT_DIR_3, TARGET_DIR_3)

    print('\n Data preparation Finished!')

def train_val_test(ROOT_DIR, TARGET_DIR, Distribution):
    """
    Function: To distribute the data in train, val and test dataset according to given distribution 

    Input:
    ROOT_DIR: path of directory containing raw images (Large size) and labels
    TARGET_DIR: path of distribued dataset directory
    Distribution: [train, val, test] distribution ratio of data, SUM(Distribution) == 1
    """

    Image_dir = ROOT_DIR + '/Images'
    Label_dir = ROOT_DIR + '/Labels'
    
    Train = Distribution[0]
    Val = Distribution[1]
    Test = Distribution[2]

    Dst_dir = TARGET_DIR
    Train_set = Dst_dir + 'Train/'
    Val_set = Dst_dir + 'Val/'
    Test_set = Dst_dir + 'Test/'

    # Create folders for train, val and test set
    if not os.path.exists(Dst_dir):
        os.mkdir(Dst_dir)
        
        os.mkdir(Train_set)
        os.mkdir(Train_set + 'Images/')
        os.mkdir(Train_set + 'Labels/')

        os.mkdir(Val_set)
        os.mkdir(Val_set + 'Images/')
        os.mkdir(Val_set + 'Labels/')
        
        os.mkdir(Test_set)
        os.mkdir(Test_set + 'Images/')
        os.mkdir(Test_set + 'Labels/')

    list_dir_image = np.array(os.listdir(Image_dir))
    list_dir_label = np.array(os.listdir(Label_dir))

    length = len(list_dir_image)
    index = np.arange(len(list_dir_image))
    shuffle(index)

    # Distribute the path of raw images and labels among train, test and val set
    Train_Image_list = list_dir_image[index[:int(Train*length)]]
    Train_Label_list = list_dir_label[index[:int(Train*length)]]

    Val_Image_list = list_dir_image[index[int(Train*length) : int(( Train + Val )*length) ]]
    Val_Label_list = list_dir_label[index[int(Train*length) : int((Train + Val )*length) ]]

    Test_Image_list = list_dir_image[index[int(( Train + Val )*length) : ]]
    Test_Label_list = list_dir_label[index[int((Train + Val )*length) :]]

    # Read the image and label file and write it to the corresponding dataset (train, val or test)
    for i in range(len(Train_Image_list)):
        image_path = os.path.join(Image_dir, Train_Image_list[i])
        label_path = os.path.join(Label_dir, Train_Label_list[i])

        img = cv2.imread(image_path)
        label = cv2.imread(label_path)

        try:
            cv2.imwrite( Train_set + 'Images/' + os.path.basename(image_path), img)
            cv2.imwrite( Train_set + 'Labels/' + os.path.basename(label_path), label)
        except:
            import pdb; pdb.set_trace()

    for i in range(len(Val_Image_list)):
        image_path = os.path.join(Image_dir, Val_Image_list[i])
        label_path = os.path.join(Label_dir, Val_Label_list[i])

        img = cv2.imread(image_path)
        label = cv2.imread(label_path)

        try:
            cv2.imwrite( Val_set + 'Images/' + os.path.basename(image_path), img)
            cv2.imwrite( Val_set + 'Labels/' + os.path.basename(label_path), label)
        except:
            import pdb; pdb.set_trace()
        
        
    for i in range(len(Test_Image_list)):
        image_path = os.path.join(Image_dir, Test_Image_list[i])
        label_path = os.path.join(Label_dir, Test_Label_list[i])

        img = cv2.imread(image_path)
        label = cv2.imread(label_path)

        try:
            cv2.imwrite( Test_set + 'Images/' + os.path.basename(image_path), img)
            cv2.imwrite( Test_set + 'Labels/' + os.path.basename(label_path), label)
        except:
            import pdb; pdb.set_trace()
    
    print('Train_set : %d  | Val_set : %d  | Test_set : %d'%(len(Train_Image_list), len(Val_Image_list), len(Test_Image_list)))

def sampler(ROOT_DIR, TARGET_DIR, setname, img_size = 256, stride = 128):
    """
    Function: Sample patches of img_size from the images and labels in a given dataset with a given stride

    Input:
    ROOT_DIR: path of directory containing train, val and test dataset
    TARGET_DIR: path of directory containing sampled train, val and test dataset 
    setname: name of dataset (train, val or test)
    img_size: size of image patch (pixels)
    stride: overlap between two patches (pixels)
    """

    # Creating Target directory and sampled dataset folders
    if not os.path.exists(TARGET_DIR):
        os.mkdir(TARGET_DIR)
    
    if not os.path.exists(TARGET_DIR + setname):
        os.mkdir(TARGET_DIR + setname)
        os.mkdir(TARGET_DIR + setname + '/Images/')
        os.mkdir(TARGET_DIR + setname + '/Labels/')

    # Sampling images and labels from a given dataset       
    for i in tqdm(range(len(os.listdir(os.path.join(ROOT_DIR + setname, 'Images'))))):
        image_path =  os.listdir(os.path.join(ROOT_DIR + setname, 'Images'))[i]
        img_name = image_path.split(".")[0]
        image = cv2.imread(os.path.join(ROOT_DIR + setname, 'Images',image_path))
        label = cv2.imread(os.path.join(ROOT_DIR + setname, 'Labels', img_name + '_mask.png'))

        # Extracting patches of size img_size for a single image based on location of patch (boundary patch, center patch, corner patch)
        for i in range(int(image.shape[0]/(img_size-stride)) - 1):    
            for j in range(int(image.shape[1]/(img_size-stride))):

                if (i  == int(image.shape[0]/(img_size-stride))-2) and not(j == int(image.shape[1]/(img_size-stride))-1):
                    sample_image(image, label, i, j, TARGET_DIR + setname, img_name, img_size, stride, Boundary = 'i_boundary')
                elif not (i == int(image.shape[0]/(img_size-stride))-2) and (j  == int(image.shape[1]/(img_size-stride))-1):
                    sample_image(image, label, i, j, TARGET_DIR + setname, img_name, img_size, stride, Boundary = 'j_boundary')
                elif (i == int(image.shape[0]/(img_size-stride))-2) and (j == int(image.shape[1]/(img_size-stride))-1):
                    sample_image(image, label, i, j, TARGET_DIR + setname, img_name, img_size, stride, Boundary = 'ij_boundary')
                else:
                    sample_image(image, label, i, j, TARGET_DIR + setname, img_name, img_size, stride, Boundary = 'Normal' )
    
    print('Number of Sampled Images : ',len(os.listdir(TARGET_DIR + setname + '/Images')))
    
def sample_image(img, img_label, i, j, path, img_name, img_size,stride, Boundary ):
    """
    Function: Sample patches of img_size from a single image and label with a given stride

    Input:
    img: image array(Large size, raw image)
    img_label: label array (Large size, raw pixel wise annotated label)
    i: row number 
    j: coloum number 
    path: Target direcotry path of the sampled dataset
    img_name: base name of the image
    img_size: size of the path (pixels)
    stride: overlap between two patches
    Boundary: i_boundary (patch on horizontal axis boundary), j_boundary (patch on vertical axis boundary), 
            ij_boundary (corner patch), normal (center patch)
    """
    
    if Boundary == 'i_boundary':
        i_min = img.shape[0]-img_size
        i_max = img.shape[0]
        
        j_min = j*stride
        j_max = j*stride + img_size
        
        img_cropped = img[i_min:i_max, j_min:j_max,:]
        img_label_cropped = img_label[i_min:i_max, j_min:j_max,:]

        # Label connected regions of the pixel wise label image
        labels, num_labels =  measure.label(img_label_cropped, background = 0, return_num= True, connectivity=1)
        cc = measure.regionprops(labels, img_label_cropped)

        for k in range(num_labels):
            if cc[k].area < 3: 
                # removing small noise area in the label image 
                img_label_cropped[img_label_cropped == k] = 0
    
        img_path = path + '/Images' + '/'+ img_name + '_' + str(i) + '_' + str(j) + '.png'
        img_label_path = path + '/Labels' + '/'+ img_name + '_' +  str(i) + '_' + str(j) + '_mask' + '.png'
        

        cv2.imwrite(img_path,img_cropped)
        cv2.imwrite(img_label_path,img_label_cropped)
    
    elif Boundary == 'j_boundary':
        i_min = i*stride
        i_max = i*stride + img_size
        
        j_min = img.shape[1]-img_size
        j_max = img.shape[1]

        img_cropped = img[i_min:i_max, j_min:j_max,:]
        img_label_cropped = img_label[i_min:i_max, j_min:j_max,:]

        img_path = path + '/Images' + '/'+ img_name + '_' + str(i) + '_' + str(j) + '.png'
        img_label_path = path + '/Labels' + '/'+ img_name + '_' +  str(i) + '_' + str(j) + '_mask' + '.png'
    
        labels, num_labels =  measure.label(img_label_cropped, background = 0, return_num= True, connectivity=1)
        cc = measure.regionprops(labels, img_label_cropped)

        for k in range(num_labels):
            if cc[k].area < 3: 
                img_label_cropped[img_label_cropped == k] = 0
    
        cv2.imwrite(img_path,img_cropped)
        cv2.imwrite(img_label_path,img_label_cropped)
     
    elif Boundary == 'ij_boundary':
        i_min = img.shape[0]-img_size
        i_max = img.shape[0]
        
        j_min = img.shape[1]-img_size
        j_max = img.shape[1]

        img_cropped = img[i_min:i_max, j_min:j_max,:]
        img_label_cropped = img_label[i_min:i_max, j_min:j_max,:]

        img_path = path + '/Images' + '/'+ img_name + '_' + str(i) + '_' + str(j) + '.png'
        img_label_path = path + '/Labels' + '/'+ img_name + '_' +  str(i) + '_' + str(j) + '_mask' + '.png'

        labels, num_labels =  measure.label(img_label_cropped, background = 0, return_num= True, connectivity=1)
        cc = measure.regionprops(labels, img_label_cropped)

        for k in range(num_labels):
            if cc[k].area < 3: 
                img_label_cropped[img_label_cropped == k] = 0
    
        cv2.imwrite(img_path,img_cropped)
        cv2.imwrite(img_label_path,img_label_cropped)
     
    
    else:
        i_min = i*stride
        i_max = i*stride + img_size

        j_min = j*stride
        j_max = j*stride + img_size
        img_cropped = img[i_min:i_max, j_min:j_max,:]
        img_label_cropped = img_label[i_min:i_max, j_min:j_max,:]

        img_path = path + '/Images' + '/'+ img_name + '_' + str(i) + '_' + str(j) + '.png'
        img_label_path = path + '/Labels' + '/'+ img_name + '_' +  str(i) + '_' + str(j) + '_mask' + '.png'
    
        labels, num_labels =  measure.label(img_label_cropped, background = 0, return_num= True, connectivity=1)
        cc = measure.regionprops(labels, img_label_cropped)

        for k in range(num_labels):
            if cc[k].area < 3: 
                img_label_cropped[img_label_cropped == k] = 0
    
        cv2.imwrite(img_path,img_cropped)
        cv2.imwrite(img_label_path,img_label_cropped)
    
def prepare_data(root):
    """
    Function: Prepare train, val and test dataset as pickle file after 
            pre processing (distribution, sampling, augmentation) dataset 
            (Image, label : redundant count map, 2, 4, 8, 16 bin size distribution histogram) 
            as pickle file
    Input:
    root: path to directory containing train_aug, val and test dataset
    """

    if os.path.exists(root + '/val.p'):
        print('Val dataset already exists....')
    else:
        prepare_dataset(root + '/val.p', root + '/Val/Images/', root + '/Val/Labels/')
        
    if os.path.exists(root + '/test.p'):
        print('Test dataset already exists....')
    else:
        prepare_dataset(root + '/test.p', root + '/Test/Images/', root + '/Test/Labels/') 
      
    if os.path.exists(root + '/train.p'):
        print('Train dataset already exists....')
    else:
        prepare_dataset(root + '/train.p', root + '/Train_Aug/Images/', root + '/Train_Aug/Labels/')

def import_data(root, num_bins):
    """
    Function: Import whole data (train, val and test) using pickle file (shuffled)
    
    Input:
    root: path to directory containing train_aug, val and test dataset pickle file

    Output:
    train_set: train dataset list containing images, redundant count maps, count, size histogram 
    val_set: val dataset list containing images, redundant count maps, count, size histogram
    test_set: test dataset list containing images, redundant count maps, count, size histogram
    
    """
    if os.path.exists(root + '/val.p'):
        print('Loading Val Datasets....')
        val_data = pickle.load(open(root + '/val.p', "rb" ))
    if os.path.exists(root + '/test.p'):
        print('Loading Test Datasets....')
        test_data = pickle.load(open(root + '/test.p', "rb" ))    

    if os.path.exists(root + '/train.p'):
        print('Loading Train Datasets....')
        train_data = pickle.load(open(root + '/train.p', "rb" ))

    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)

    if num_bins == 2:
        k = 3
    elif num_bins == 4:
        k = 4
    elif num_bins == 8:
        k = 5
    elif num_bins == 16:
        k = 6
    else:
        print("Error: wrong value for num of bins !!")

    np_train_dataset_x = np.asarray([d[0] for d in train_data], dtype=theano.config.floatX)
    np_train_dataset_y = np.asarray([d[1] for d in train_data], dtype=theano.config.floatX)
    np_train_dataset_c = np.asarray([d[2] for d in train_data], dtype=theano.config.floatX)
    np_train_dataset_s = np.asarray([d[k] for d in train_data], dtype=theano.config.floatX)

    np_val_dataset_x = np.asarray([d[0] for d in val_data], dtype=theano.config.floatX)
    np_val_dataset_y = np.asarray([d[1] for d in val_data], dtype=theano.config.floatX)
    np_val_dataset_c = np.asarray([d[2] for d in val_data], dtype=theano.config.floatX)
    np_val_dataset_s = np.asarray([d[k] for d in val_data], dtype=theano.config.floatX)

    np_test_dataset_x = np.asarray([d[0] for d in test_data], dtype=theano.config.floatX)
    np_test_dataset_y = np.asarray([d[1] for d in test_data], dtype=theano.config.floatX)
    np_test_dataset_c = np.asarray([d[2] for d in test_data], dtype=theano.config.floatX)
    np_test_dataset_s = np.asarray([d[k] for d in test_data], dtype=theano.config.floatX)
    
    #Data Normalization
    np_train_dataset_x = np_train_dataset_x/255.0
    np_val_dataset_x = np_val_dataset_x/255.0
    np_test_dataset_x = np_test_dataset_x/255.0
    data_mean = np.mean(np_train_dataset_x, axis = 0)
    np_train_dataset_x = np_train_dataset_x - data_mean
    np_val_dataset_x = np_val_dataset_x - data_mean
    np_test_dataset_x = np_test_dataset_x - data_mean
    
    np_train_dataset_x = np_train_dataset_x.transpose((0,3,1,2))
    np_val_dataset_x = np_val_dataset_x.transpose((0,3,1,2))
    np_test_dataset_x = np_test_dataset_x.transpose((0,3,1,2))
    
    print("np_train_dataset_x", np_train_dataset_x.shape)
    print("np_train_dataset_y", np_train_dataset_y.shape)
    print("np_train_dataset_c", np_train_dataset_c.shape)
    print("np_train_dataset_s", np_train_dataset_s.shape)

    print("\n np_val_dataset_x", np_val_dataset_x.shape)
    print("np_val_dataset_y", np_val_dataset_y.shape)
    print("np_val_dataset_c", np_val_dataset_c.shape)
    print("np_val_dataset_s", np_val_dataset_s.shape)


    print("\n np_test_dataset_x", np_test_dataset_x.shape)
    print("np_test_dataset_y", np_test_dataset_y.shape)
    print("np_test_dataset_c", np_test_dataset_c.shape)
    print("np_test_dataset_s", np_test_dataset_s.shape)

    print("number of TRAIN counts total ", np_train_dataset_c.sum())
    print("number of TRAIN counts on average ", np_train_dataset_c.mean(), "+-", np_train_dataset_c.std())
    print("counts TRAIN min:", np_train_dataset_c.min(), "max:", np_train_dataset_c.max())

    print("\n number of VAL counts total ", np_val_dataset_c.sum())
    print("number of VAL counts on average ", np_val_dataset_c.mean(), "+-", np_val_dataset_c.std())
    print("counts VAL min:", np_val_dataset_c.min(), "max:", np_val_dataset_c.max())

    print("\n number of TEST counts total ", np_test_dataset_c.sum())
    print("number of TEST counts on average ", np_test_dataset_c.mean(), "+-", np_test_dataset_c.std())
    print("counts TEST min:", np_test_dataset_c.min(), "max:", np_test_dataset_c.max())

    print("\n Total cells in training", np.sum(np_train_dataset_c[0:], axis=0))
    print("Total cells in validation", np.sum(np_val_dataset_c[0:], axis=0))
    print("Total cells in testing", np.sum(np_test_dataset_c[0:], axis=0))
    
    train_set = [np_train_dataset_x, np_train_dataset_y, np_train_dataset_c,  np_train_dataset_s]
    val_set = [np_val_dataset_x, np_val_dataset_y, np_val_dataset_c,  np_val_dataset_s]
    test_set = [np_test_dataset_x, np_test_dataset_y, np_test_dataset_c,  np_test_dataset_s]
    return train_set, val_set, test_set, data_mean

def prepare_dataset(datasetfilename, root_img, root_label):
    """
    Function: Prepare a dataset as pickle file after 
            pre processing (distribution, sampling, augmentation) dataset 
            (Image, label : redundant count map, 2, 4, 8, 16 bin size distribution histogram) 
            as pickle file
    Input:
    datasetfilename: name of the dataset pickle file
    root_img: path to image directory of the dataset
    root_label: path to label directory of the dataset
    """
    img_list = os.listdir(root_img)
    label_list  = os.listdir(root_label)

    # process an image and label to prepare label (count map and histogram)
    def processInput(i):    

        img_path = root_img + img_list[i]
        label_path = root_label + label_list[i]

        if not (img_path.split('.')[0] == label_path.split('.')[0]):
            print("Error: Image and Label name are not same!!")
            import pdb; pdb.set_trace()
        
        img = cv2.imread(img_path)
        redundantCountMap, count, gt_hist2, gt_hist4, gt_hist8, gt_hist16, _ = prepare_Label(label_path, patch_size = 32)

        return (img,redundantCountMap ,count, gt_hist2, gt_hist4, gt_hist8, gt_hist16)

    num_cores = multiprocessing.cpu_count()
    print('Working on Num Cores :', num_cores)
    
    dataset = []
    # Prepare dataset using parallelization
    dataset.append(Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in tqdm(range(len(img_list)))))
    dataset = dataset[0]
    print("Writing...", datasetfilename)
    out = open(datasetfilename, "wb", 0)
    pickle.dump(dataset, out)
    out.close()
    print("DONE !!!")

def prepare_Label(labelPath, patch_size):
    """
    Function: prepare labels (size distribution histogram 2, 4, 8, 16 bin, redundant Count Map)
    
    Input:
    labelPath: path of the pixel wise annotated label
    patch_size: the size of patch (based on receptive field of network) for predicting redundant count map

    Output:
    redundantCountMap: redundant count map 
    count: total object count in an image
    hist2, hist4, hist8, hist16: 2,4,8,16 bin size distribution histogram
    larvae_sizes: size of larvae instances in an image
    """
    count_map, hist2, hist4, hist8, hist16, larvae_sizes = prepare_CountMap_HistLabels(labelPath)
    count_map = np.pad(count_map, pad_width = ((patch_size, patch_size), (patch_size, patch_size), (0,0)), mode = "constant", constant_values=-1)
    redundantCountMap, count  = prepare_redundantCountMap(count_map, patch_size)
    return redundantCountMap, count, hist2, hist4, hist8, hist16, larvae_sizes

def prepare_CountMap_HistLabels(labelPath): 
    """
    Function: prepare count map and size distribution histogram 2, 4, 8, 16 bin
    
    Input:
    labelPath: path of the pixel wise annotated label

    Output:
    count_map: count map (Integral over count map == object count)
    hist2, hist4, hist8, hist16: 2,4,8,16 bin size distribution histogram
    larvae_sizes: size of larvae instances in an image
    """ 
    bins16 = np.linspace(0,200,17)
    bins8 = np.linspace(0,200,9)
    bins4 = np.linspace(0,200,5)
    bins2 = np.linspace(0,200,3)
    
    larvae_sizes = []
    labs_1 = cv2.imread(labelPath)
    labels, _ =  measure.label(labs_1[:,:,0], background = 0, return_num= True, connectivity=1)
    cc = measure.regionprops(labels, labs_1[:,:,0])
    count_map = np.zeros(labs_1.shape[0:2])
    for i in range(len(cc)):
        count_map[int(cc[i].centroid[0]), int(cc[i].centroid[1])] = 1
        larvae_sizes.append(cc[i].area)
    
    larvae_sizes = np.array(larvae_sizes)    
    larvae_sizes[larvae_sizes > 200] = 199
    
    hist2 = plt.hist(larvae_sizes, bins2)[0]
    hist4 = plt.hist(larvae_sizes, bins4)[0]
    hist8 = plt.hist(larvae_sizes, bins8)[0]
    hist16 = plt.hist(larvae_sizes, bins16)[0]
    
    return count_map, hist2, hist4, hist8, hist16, larvae_sizes

def prepare_redundantCountMap(count_map, patch_size):
    """
    Function: prepare redundant count map from the count map given patch_size
    
    Input:
    count_map : count map (Integral over count map == object count)
    patch_size: the size of patch (based on receptive field of network) for predicting redundant count map

    Output:
    redundantCountMap: redundant count map 
    count: total object count in an image
    """ 
    height = 256
    width = 256
    redundantCountMap = np.zeros((1, height, width))
    for y in range(0,height):
        for x in range(0,width):
            count = (count_map[x:x+patch_size, y:y+patch_size] == 1).sum()
            redundantCountMap[0][y][x] = count

    total_count = (count_map == 1).sum()
    return redundantCountMap, total_count

###################################################################
#                       Data Augmentation utils                   #
###################################################################

def random_noise(image, max_noise = 0.1):
    """
    Function: add random noise in the image
    
    Input:
    image : image array
    max_noise: maximum noise value

    Output:
    image: augmented image
    """
    image_format = image.dtype
    if image_format == np.uint8:
        max_noise *= 255
        image = image.astype(np.float32)

    noise_mat = max_noise*np.random.rand(image.shape[0],image.shape[1],image.shape[2])
    image = image + noise_mat

    if image_format == np.uint8:
        image[image < 0] = 0
        image[image > 255] = 255
        image = image.astype(np.uint8)
    else:
        image[image < 0] = 0
        image[image > 1] = 1
    return image

def random_contrast(image, min_contrast = 0.3, max_contrast = 3.0):
    """
    Function: add random contrast in the image
    
    Input:
    image : image array
    min_contrast: min threshold for contrast
    max_contrast: max threshold for contrast

    Output:
    image: augmented image
    """
    #convert to float image if uint8
    image_format = image.dtype
    if image_format == np.uint8:
        image = image.astype(np.float32)/255.0

    maxIntensity = 1.0 # depends on dtype of image data

    # Parameters for manipulating image data
    phi = 1
    theta = 1

    contrast = np.random.rand(1,1) * (max_contrast - min_contrast) + min_contrast

    # change contrast
    image = (maxIntensity/phi)*(image/(maxIntensity/theta))**contrast
    image[image < 0] = 0
    image[image > 1] = 1

    #convert back to uint8 if it was uint8 before
    if image_format == np.uint8:
        image *= 255
        image = image.astype(np.uint8)

    return image

def augmentation(img_path, label_path, img_aug_path, label_aug_path):
    """
    Function: applying augmentation on a given image and label
    
    Input:
    img_path : path of the image file
    label_path : path of the label file
    img_aug_path: target path of the image folder
    label_aug_path: target path of the label folder
    """
    img = cv2.imread(img_path)
    label = cv2.imread(label_path)
    img_name = os.path.basename(img_path).split('.')[0]
    label_name = os.path.basename(label_path).split('.')[0]
    cv2.imwrite(img_aug_path + img_name + '_0.png', img)
    cv2.imwrite(label_aug_path + label_name + '_0.png', label)
    
    #mirror_vertical 
    img_1 = cv2.flip(img,0)
    label_1 = cv2.flip(label,0)
    cv2.imwrite(img_aug_path + img_name + '_1.png', img_1)
    cv2.imwrite(label_aug_path + label_name + '_1.png', label_1)
    
    #mirror_vertical again
    img_2 = cv2.flip(img_1,1)
    label_2 = cv2.flip(label_1,1)
    cv2.imwrite(img_aug_path + img_name + '_2.png', img_2)
    cv2.imwrite(label_aug_path + label_name + '_2.png', label_2)

    #mirror_horizontal
    img_3 = cv2.flip(img,1)
    label_3 = cv2.flip(label,1)
    cv2.imwrite(img_aug_path + img_name + '_3.png', img_3)
    cv2.imwrite(label_aug_path + label_name + '_3.png', label_3)
        
    #Random contrast
    img_4= random_contrast(img, label)
    cv2.imwrite(img_aug_path + img_name + '_4.png', img_4)
    cv2.imwrite(label_aug_path + label_name + '_4.png', label)

    #Random noise
    img_5 = random_noise(img, max_noise = 0.2)
    cv2.imwrite(img_aug_path + img_name + '_5.png', img_5)
    cv2.imwrite(label_aug_path + label_name + '_5.png', label)
    
def data_augmentation(ROOT_DIR, TARGET_DIR):
    """
    Function: apply data augmentation on a given dataset
    
    Input:
    ROOT_DIR : path of the directory contraining image and label folders
    TARGET_DIR : path of the target directory contraining augmented image and label folders
    """
    Train_set = ROOT_DIR
    Train_set_aug = TARGET_DIR

    if not os.path.exists(Train_set_aug):           
        os.mkdir(Train_set_aug)
        os.mkdir(Train_set_aug + 'Images/')
        os.mkdir(Train_set_aug + 'Labels/')
    list_dir_image = os.listdir(Train_set + 'Images/')
    list_dir_label = os.listdir(Train_set + 'Labels/')

    for i in tqdm(range(len(list_dir_image))):
        image_path = os.path.join(Train_set, 'Images/', list_dir_image[i])
        label_path = os.path.join(Train_set, 'Labels/', list_dir_label[i])
        augmentation(image_path, label_path, Train_set_aug + 'Images/', Train_set_aug + 'Labels/')
    
    print('Train_aug : ', len(os.listdir(TARGET_DIR + '/Images')))
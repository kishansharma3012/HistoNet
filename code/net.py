import lasagne
import theano
import theano.tensor as T 
from lasagne.layers.normalization import BatchNormLayer
from lasagne.layers import InputLayer, ConcatLayer, Conv2DLayer
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax
from lasagne.regularization import regularize_network_params, l2

def HistoNet(num_bins, patch_size = 32):
    """
    Function: Creating HistoNet network 
    
    Input:
    num_bins : size of histogram vector (2, 4, 8, 16 bins)
    patch_size: the size of patch (based on receptive field of network) for predicting redundant count map

    Output:
    net_count: output redundant count map
    net_hist: output histogram
    input_var: placeholder for image
    input_var_ex: placeholder for image index (batches) 
    """
    
    input_var = T.tensor4('inputs')
    input_var_ex = T.ivector('input_var_ex')

    input_shape = (None, 3, 256, 256)
    img = InputLayer(shape=input_shape, input_var=input_var[input_var_ex])
    net = img

    net1 = ConvFactory(net, filter_size=3, num_filter=64, pad=patch_size)
    print(net1.output_shape)
    net2 = SimpleFactory(net1, 16, 16)
    print(net2.output_shape)
    net3 = SimpleFactory(net2, 16, 32)
    print(net3.output_shape)
    net4 = ConvFactory(net3, filter_size=14, num_filter=16) 
    print(net4.output_shape)
    net5 = SimpleFactory(net4, 112, 48)
    print(net5.output_shape)
    net6 = SimpleFactory(net5, 64, 32)
    print(net6.output_shape)
    net7 = SimpleFactory(net6, 40, 40)
    print(net7.output_shape)
    net8 = SimpleFactory(net7, 32, 96)
    print(net8.output_shape)
    net9 = ConvFactory(net8, filter_size=18, num_filter=32) 
    print(net9.output_shape)
    net10 = ConvFactory(net9, filter_size=1, pad=0, num_filter=64)
    print(net10.output_shape)
    net11 = ConvFactory(net10, filter_size=1, pad=0, num_filter=64)
    print(net11.output_shape)

    print("\n Resnet Hist Head")
    _ , net_hist = resnet50(net1, num_bins)
    
    print("\n Count Head")
    net_count = ConvFactory(net11, filter_size=1, num_filter=1)
    print(net_count.output_shape)

    return net_count, net_hist, input_var, input_var_ex

def HistoNet_DSN(num_bins = [2,4,8], patch_size = 32):
    """
    Function: Creating HistoNet deep supervised network 
    
    Input:
    num_bins : list of output size of histogram vector [Early layer histogram output, middle layer histogram output, final histogram output]
    patch_size: the size of patch (based on receptive field of network) for predicting redundant count map

    Output:
    net_count: output redundant count map
    net_hist: output histogram of size num_bins[2]
    net_hist_dsn1: output histogram of size num_bins[0]
    net_hist_dsn2: output histogram of size num_bins[1]
    input_var: placeholder for image
    input_var_ex: placeholder for image index (batches) 
    """
    input_var = T.tensor4('inputs')
    input_var_ex = T.ivector('input_var_ex')

    input_shape = (None,3 , 256, 256)
    img = InputLayer(shape=input_shape, input_var=input_var[input_var_ex])
    net = img

    net1 = ConvFactory(net, filter_size=3, num_filter=64, pad=patch_size)
    print(net1.output_shape)
    net2 = SimpleFactory(net1, 16, 16)
    print(net2.output_shape)
    net3 = SimpleFactory(net2, 16, 32)
    print(net3.output_shape)
    net4 = ConvFactory(net3, filter_size=14, num_filter=16) 
    print(net4.output_shape)
    net5 = SimpleFactory(net4, 112, 48)
    print(net5.output_shape)
    net6 = SimpleFactory(net5, 64, 32)
    print(net6.output_shape)
    net7 = SimpleFactory(net6, 40, 40)
    print(net7.output_shape)
    net8 = SimpleFactory(net7, 32, 96)
    print(net8.output_shape)
    net9 = ConvFactory(net8, filter_size=18, num_filter=32) 
    print(net9.output_shape)
    net10 = ConvFactory(net9, filter_size=1, pad=0, num_filter=64)
    print(net10.output_shape)
    net11 = ConvFactory(net10, filter_size=1, pad=0, num_filter=64)
    print(net11.output_shape)

    print("\n Resnet Hist Head")
    _ , net_hist, net_hist_dsn1, net_hist_dsn2 = resnet50_DSN(net1, num_bins)    

    print("\n Count Head")
    net_count = ConvFactory(net11, filter_size=1, num_filter=1)
    print(net_count.output_shape)

    return net_count, net_hist, net_hist_dsn1, net_hist_dsn2, input_var, input_var_ex

def resnet50(net1, num_bins):
    """
    Function: Creating modified ResNet50 Layer deep residual network 
    
    Input:
    net1 : output after first layer of HistoNet
    num_bins : size of histogram vector (2, 4, 8, 16 bins)

    Output:
    net: Dictonary containing output at various level of ResNet 50 network
    net['histogram']: output histogram of size num_bins
    """
    net = {}
    net['input'] = net1 
    sub_net, parent_layer_name = build_simple_block(
        net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
        64, 7, 2, 3, use_bias=True)
    net.update(sub_net)
    print(sub_net[parent_layer_name].output_shape)
    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False)
    print(net['pool1'].output_shape)
    block_size = list('abc')
    parent_layer_name = 'pool1'
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='2%s' % c)
        net.update(sub_net)
    print(sub_net[parent_layer_name].output_shape)
    block_size = list('abcd')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='3%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='3%s' % c)
        net.update(sub_net)
    print(sub_net[parent_layer_name].output_shape)
    block_size = list('abcdef')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='4%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='4%s' % c)
        net.update(sub_net)
    print(sub_net[parent_layer_name].output_shape)
    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='5%s' % c)
        net.update(sub_net)
    
    print(sub_net[parent_layer_name].output_shape)
    net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0,
                             mode='average_exc_pad', ignore_border=False)
    print(net['pool5'].output_shape)

    # Modified end layers of ResNet50 for predicting size distribution histogram
    conv1 = ConvFactory(net['pool5'], filter_size=3, stride = 1, num_filter=256, pad=1)
    print(conv1.output_shape)
    conv2 = ConvFactory(conv1, filter_size=1, stride = 1, num_filter=16, pad=0)
    print(conv2.output_shape)
    fc1 = lasagne.layers.FlattenLayer(conv2)
    print(fc1.output_shape)
    fc1_drop = lasagne.layers.DropoutLayer(fc1, p= 0.2)
    net['fc128'] = DenseLayer(fc1_drop, num_units=128, nonlinearity= rectify)
    print(net['fc128'].output_shape)
    fc2_drop = lasagne.layers.DropoutLayer(net['fc128'], p= 0.4)
    net['hist'] = DenseLayer(fc2_drop, num_units=num_bins, nonlinearity= rectify)
    print(net['hist'].output_shape)
    return net, net['hist']

def resnet50_DSN(net1, num_bins):
    """
    Function: Creating modified ResNet50 Layer deep residual network 
    
    Input:
    net1 : output after first layer of HistoNet
    num_bins : list of output size of histogram vector [Early layer histogram output, middle layer histogram output, final histogram output]

    Output:
    net: Dictonary containing output at various level of ResNet 50 network
    net['histogram']: output histogram of size num_bins[2]
    dsn_1: output histogram of size num_bins[0]
    dsn_2: output histogram of size num_bins[1]
    """
    net = {}
    net['input'] = net1 
    sub_net, parent_layer_name = build_simple_block(
        net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
        64, 7, 2, 3, use_bias=True)
    net.update(sub_net)
    print(sub_net[parent_layer_name].output_shape)
    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False)
    print(net['pool1'].output_shape)
    block_size = list('abc')
    parent_layer_name = 'pool1'
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='2%s' % c)
        net.update(sub_net)
    print(sub_net[parent_layer_name].output_shape)
    block_size = list('abcd')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='3%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='3%s' % c)
        net.update(sub_net)
    print(sub_net[parent_layer_name].output_shape)
    dsn_1 = deep_supervision1(net[parent_layer_name], num_bins[0])
    print("DSN 1 : ",dsn_1.output_shape)
    block_size = list('abcdef')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='4%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='4%s' % c)
        net.update(sub_net)
    print(sub_net[parent_layer_name].output_shape)
    dsn_2 = deep_supervision2(net[parent_layer_name], num_bins[1])
    print("DSN 2 : ",dsn_2.output_shape)
    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='5%s' % c)
        net.update(sub_net)
    
    print(sub_net[parent_layer_name].output_shape)
    net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0,
                             mode='average_exc_pad', ignore_border=False)
    print(net['pool5'].output_shape)

    # Modified end layers of ResNet50 for predicting size distribution histogram
    conv1 = ConvFactory(net['pool5'], filter_size=3, stride = 1, num_filter=256, pad=1)
    print(conv1.output_shape)
    conv2 = ConvFactory(conv1, filter_size=1, stride = 1, num_filter=16, pad=0)
    print(conv2.output_shape)
    fc1 = lasagne.layers.FlattenLayer(conv2)
    print(fc1.output_shape)
    fc1_drop = lasagne.layers.DropoutLayer(fc1, p= 0.2)
    net['fc128'] = DenseLayer(fc1_drop, num_units=128, nonlinearity= rectify)
    print(net['fc128'].output_shape)
    fc2_drop = lasagne.layers.DropoutLayer(net['fc128'], p= 0.4)
    net['hist'] = DenseLayer(fc2_drop, num_units=num_bins[2], nonlinearity= rectify)
    print(net['hist'].output_shape)
    return net, net['hist'], dsn_1, dsn_2

###################################################################
#                       Network Utils                             #
###################################################################

def build_simple_block(incoming_layer, names,
                       num_filters, filter_size, stride, pad,
                       use_bias=False, nonlin=rectify):
    """
    Function : Creates stacked Lasagne layers ConvLayer -> BN -> (ReLu) for resnet 50
    
    Input:
    incoming_layer : instance of Lasagne layer
        Parent layer
    names : list of string
        Names of the layers in block
    num_filters : int
        Number of filters in convolution layer
    filter_size : int
        Size of filters in convolution layer
    stride : int
        Stride of convolution layer
    pad : int
        Padding of convolution layer
    use_bias : bool
        Whether to use bias in conlovution layer
    nonlin : function
        Nonlinearity type of Nonlinearity layer
    
    Ouput:
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    net = []
    net.append((
            names[0],
            ConvLayer(incoming_layer, num_filters, filter_size, stride, pad,
                      flip_filters=False, nonlinearity=None) if use_bias
            else ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None,
                           flip_filters=False, nonlinearity=None)
        ))

    net.append((
            names[1],
            BatchNormLayer(net[-1][1])
        ))
    if nonlin is not None:
        net.append((
            names[2],
            NonlinearityLayer(net[-1][1], nonlinearity=nonlin)
        ))

    return dict(net), net[-1][0]

def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                         upscale_factor=4, ix=''):
    """
    Function: Creates two-branch residual block
    
    Input:
    incoming_layer : instance of Lasagne layer
        Parent layer
    ratio_n_filter : float
        Scale factor of filter bank at the input of residual block
    ratio_size : float
        Scale factor of filter size
    has_left_branch : bool
        if True, then left branch contains simple block
    upscale_factor : float
        Scale factor of filter bank at the output of residual block
    ix : int
        Id of residual block
    
    Output:
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']

    net = {}

    # right branch
    net_tmp, last_layer_name = build_simple_block(
        incoming_layer, list(map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern)),
        int(lasagne.layers.get_output_shape(incoming_layer)[1]*ratio_n_filter), 1, int(1.0/ratio_size), 0)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], list(map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern)),
        lasagne.layers.get_output_shape(net[last_layer_name])[1], 3, 1, 1)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], list(map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern)),
        lasagne.layers.get_output_shape(net[last_layer_name])[1]*upscale_factor, 1, 1, 0,
        nonlin=None)
    net.update(net_tmp)

    right_tail = net[last_layer_name]
    left_tail = incoming_layer

    # left branch
    if has_left_branch:
        net_tmp, last_layer_name = build_simple_block(
            incoming_layer, list(map(lambda s: s % (ix, 1, ''), simple_block_name_pattern)),
            int(lasagne.layers.get_output_shape(incoming_layer)[1]*4*ratio_n_filter), 1, int(1.0/ratio_size), 0,
            nonlin=None)
        net.update(net_tmp)
        left_tail = net[last_layer_name]

    net['res%s' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1)
    net['res%s_relu' % ix] = NonlinearityLayer(net['res%s' % ix], nonlinearity=rectify)

    return net, 'res%s_relu' % ix

def ConvFactory(data, num_filter, filter_size, stride=1, pad=0, nonlinearity=lasagne.nonlinearities.leaky_rectify):
    data = lasagne.layers.batch_norm(Conv2DLayer(
        data, num_filters=num_filter,
        filter_size=filter_size,
        stride=stride, pad=pad,
        nonlinearity=nonlinearity,
        W=lasagne.init.GlorotUniform(gain='relu')))
    return data

def SimpleFactory(data, ch_1x1, ch_3x3):
    conv1x1 = ConvFactory(data=data, filter_size=1, pad=0, num_filter=ch_1x1)
    conv3x3 = ConvFactory(data=data, filter_size=3, pad=1, num_filter=ch_3x3) 
    concat = ConcatLayer([conv1x1, conv3x3])
    return concat

def deep_supervision1(net, num_bins):
    print("-DSN1-")
    pool1 = PoolLayer(net, pool_size=7, stride=1, pad=0,
                             mode='average_exc_pad', ignore_border=False)
    print(pool1.output_shape)
    conv1 = ConvFactory(pool1, filter_size=3, stride = 1, num_filter=256, pad=1)
    print(conv1.output_shape)
    conv2 = ConvFactory(conv1, filter_size=1, stride = 1, num_filter=16, pad=0)
    print(conv2.output_shape)
    fc1 = lasagne.layers.FlattenLayer(conv2)
    print(fc1.output_shape)
    fc1_drop = lasagne.layers.DropoutLayer(fc1, p= 0.2)
    fc2 = lasagne.layers.DenseLayer(fc1_drop, 128, nonlinearity=lasagne.nonlinearities.rectify)
    print(fc2.output_shape)
    fc2_drop = lasagne.layers.DropoutLayer(fc2, p= 0.4)
    fc3 = lasagne.layers.DenseLayer(fc2_drop, num_bins , nonlinearity=lasagne.nonlinearities.rectify)
    print(fc3.output_shape)
    return fc3

def deep_supervision2(net, num_bins):
    print("-DSN2-")
    pool1 = PoolLayer(net, pool_size=7, stride=1, pad=0,
                             mode='average_exc_pad', ignore_border=False)
    print(pool1.output_shape)
    conv1 = ConvFactory(pool1, filter_size=3, stride = 1, num_filter=256, pad=1)
    print(conv1.output_shape)
    conv2 = ConvFactory(conv1, filter_size=1, stride = 1, num_filter=16, pad=0)
    print(conv2.output_shape)
    fc1 = lasagne.layers.FlattenLayer(conv2)
    print(fc1.output_shape)
    fc1_drop = lasagne.layers.DropoutLayer(fc1, p= 0.2)
    fc2 = lasagne.layers.DenseLayer(fc1_drop, 128, nonlinearity=lasagne.nonlinearities.rectify)
    print(fc2.output_shape)
    fc2_drop = lasagne.layers.DropoutLayer(fc2, p= 0.4)
    fc3 = lasagne.layers.DenseLayer(fc2_drop, num_bins, nonlinearity=lasagne.nonlinearities.rectify)
    print(fc3.output_shape)
    return fc3

from paddle.trainer_config_helpers import *

DATA_PRE_HOUR = 12
DATA_PRE_DAY = 24 * DATA_PRE_HOUR
############################################## Parameters ############################################

TRAIN_DAY = 1
TEST_HOUR = 2
CROSS_VAL_DAY = 1
emb_size = 16
hid_lr = 0.05

######################################################################################################
TRAIN_NUM=24
hid_size = TRAIN_NUM*2+1
#TRAIN_NUM = DATA_PRE_DAY * TRAIN_DAY
TEST_NUM = DATA_PRE_HOUR * TEST_HOUR
CROSS_VALIDATION = DATA_PRE_DAY * CROSS_VAL_DAY

############################################## Data Configuration #####################################

is_predict = get_config_arg('is_predict', bool, False)
trn = './data/train.list' if not is_predict else None
tst = './data/test.list' if not is_predict else './data/pred.list'
process = 'process' if not is_predict else 'process_predict'
define_py_data_sources2(
    train_list=trn, test_list=tst, module="dataprovider", obj=process)
batch_size = 1280 if not is_predict else 1

############################################# Network Settings ########################################

settings(
    batch_size=batch_size,
    learning_rate=1e-2, #changed from 2e-3
    learning_method=AdamOptimizer(),
    regularization=L2Regularization(8e-4),
    gradient_clipping_threshold=25)

layer_attr = ExtraLayerAttribute(drop_rate=0.5)
fc_para_attr = ParameterAttribute(learning_rate=hid_lr)
lstm_para_attr = ParameterAttribute(initial_std=0., learning_rate=0.05)
para_attr = [fc_para_attr, lstm_para_attr]
bias_attr = ParameterAttribute(initial_std=0., l2_rate=0.)
relu = ReluActivation()
linear = LinearActivation()



output_label = []
data = data_layer(name='data', size=TRAIN_NUM)
emb = embedding_layer(input=data, size=2)

for i in xrange(TEST_NUM):

    #fc1 = fc_layer(input=data,size=7,act=linear)
    #lstm1 = lstmemory(input=fc1,reverse=0,act=relu,bias_attr=bias_attr,layer_attr=layer_attr)
    #input1=fc1+lstm1
    #fc2 = fc_layer(input=input1,size=7,act=linear,param_attr=para_attr,bias_attr=bias_attr)
    #lstm2 = lstmemory(input=fc2,reverse=1,act=relu,bias_attr=bias_attr,layer_attr=layer_attr)

    fc1 =fc_layer(input = emb,size = hid_size ,act = ReluActivation())    
    lstm1 = simple_lstm(
        input=fc1, size=hid_size, lstm_cell_attr=ExtraAttr(drop_rate=0.25))
    input1 = [fc1,lstm1]   
    
    fc2 = fc_layer(input = input1 ,size = hid_size, act = ReluActivation())
    
    lstm2 = simple_lstm(
        input=fc2, size=hid_size, lstm_cell_attr=ExtraAttr(drop_rate=0.25))

    fc_max = pooling_layer(input = fc2,pooling_type=MaxPooling())    
  
    lstm_max = pooling_layer(input=lstm1, pooling_type=MaxPooling())
    input2=[fc_max,lstm_max]
    score = fc_layer(input=input2, size=4, act=SoftmaxActivation())
    if is_predict:
        maxid = maxid_layer(score)
        output_label.append(maxid)
    else:
        label = data_layer(name='label_%dmin' % ((i + 1) * 5), size=4)
        cls = classification_cost(
            input=score, name="cost_%dmin" % ((i + 1) * 5), label=label)
        output_label.append(cls)
outputs(output_label)

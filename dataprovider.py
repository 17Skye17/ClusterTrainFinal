from paddle.trainer.PyDataProvider2 import *
import sys
import numpy as np
#import collections
DATA_PRE_HOUR = 12
DATA_PRE_DAY = 24 * DATA_PRE_HOUR
############################################## Parameters ############################################

TRAIN_DAY = 1
TEST_HOUR = 2
CROSS_VAL_DAY = 1

######################################################################################################
TRAIN_NUM=24
#TRAIN_NUM = DATA_PRE_DAY * TRAIN_DAY
TEST_NUM = DATA_PRE_HOUR * TEST_HOUR
CROSS_VALIDATION = DATA_PRE_DAY * CROSS_VAL_DAY

def initHook(settings, file_list, **kwargs):
    del kwargs  #unused 

    settings.pool_size = sys.maxint
  #  dic = collections.OrderedDict()
    settings.input_types = {'data':integer_value_sequence(4)}
    for i in range(1,TEST_NUM+1,1):
        settings.input_types['label_'+str(i * 5)+'min'] = integer_value(4)
  #      dic[] = integer_value(4)

@provider(
    init_hook=initHook, cache=CacheType.CACHE_PASS_IN_MEM, should_shuffle=True)
def process(settings, file_name):
    with open(file_name) as f:
        #abandon fields name
        f.next()
        for row_num, line in enumerate(f):
            speeds = map(int, line.rstrip('\r\n').split(",")[1:])
            # Get the max index.
            end_time = len(speeds) - CROSS_VALIDATION
          #  end_time = 2880
            # Scanning and generating samples
            for i in range(TRAIN_NUM, end_time - TRAIN_NUM,96):  #2hour step
                # For dense slot
                pre_spd = map(int, speeds[i - TRAIN_NUM:i])

                # Integer value need predicting, values start from 0, so every one minus 1.
                fol_spd = [j - 1 for j in speeds[i:i + TEST_NUM]]

                # Predicting label is missing, abandon the sample.
                if -1 in fol_spd:
                    continue
 #               yield {'data':pre_spd,'label': fol_spd}
                dic = {}
                dic['data'] = pre_spd
                for i in range(1,TEST_NUM+1,1):
                    dic['label_'+str(i*5)+'min'] = fol_spd[i-1]   #start from 0
                yield dic

def predict_initHook(settings, file_list, **kwargs):
    settings.pool_size = sys.maxint
    settings.input_types = {'data': integer_value_sequence(TRAIN_NUM)}

@provider(init_hook=predict_initHook, should_shuffle=False)
def process_predict(settings, file_name):
    with open(file_name) as f:
        #abandon fields name
        f.next()
        for row_num, line in enumerate(f):
            speeds = map(int, line.rstrip('\r\n').split(","))
            end_time = len(speeds) - CROSS_VALIDATION
            pre_spd = map(int, speeds[end_time - TRAIN_NUM:end_time])
            yield {'data':pre_spd}

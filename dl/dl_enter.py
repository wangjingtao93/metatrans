import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')


import os
import argparse
import time
import csv

from common.dataloader import *
import common.utils as utils

import dl.dl_func as dlf


def dl_enter(args, test_data):

    dlf.trainer(args,test_data[0], test_data[1], test_data[2], 0)

def predict_enter(args, test_data, meta_epoch_for_predict):
    if args.isheatmap:
        dlf.heat_map(args, meta_epoch_for_predict)
        
    else:
        dlf.predict(args, test_data[2], meta_epoch_for_predict)
        # dlf.predict(args, test_data[1], meta_epoch_for_predict, setname='val')

        # dlf.predict_tmp(args, meta_epoch_for_predict)



    

    
    

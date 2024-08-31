
# 加载元训练参数，进行dl测试
import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')

""" Generate commands for test. """
import os

def run_exp():
    gpu = 3
    
    load = ''
    load_interrupt_path = ''

    prefix = 'dl'
    save_path = 'result_20230908_sub10' 
    
    remark='dl_sub'
    n_epoch = 30

    description_name = '临时测试'

      
    the_command = 'python3 ../main_test.py --gpu=' + str(gpu) \
        + ' --prefix=' + prefix\
        + ' --load=' + load \
        + ' --load_interrupt_path=' + load_interrupt_path \
        + ' --save_path=' + save_path \
        + ' --test_meta_size=' + str(1) \
        + ' --n_test_tasks=' + str(1) \
        + ' --remark=' + remark \
        + ' --n_epoch=' + str(n_epoch) \
        + ' --description_name=' + description_name \
        
    os.system(the_command)

run_exp()


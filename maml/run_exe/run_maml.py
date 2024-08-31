
# 加载预训练参数，进行元训练
import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')

""" Generate commands for test. """
import os

def run_exp():
    gpu = 4
    
    load = ''
    load_interrupt_path = ''

    prefix = 'maml'
    save_path = 'result_20230710' 

    n_train_task = 2000 #2000用于跑模型
    n_val_task = 100

    the_command = 'python3 ../main.py --gpu=' + str(gpu) \
        + ' --prefix=' + prefix\
        + ' --load=' + load \
        + ' --load_interrupt_path=' + load_interrupt_path \
        + ' --save_path=' + save_path \
        + ' --meta_size=' + str(5) \
        + ' --test_meta_size=' + str(1) \
        + ' --n_inner=' + str(5) \
        + ' --n_train_tasks=' + str(n_train_task) \
        + ' --n_val_task=' + str(n_val_task) \
        + ' --n_test_tasks=' + str(1) \
        + ' --test_k_shot=' + str(120000) \
        + ' --test_k_qry=' + str(400)
    os.system(the_command)

run_exp()


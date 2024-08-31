
# 加载预训练参数，进行元训练
import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')

""" Generate commands for test. """
import os

def run_exp():
    gpu = 2
    
    load = ''
    load_interrupt_path = ''

    prefix = 'maml'
    save_path = 'result_20230908_sub10' 

    n_train_tasks = 200 #2000用于跑模型

    description_name = '不加载预训练15_shot_5_query'

    the_command = 'python3 ../main.py --gpu=' + str(gpu) \
        + ' --prefix=' + prefix\
        + ' --load=' + load \
        + ' --load_interrupt_path=' + load_interrupt_path \
        + ' --save_path=' + save_path \
        + ' --meta_size=' + str(5) \
        + ' --test_meta_size=' + str(1) \
        + ' --n_way=' + str(4) \
        + ' --n_inner=' + str(5) \
        + ' --n_train_tasks=' + str(n_train_tasks) \
        + ' --k_shot=' + str(15) \
        + ' --k_qry=' + str(5) \
        + ' --n_test_tasks=' + str(1) \
        + ' --description_name=' + description_name \
        + ' --lr_sched=True' \
        
    os.system(the_command)

run_exp()



# 加载预训练参数，进行元训练
import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')

""" Generate commands for test. """
import os

def run_exp():
    gpu = 4
    
    load = ''
    load_interrupt_path = ''

    # prefix = 'imaml'
    save_path = 'result_20230908_sub10' 

    n_train_task = 200 #2000用于跑模型

    description_name = 'outer_lr_lr_sched'
    description = '未加载预训练'
    n_epoch = 20
    outer_lr = 1e-4
    

    the_command = 'python3 ../main.py --gpu=' + str(gpu) \
        + ' --load=' + load \
        + ' --load_interrupt_path=' + load_interrupt_path \
        + ' --save_path=' + save_path \
        + ' --meta_size=' + str(5) \
        + ' --test_meta_size=' + str(1) \
        + ' --n_way=' + str(4) \
        + ' --n_inner=' + str(10) \
        + ' --n_train_tasks=' + str(n_train_task) \
        + ' --k_shot=' + str(10) \
        + ' --k_qry=' + str(10) \
        + ' --n_test_tasks=' + str(1) \
        + ' --version=GD'\
        + ' --cg_steps=' + str(5) \
        + ' --outer_opt=Adam' \
        + ' --lambda=' + str(2) \
        + ' --description_name=' + description_name \
        + ' --description=' + description \
        + ' --n_epoch=' + str(n_epoch) \
        + ' --outer_lr=1e-4' \
        + ' --lr_sched=true'
        
    os.system(the_command)

run_exp()


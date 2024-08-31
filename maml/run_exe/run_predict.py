
# 加载预训练参数，进行元训练
import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')

""" Generate commands for test. """
import os

def run_exp():
    gpu = 1
    
    load = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/maml/result_20230703/resnet18/dl/2023-07-04-20-21-34/best_model_2023-07-04-20-21-34.pth'
    load_interrupt_path = ''

    prefix = 'maml'
    save_path = 'result_20230703' 
      
    the_command = 'python3 ../predict.py --gpu=' + str(gpu) \
        + ' --prefix=' + prefix\
        + ' --load=' + load \
        + ' --load_interrupt_path=' + load_interrupt_path \
        + ' --save_path=' + save_path \
        + ' --meta_size=' + str(5) \
        + ' --test_meta_size=' + str(1) \
        + ' --n_inner=' + str(2) \
        + ' --n_train_task=' + str(10) \
        + ' --n_val_task=' + str(10) \
        + ' --n_test_task=' + str(1) \
        + ' --test_k_shot=' + str(15000) \
        + ' --test_k_qry=' + str(400)
    os.system(the_command)

run_exp()


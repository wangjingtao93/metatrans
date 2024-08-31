import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/github/metatrans')
import yaml
import os

""" Generate commands for test. """

def gen_args():
    # description_name = 'tic'
    project_path = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/github/metatrans/github/metatrans'
    relative_path = project_path.replace('pycharm_remote/', 'pycharm_remote/result/')
    save_path = os.path.join(relative_path, 'result/result_202405028/' )
    # save_path = os.path.join(relative_path, 'result/tmp/' )
    gpu = 0

    data_path_prefix="/data1/wangjingtao/workplace/python/data/classification/lymph/"
    data_type_ls = {0:'throid_4x', 1:'throid_10x', 2:'multi_centers_ffpe_4x',3:'multi_centers_ffpe_10x',4:'multi_centers_bf_4x',5:'multi_centers_bf_10x', 6:'camely_16_4x', 7:'camely_16_10x', 8:'camely_17_4x', 9:'camely_17_10x', 10:'camely_all_4x', 11:'camely_all_10x', 12:'crc-4x', 13:'crc-10x', 14:'background_10x'}
    datatype = data_type_ls[3]

    algs = {0:'dl', 1:'pretrain', 2:'imaml', 3:'maml', 4:'reptile', 5:'predict/dl', 6:'predict/imaml', 7:'predict/maml', 8:'predict/reptile'}
    alg = algs[5]

    model_names = {0:'alexnet', 1:'squeezenet1_0', 2:'squeezenet1_1', 3:'densenet121',4:'densenet169', 5:'densenet201', 6:'densenet201', 7:'densenet161',8:'vgg11', 9:'vgg11_bn', 10:'vgg13', 11:'vgg13_bn', 12:'vgg16', 13:'vgg16_bn',14:'vgg19', 15:'vgg19_bn', 16:'resnet18', 17:'resnet34', 18:'resnet50', 19:'resnet101',20:'resnet152', 21:'convnet_4', 22:'vit_base_patch16_224', 23:'vit_base_patch16_224_depth_6', 24:'vit_base_patch16_224_depth_3', 25:'vit_tiny_patch16_224', 26:'vit_small_patch16_224', 27:'mt_tiny_model_lymph', 29:'mtb', 30:'mtb_res', 31:'mtb_6b_mfc', 32:'metatrans'}
    # 使用了 1 16 17 18 21 22 26 32
    net = model_names[16]

    with open('configs.yaml', 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)

    load = ''
    load_interrupt_path = ''

    args_dict = {}

    # base
    args_dict['gpu'] = [gpu]
    args_dict['save_path'] = save_path
    # args_dict['description_name'] = description_name
   
    args_dict['acc_average'] = 'weighted' # weighted or macro

    # data
    args_dict['datatype'] = datatype
    args_dict['resize'] = 224
    args_dict['data_path_prefix'] = data_path_prefix
    # args_dict['predict_class'] = ['ffpe_itc', 'ffpe_normal']
    # args_dict['predict_class'] = ['ffpe_macro', 'ffpe_normal']
    # args_dict['predict_class'] = ['ffpe_micro', 'ffpe_normal']

    # multi_centers
    args_dict['multi_centers_test'] = '' # jingxia, throid, crc

    # predict
    args_dict['isheatmap'] = True
    

    # net 
    args_dict['alg'] = alg
    args_dict['net'] = net
    args_dict['num_classes'] = 2

    #net vit

    #net mt
    args_dict['trans_depth'] = 6

    # net load
    args_dict['is_load_imagenet'] = False
    args_dict['is_meta_load_imagenet'] = False
    # args_dict['load'] = load
    args_dict['load_interrupt_path'] = load_interrupt_path
    args_dict['is_lock_notmeta'] = True
    # args_dict['meta_epoch_for_predict']

    # dl
    args_dict['n_epoch'] = result[alg]['n_epoch'] # 用于meta test 或dl train
    args_dict['batch_size_train'] = 64
    args_dict['batch_size_val'] = 64
    args_dict['batch_size_test'] = 16
    args_dict['is_save_val_net'] = True # 最好不用用字符串，尤其是'false', 都会当做true

    

    # meta
    args_dict['n_meta_epoch'] = 30
    args_dict['meta_size'] = 5
    args_dict['outer_lr'] = result[alg]['outer_lr']
    args_dict['inner_lr'] = result[alg]['inner_lr']
    args_dict['n_train_tasks'] = 500
    args_dict['n_val_tasks'] = 5
    args_dict['n_test_tasks'] = 5
    args_dict['test_meta_size'] = 1
    args_dict['n_way'] = 2
    args_dict['n_inner'] = result[alg]['n_inner']
    args_dict['k_shot'] = 5
    args_dict['k_qry'] =  5

    # imaml
    args_dict['version'] = 'GD'
    args_dict['cg_steps'] = 5
    args_dict['outer_opt'] = 'Adam'
    args_dict['lambda'] = 2
    args_dict['lr_sched'] = True


    args_dict['description_name'] = 'crc_acc'

    if 'predict' in args_dict['alg']:
        args_dict = set_predict_load(args_dict, result)

    check_error(args_dict)

    return args_dict
    
    
    # args_dict['use_data_percent'] = 1.0
    # args_dict['index_fold'] = index_fold
    # args_dict['n_channels'] = result[datatype]['n_channels']
    # args_dict['dl_lr'] = 0.01   
    # args_dict['meta_learner_load'] = meta_learner_load
    # args_dict['is_mid_val'] = False
    # args_dict['n_mid_val'] = 5
    # args_dict['trans_depth'] = 12
    # args_dict['meta_resize'] = 256
    # args_dict['n_val_tasks'] = 50
    # args_dict['test_meta_size'] = 1
    # args_dict['real_data_csv'] = result[datatype]['real_data_csv']
    # args_dict['synthetic_data_csv'] = result[datatype]['synthetic_data_csv']
    # args_dict['train_csv'] = result[datatype]['train_csv']
    # args_dict['val_csv'] = result[datatype]['val_csv']
    # args_dict['test_csv'] =  result[datatype]['test_csv']

def set_predict_load(args_dict, yaml_res):
    args_dict['load'] = yaml_res['load'][args_dict['datatype']][args_dict['alg']][args_dict['net']]['load_path']

    args_dict['load']  = os.path.join('/data1/wangjingtao/workplace/python/pycharm_remote/result',args_dict['load'])

    args_dict['meta_epoch_for_predict'] = yaml_res['load'][args_dict['datatype']][args_dict['alg']][args_dict['net']]['meta_epoch_for_predict']

    return args_dict

    
def run_command(args_dict):

    command_str = ''
    for key, values in args_dict.items():
        command_str += ' --' + key + '=' + str(values)

    the_command = 'python ../main.py --is_run_command=True' + command_str

    os.system(the_command)

def check_error(args_dict):
    
    # predict, predict load path check
    if 'predict' in args_dict['alg'] or 'heatmap' in args_dict['alg']:
        if args_dict['load'] == '':
            raise ValueError('predict mode load path must been exist')
        load_values = args_dict['load'].split('/')
        datatpye = load_values[10]
        if args_dict['datatype'].split('_')[-1] != datatpye.split('_')[-1]:
            raise ValueError('Resolution is not correspond')
        net = load_values[12]
        if args_dict['net'] != net:
            raise ValueError(f'net is not correspond and {net}')
        
        # meta
        alg = load_values[11]
        if args_dict['alg'].split('/')[-1] != alg:
            raise ValueError('meta load is not correspond')
        
        if 'dl' in args_dict['alg']:
            if args_dict['meta_epoch_for_predict'] != 0:
                raise ValueError('predict/dl meta_epoch_for_predict must be 0')



if __name__ == '__main__':
    
    args_dict = run_command()

    run_command(args_dict)

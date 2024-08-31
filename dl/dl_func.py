import os
import csv
from copy import deepcopy
from torch.utils.data import  DataLoader
from copy import deepcopy

from common.dl_comm import dl_comm
from common.dataloader import *
from common.lymph.dataloader_ly import LY_dataset
import common.utils as utils
from common.data_enter import lymph_data
import matplotlib.pyplot as plt
import seaborn as sns
import math
import torchvision.models
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

# 测试任务包含S Q 和final test需要将S和Q重新组batch,S作为train data, Q 作为val data,
def trainer(args, sppport_all_task, query_all_task, final_test_task, meta_epoch):

    # 创建一个记录测试任务的
    dir_meta_epoch = os.path.join(args.store_dir,'meta_epoch')
    utils.mkdir(dir_meta_epoch)

    metric_dir = os.path.join(dir_meta_epoch,'metric_' + str(meta_epoch) + '.csv')
    with open(str(metric_dir), 'w') as f:
        fields = ['task_idx', 'epoch', 'loss','acc', 'auc', 'precision','recall','f1','sensi', 'spec', 'best_acc', 'best_epoch']
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)  


    task_num = len(sppport_all_task) # 多构建几个任务，结果会更准确
    # 记录所有任务每个epoch的val 的acc
    val_acc_all_epoch = [0] * args.n_epoch
    val_acc_all_task = [0] * task_num
    # test_acc_all_epoch= []

    val_values_all_task = [0] * task_num
    test_values_all_task = []
    for task_idx in range(task_num):
    # for task_idx in range(3):

        # 创建 网络 对象
        dl_ob = dl_comm(args)
        dl_ob._init_net()
        dl_ob._init_opt()
        if args.alg == 'dl' and args.load != '':
            dl_ob.model.load_state_dict(torch.load(args.load))
        elif args.alg != 'dl':
            # meta的测试
            # 先采用命名规则吧
            path = os.path.join(args.store_dir,'save_meta_pth', f'meta_epoch_{meta_epoch}.pth')
            dl_ob.model.load_state_dict(torch.load(path))


        # writer = SummaryWriter('{}/tensorboard_log/{}_meta_epoch/{}_task_id'.format(args.store_dir,meta_epoch, task_idx))

        # train_set = sppport_all_task[task_idx]
        # val_set = query_all_task[task_idx]
        # test_set = final_test_task[task_idx]

        # train_loader = DataLoader(train_set, shuffle=True, batch_size = args.batch_size_train, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        # val_loader = DataLoader(val_set, shuffle=False, drop_last=True, batch_size = args.batch_size_val, num_workers=args.num_workers, pin_memory=True)
        # test_loader = DataLoader(test_set, shuffle=False, drop_last=True, batch_size = args.batch_size_val, num_workers=args.num_workers, pin_memory=True)
        train_loader = sppport_all_task[task_idx]
        val_loader = query_all_task[task_idx]
        test_loader = final_test_task[task_idx]


        # for val
        best_val = 0.0
        best_val_epoch = 1
        best_final_val_state_dict = deepcopy(dl_ob.model.state_dict()) # 不用deepcopy是万万不行的，否则会随dl_ob.model发生变化

        # 注意训练过程中，对test_loder进行测试，会影响最后模型的精度，因为random 的原因
        # for test
        # best_test = 0.0
        # best_epoch_for_test = 1
        
        for epoch in range(1, args.n_epoch + 1):
            dl_ob.adjust_learning_rate(epoch)
            print('task_idx--------: ', task_idx)
            train_loss = dl_ob.train(train_loader, epoch)
            val_value,_ = dl_ob.val(val_loader)
            
            # val_acc_all_epoch.append(val_value[1])
            val_acc_all_epoch[epoch-1] = val_acc_all_epoch[epoch-1] + val_value[1]

            is_val_set_best = val_value[1] > best_val
            best_val = max(best_val, val_value[1])
            if is_val_set_best:
                best_val_epoch = epoch
                val_values_all_task[task_idx] = val_value + [best_val, best_val_epoch]
                best_final_val_state_dict = deepcopy(dl_ob.model.state_dict())
                if args.is_save_val_net:
                    taskpth_store_dir = os.path.join(dir_meta_epoch, f'taskid_{task_idx}')
                    utils.mkdir(taskpth_store_dir)
                    torch.save(best_final_val_state_dict,os.path.join(taskpth_store_dir,f'best_model_for_valset_{meta_epoch}.pth' ))

            with open(str(metric_dir), 'a+') as f:
                csv_write = csv.writer(f, delimiter=',')
                data_row = [task_idx, epoch] + val_value
                data_row.append(best_val)
                data_row.append(best_val_epoch)
                csv_write.writerow(data_row)     

        dl_ob.model.load_state_dict(best_final_val_state_dict)
        test_values,_ = dl_ob.val(test_loader)

        test_values_all_task.append(test_values + [best_val, best_val_epoch])

        with open(str(metric_dir), 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            data_row = ['finaltest', ' '] + test_values
            data_row.append(best_val)
            data_row.append(best_val_epoch)
            csv_write.writerow(data_row)

        val_acc_all_task[task_idx] = best_val

    all_tasks_val_ave = np.around(np.mean(val_values_all_task, axis=0), 4).tolist()
    with open(str(metric_dir), 'a+') as f:
        csv_write = csv.writer(f, delimiter=',')
        data_row = ['Final_ave_val', ' '] + all_tasks_val_ave
        csv_write.writerow(data_row)

    all_tasks_ave = np.around(np.mean(test_values_all_task, axis=0), 4).tolist()
    with open(str(metric_dir), 'a+') as f:
        csv_write = csv.writer(f, delimiter=',')
        for task_idx, task_valuse_ls in  enumerate(test_values_all_task):
            data_row = [task_idx, ' '] + task_valuse_ls
            csv_write.writerow(data_row)
        csv_write.writerow(['Final_ave_test', ' '] + all_tasks_ave)

    
    return [round(item / task_num, 4) for item in val_acc_all_epoch], all_tasks_ave


def predict(args,final_test_task, meta_epoch, setname='test'):
    dir_meta_epoch = os.path.join(args.store_dir,'meta_epoch')
    utils.mkdir(dir_meta_epoch)

    metric_dir = os.path.join(dir_meta_epoch,f'metric_{setname}_' + str(meta_epoch) + '.csv')
    with open(str(metric_dir), 'w') as f:
        fields = ['task_idx','loss','acc', 'auc', 'precision','recall','f1', 'sensi', 'speci']
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)

    cm_dir = os.path.join(args.store_dir, 'cm', f'meta_epoch_{meta_epoch}', f'{setname}_dir')
    utils.mkdir(cm_dir)
    
    cm_file = os.path.join(cm_dir, 'cm.csv')
    with open(str(cm_file), 'w') as f:
        fields = ['task_idx','negative','postive']
        datawrite = csv.writer(f, delimiter=',')
        datawrite.writerow(fields)

    task_num = len(final_test_task) # 多构建几个任务，结果会更准确
    test_values_all_task = []
    for task_idx in range(task_num):
        dl_ob = dl_comm(args)
        dl_ob._init_net()
        dl_ob._init_opt()

        if args.load == '':
            exit('Predict Mode must give load path')
        else:
            # dl_ob.model.load_state_dict(torch.load(args.load))
            state_dict_path = os.path.join(args.load, 'meta_epoch', f'taskid_{task_idx}', f'best_model_for_valset_{meta_epoch}.pth')
            dl_ob.model.load_state_dict(torch.load(state_dict_path))

        # test_set = final_test_task[task_idx]
        # test_loader = DataLoader(test_set, shuffle=False, drop_last=True, batch_size = args.batch_size_test, num_workers=args.num_workers, pin_memory=True)
        test_loader = final_test_task[task_idx]
        
        test_values, scores = dl_ob.val(test_loader)

        test_values_all_task.append(test_values)

        # with open(str(metric_dir), 'a+') as f:
        #     csv_write = csv.writer(f, delimiter=',')
        #     data_row = [task_idx] + test_values
        #     csv_write.writerow(data_row)

        
        with open(cm_file, 'a+') as f:
            csv_write = csv.writer(f, delimiter=',')
            tn, fp, fn, tp = scores[0].ravel()
            
            csv_write.writerow([task_idx, tn, fp])
            csv_write.writerow([' ', fn, tp])

        score_file = os.path.join(cm_dir, f'predict_score_task_{task_idx}.csv')
        with open(str(score_file), 'w') as f:
            fields = ['label','negative','postive', 'nor_negative', 'nor_postive']
            datawrite = csv.writer(f, delimiter=',')
            datawrite.writerow(fields)
      
            y_score = scores[2]
            y_pro = softmax(scores[2])
            roc_len = len(scores[3][0])
            for i in range(len(scores[1])):
                datawrite.writerow([scores[1][i], y_score[i][0], y_score[i][1], y_pro[i][0], y_pro[i][1]])

        roc_dir = os.path.join(cm_dir, f'roc_task_{task_idx}.csv')
        with open(str(roc_dir), 'w') as f:
            fields = ['fpr','tpr']
            datawrite = csv.writer(f, delimiter=',')
            datawrite.writerow(fields)

            for i in range(len(scores[3][0])):
                datawrite.writerow([round(scores[3][0][i],6), round(scores[3][1][i],6)])
    
    all_tasks_ave = np.around(np.mean(test_values_all_task, axis=0), 4).tolist()
    with open(str(metric_dir), 'a+') as f:
        csv_write = csv.writer(f, delimiter=',')
        for task_idx, task_valuse_ls in  enumerate(test_values_all_task):
            data_row = [task_idx] + task_valuse_ls
            csv_write.writerow(data_row)
        csv_write.writerow(['ave'] + all_tasks_ave)
    
                
def heat_map(args,meta_epoch):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 非镜下
    # csv_file = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/multi_centers/captions/feijingxia/choice.csv'

    # 镜下
    # csv_file = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/multi_centers/captions/jinxia_captions.csv'

    # whole slide
    # csv_file = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/multi_centers/captions/whole_slide/whole_slide.csv'
    
    # throid
    # csv_file = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/thyroid/caption/forheatmap.csv'

    # crc
    csv_file = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/CRC-LN/captions/forheatmap.csv'

    relative_path = '/data1/wangjingtao/workplace/python/data/classification/lymph'

    


    dframe = pd.read_csv(csv_file)

    res = args.datatype.split('_')[-1]
    class_prefix = args.datatype.split('_')[-2]

    df_res = dframe[dframe['Resolution'] == res]

    df_class_postive = df_res[df_res['Class'] == class_prefix + '_micro']
    df_class_negative = df_res[df_res['Class'] == class_prefix + '_normal']

    df_test = pd.DataFrame()
    # df_test = pd.concat([df_class_postive, df_class_negative],ignore_index=True) 
    df_test =  pd.concat([df_class_postive],ignore_index=True) 

    task_num = 5 # 多构建几个任务，结果会更准确

    # 去除背景
    bck_model = torchvision.models.resnet18(weights=None)
    num_ftrs = bck_model.fc.in_features
    bck_model.fc = nn.Linear(num_ftrs,  2)
    bck_model.load_state_dict(torch.load('/data1/wangjingtao/workplace/python/pycharm_remote/result/meta-learning-classfication/result/result_202405028/background_10x/dl/resnet18/2024-07-24-20-18-05/meta_epoch/taskid_0/best_model_for_valset_0.pth', map_location=device))
    bck_model.to(device=device)
    bck_model.eval()
    for task_idx in range(task_num):
        # task_idx = 3 # 非镜下， 4x最好哦用task3， 10x用task4

        dl_ob = dl_comm(args)
        dl_ob._init_net()
        dl_ob._init_opt()

        if args.load == '':
            exit('Predict Mode must give load path')
        else:
            # dl_ob.model.load_state_dict(torch.load(args.load))
            state_dict_path = os.path.join(args.load, 'meta_epoch', f'taskid_{task_idx}', f'best_model_for_valset_{meta_epoch}.pth')
            dl_ob.model.load_state_dict(torch.load(state_dict_path, map_location=device))

        dl_ob.model.eval()
        for index,row in df_test.iterrows():
            img_file = row['Image_path']
            
            caption_name =  img_file.split('/')[-1].replace('.jpg', '')
            
            heat_metric_store_dir = os.path.join(args.store_dir, 'heatmap', f'meta_epoch_{meta_epoch}',row['Center'], row['Resolution'], row['Class'],f'taskid_{task_idx}')
            utils.mkdir(heat_metric_store_dir)

            store_ori_image = os.path.join(args.store_dir, 'heatmap', f'meta_epoch_{meta_epoch}',row['Center'], row['Resolution'], row['Class'],f'taskid_{task_idx}', 'ori_image')

            image = cv2.imread(os.path.join(relative_path,img_file))
            if image is None:
                raise ValueError("Failed to load the image")
            # 将图像裁剪成 patches
            patches = []
            height, width = image.shape[:2]
            
            #  for whole slide 4x
            # height = int(height * 2/3)
            # width = int(width *2/3)
            # image = cv2.resize(image, [width,height], interpolation=cv2.INTER_CUBIC)

            # for captions 
            height = math.ceil(height / 256) * 256
            width = math.ceil(width / 256) * 256
            image = cv2.resize(image, [width,height], interpolation=cv2.INTER_CUBIC)
            n_rows = 0
            for i in range(0, height, 256):
                n_rows += 1
                for j in range(0, width, 256):
                    patch = image[i:i+256, j:j+256, :]

                    patches.append(patch)
                    # patch_store =  os.path.join(heat_metric_store_dir, 'patches')
                    # cv2.imwrite(f'{patch_store}/{i}_{j}.jpg', patches)
            n_cols = int(len(patches) / n_rows)
            
            # 对每个 patch 进行预测，并记录预测概率
            pred_score_list = []
            predictions = []
            for patch in patches:
                # 预处理图像
                input_image = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                input_tensor = preprocess_image(input_image).to(device)
                # with torch.no_grad():
                #     check_bck = bck_model(input_tensor)
                with torch.no_grad():
                    output = dl_ob.model(input_tensor)
                
                
                # 使用softmax将输出转换为概率
                # bck_probability = torch.softmax(check_bck, dim=1)

                # 选择概率最大的类别
                # bck_classes = torch.argmax(bck_probability, dim=1)
                # if bck_classes == 0:
                #     # output = torch.zeros_like(output)
                #     output = torch.tensor([[1, 0]], dtype=torch.float32).to(device=device)

                pred_score_list.append(output.data.cpu().detach().numpy())
                
            y_score = np.concatenate(pred_score_list)
            predictions = softmax(y_score)


            # 将预测概率映射到原图像上
            heatmap = np.zeros_like(image[:,:,0], dtype=np.float32)
            count = 0
            for i in range(0, height, 256):
                for j in range(0, width, 256):
                    heatmap[i:i+256,j:j+256]  += predictions[count][1]  # 此处假设预测概率的第二个元素是正类的概率
                    count += 1

            # 归一化热力图
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) * 255
            heatmap = heatmap.astype(np.uint8)

            # 可视化彩色图像
            color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            
            # 放在同一张画布上
            concat_horizontal = np.hstack((color_map, image))
            # 将彩色图像保存为图像文件
            heatmap_store_path =  os.path.join(heat_metric_store_dir, f'{caption_name}.jpg')
            cv2.imwrite(heatmap_store_path, concat_horizontal)
            
            score_file = os.path.join(heat_metric_store_dir, f'{caption_name}_probabity_score.csv')
            with open(str(score_file), 'w') as f:
                fields = list(range(1, n_rows+1, 1))
                datawrite = csv.writer(f, delimiter=',')
                datawrite.writerow(fields)
                
                for i in range(n_rows):
                    index_start= i * n_cols
                    index_end = (i+1) * n_cols
                    pro_rows = predictions[index_start:index_end, 1].tolist()
                    datawrite.writerow([round(value, 6) for value in pro_rows])
                
                datawrite.writerow(['score','++', '++'])
                datawrite.writerow(['score','++', '++'])
                datawrite.writerow(['score','++', '++'])
                datawrite.writerow(['score','++', '++'])

                for i in range(n_rows):
                    index_start= i * n_cols
                    index_end = (i+1) * n_cols
                    socres_rows = y_score[index_start:index_end,1].tolist()
                    datawrite.writerow([round(value, 6) for value in socres_rows])
        # break        
      
            
def predict_tmp(args,meta_epoch):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    relative_path = '/data1/wangjingtao/workplace/python/data/classification/lymph'

    # all centers ffpe-4x
    # df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/multi_centers/captions/patch_use/all_centers/ffpe-4x/f_1.csv'))
    # all centers ffpe-10x
    # df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/multi_centers/captions/patch_use/all_centers/ffpe-10x/f.csv'))
    # df_test_ffpe_micro = df_test[df_test['Class'] == 'ffpe_micro']
    # df_test_ffpe_normal = df_test[df_test['Class'] == 'ffpe_normal']

    # all centers bf-4x
    # df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/multi_centers/captions/patch_use/all_centers/bf-4x/f_1.csv'))
    # # all centers bf-10x
    # df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/multi_centers/captions/patch_use/all_centers/bf-10x/f_1.csv'))
    # df_test_ffpe_micro = df_test[df_test['Class'] == 'bf_micro']
    # df_test_ffpe_normal = df_test[df_test['Class'] == 'bf_normal']

    # throid 10x
    # df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/thyroid/10x/f.csv'))
    # crc 4x
    df_test = pd.read_csv(os.path.join(args.project_path, 'data/lymph/CRC-LN/10x/balance/f_1.csv'))
    
    df_test_ffpe_micro = df_test[df_test['Class'] == 'ffpe_micro']
    df_test_ffpe_normal = df_test[df_test['Class'] == 'ffpe_normal']

    
    
    df_test =  pd.concat([df_test_ffpe_micro, df_test_ffpe_normal],ignore_index=True) 
    # df_test.to_csv('123.csv', index=False)


    task_num = 5 # 多构建几个任务，结果会更准确
    normal_right_all_tasks = {}
    for task_idx in range(task_num):

        dl_ob = dl_comm(args)
        dl_ob._init_net()
        dl_ob._init_opt()

        if args.load == '':
            exit('Predict Mode must give load path')
        else:
            # dl_ob.model.load_state_dict(torch.load(args.load))
            state_dict_path = os.path.join(args.load, 'meta_epoch', f'taskid_{task_idx}', f'best_model_for_valset_{meta_epoch}.pth')
            dl_ob.model.load_state_dict(torch.load(state_dict_path, map_location=device))

        dl_ob.model.eval()
        normal_right = []
        micro_might = []
        for index,row in df_test.iterrows():
            img_file = row['Image_path']
            
            split_values = img_file.split('/')
            image_name = split_values[-2] +'-' + split_values[-1] 
            
            normal_right_store_dir = os.path.join(args.store_dir, 'miss_detect', f'meta_epoch_{meta_epoch}', row['Resolution'],f'taskid_{task_idx}', 'normal_right')
            utils.mkdir(normal_right_store_dir)

            # normal_fail_store_dir = os.path.join(args.store_dir, 'miss_detect', f'meta_epoch_{meta_epoch}',row['Center'], row['Resolution'],f'taskid_{task_idx}', 'normal_fail')
            # utils.mkdir(normal_fail_store_dir)

            micro_right_store_dir = os.path.join(args.store_dir, 'miss_detect', f'meta_epoch_{meta_epoch}', row['Resolution'],f'taskid_{task_idx}', 'micro_right')
            utils.mkdir(micro_right_store_dir)


            # micro_fail_store_dir = os.path.join(args.store_dir, 'miss_detect', f'meta_epoch_{meta_epoch}',row['Center'], row['Resolution'],f'taskid_{task_idx}', 'micro_fail')
            # utils.mkdir(micro_fail_store_dir)

            

            image = cv2.imread(os.path.join(relative_path,img_file))
            if image is None:
                raise ValueError("Failed to load the image")
            # 将图像裁剪成 patches
            # patches = []
            # height, width = image.shape[:2]
            # height = math.ceil(height / 256) * 256
            # width = math.ceil(width / 256) * 256
            # image = cv2.resize(image, [width,height], interpolation=cv2.INTER_CUBIC)
            # n_rows = 0
            # for i in range(0, height, 256):
            #     n_rows += 1
            #     for j in range(0, width, 256):
            #         patch = image[i:i+256, j:j+256, :]

            #         patches.append(patch)
            #         # patch_store =  os.path.join(heat_metric_store_dir, 'patches')
            #         # cv2.imwrite(f'{patch_store}/{i}_{j}.jpg', patches)
            # n_cols = int(len(patches) / n_rows)
            
            # # 对每个 patch 进行预测，并记录预测概率
            # pred_score_list = []
            # predictions = []
            # for patch in patches:
                # 预处理图像
            input_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            input_tensor = preprocess_image(input_image).to(device)
        
            with torch.no_grad():
                output = dl_ob.model(input_tensor)
            
            # # 计算预测概率
            # probability = torch.sigmoid(output[0][:2]).cpu().numpy()
            # predictions.append(probability)
            # pred_score_list.append(output.data.cpu().detach().numpy())
            
            # y_score = np.concatenate(pred_score_list)
            # predictions = softmax(y_score)

            # predictions_score = torch.sigmoid(output[0][:2]).cpu().numpy()
            predictions = np.argmax(output.cpu().numpy(), axis=1)

            if row['Label'] == 0:
                if predictions == 0:
                    # cv2.imwrite(f'{normal_right_store_dir}/{image_name}', image)
                    normal_right.append(row['ID'])
                # else:
                #     cv2.imwrite(f'{normal_fail_store_dir}/{image_name}', image)
            if row['Label'] == 1:
                if predictions == 1:
                    # cv2.imwrite(f'{micro_right_store_dir}/{image_name}', image)
                    micro_might.append(row['ID'])
            #     else:
            #         cv2.imwrite(f'{micro_fail_store_dir}/{image_name}', image)

        normal_right_all_tasks[task_idx] = normal_right
        df_normal_right = df_test[df_test['ID'].isin(normal_right)]
        df_normal_right.to_csv(os.path.join(normal_right_store_dir, 'ffpe_normal.csv'), index=False)

        df_micro_right = df_test[df_test['ID'].isin(micro_might)]
        df_micro_right.to_csv(os.path.join(micro_right_store_dir, 'ffpe_right.csv'), index=False)

    choice = normal_right_all_tasks[0] + normal_right_all_tasks[4]
    
    choice = list(set(choice))
    data3 = df_test[df_test['ID'].isin(choice)]
    data3.to_csv(os.path.join(normal_right_store_dir, 'ffpe_normal_choice.csv'), index=False)




    
           
            




    #     

def test_bf_enter(args):
    record_datatype = args.datatpye
    args.datatpye = args.test_bf
    test_data_ls, _ = lymph_data(args)

    value = predict()

    args.datatpye = record_datatype

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# 定义图像预处理和裁剪函数
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)
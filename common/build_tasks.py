import numpy as np
import random
import numpy as np
import pandas as pd
import os
import csv
# 按比例随机划分list
def data_split(full_list, ratio, shuffle=False):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

# 小涵数
# 莫当真，将shot和quer平均分配到每个patient上
def choose_sq_scans(patient_ls, total_scans, df_class):
    per_patient_scan = int(total_scans // len(patient_ls))
    yu = total_scans % len(patient_ls)
    
    
    if per_patient_scan == 0:
        patient_ls = random.sample(patient_ls, total_scans)
    
    i = 0   
    index_lst = []
    for patient_name in patient_ls:
        df_patient = df_class[df_class['Patient'] == patient_name]# 索引值不会变化
        df_patient_index = df_patient.index.to_list()
        
        
        # 健壮性，先不加
        # if per_patient_scan + 1 > len(df_patient_index):
        #     index_lst += df_patient_index
        #     continue
        
        # 这个余数很巧妙
        if i < yu:
            # index_lst += df_patient_index[:per_patient_scan+1]
            index_lst += random.sample(df_patient_index, per_patient_scan+1)
            
        else:
            # index_lst += df_patient_index[:per_patient_scan]         
            index_lst += random.sample(df_patient_index, per_patient_scan)
        i += 1
        
    return index_lst
        
        
    

# 按病人选scans
def choose_scans(pdframe,shot, query, ls_tmp):
    patient_arr = np.unique(pdframe['Patient'])
    radio = shot/(shot + query)
    patient_support_ls, patient_query_ls = data_split(list(patient_arr), radio, shuffle=True)
    
    index_ls = pdframe.index.to_list()
        
    s_index_ls = choose_sq_scans(patient_support_ls, shot, pdframe)
    q_index_ls = choose_sq_scans(patient_query_ls, query, pdframe)
    
    return s_index_ls, q_index_ls
    
    
class Train_task():
    def __init__(self, args, dataframe, task_index, mode='train'):
        self.dataframe = dataframe
        self.args = args
        self.all_classes = list(dataframe['Class'].unique())
        self.dataframe = dataframe
        self.mode = mode
        self.task_index = task_index

        self.query_roots = [] # test_task_query
        self.support_roots = [] # test_task_support
               
        
        if args.datatype == "OCT":
            self.st_oct()
        elif 'lymph' in args.datatype or 'thyroid' in args.datatype or 'ffpe' in args.datatype or 'bf' in args.datatype or 'camely' in args.datatype or 'background' in  args.datatype:
            self.lymph()
        else:
            raise ValueError('Not implemented buid meta train tasks datatype')

    # st oct数据
    def st_oct(self):
        sampled_classes = random.sample(self.all_classes, self.args.n_way)# 从所有类中随机选取几类

        task_ls_sup = []
        task_ls_qry = []

        for i, c in enumerate(sampled_classes):
            cframe = self.dataframe[self.dataframe["Class"] == c]
            paths = cframe[["Image_path"]]         
            c_index = cframe.index.to_list()
            
            # 要按病人来选scan了，不能简单的按随机
            # sample_idxs = np.random.choice(samples_per_class, samples_per_class, replace=False)
            # support_idxs = sample_idxs[:shot]
            # query_idxs = sample_idxs[shot:(shot + query)]
            support_idxs, query_idxs= choose_scans(cframe, self.args.k_shot, self.args.k_qry, c_index)
            
            for idx in query_idxs:
                self.query_roots.append((paths.iloc[c_index.index(idx)][0], i))# iloc取得是行号，并不是索引,label每次都得变
            for idx in support_idxs:
                self.support_roots.append((paths.iloc[c_index.index(idx)][0], i))    

            task_ls_sup += support_idxs
            task_ls_qry += query_idxs
            task_ls_sup.sort()
            task_ls_qry.sort()
        
        if self.mode == 'train':
            path_1 = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/linux/meta_tasks/train_task'
            store_path = os.path.join(path_1, 'task_' + str(self.task_index) + '.csv')
        else:
            path_1 = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/linux/meta_tasks/val_task'
            store_path = os.path.join(path_1, 'task_' + str(self.task_index) + '.csv')
        self.dataframe.iloc[task_ls_sup].to_csv(store_path)
        self.dataframe.iloc[task_ls_qry].to_csv(store_path,mode='a')
        
        return task_ls_sup, task_ls_qry
    
    # 病理数据
    def lymph(self):
        sampled_classes = random.sample(self.all_classes, self.args.n_way)# 从所有类中随机选取几类

        task_ls_sup = []
        task_ls_qry = []

        for i, c in enumerate(sampled_classes):
            cframe = self.dataframe[self.dataframe["Class"] == c]
            paths = cframe[["Image_path"]]         
            c_index = cframe.index.to_list()
            
            # 这里要确保每个类的patches都大于 self.args.k_shot和self.args.k_qry总和
            random.shuffle(c_index)
            support_idxs = c_index[:self.args.k_shot]
            query_idxs = c_index[self.args.k_shot:self.args.k_shot+self.args.k_qry]


            for idx in query_idxs:
                self.query_roots.append((paths.iloc[c_index.index(idx)][0], i))# iloc取得是行号，并不是索引,label每次都得变
            for idx in support_idxs:
                self.support_roots.append((paths.iloc[c_index.index(idx)][0], i))    

            task_ls_sup += support_idxs
            task_ls_qry += query_idxs
            task_ls_sup.sort()
            task_ls_qry.sort()
        
        # if self.mode == 'train':
        #     path_1 = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/thyroid/meta_tasks/train_tasks'
        #     store_path = os.path.join(path_1, 'task_' + str(self.task_index) + '.csv')
        # else:
        #     path_1 = '/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/data/lymph/thyroid/meta_tasks/val_tasks'
        #     store_path = os.path.join(path_1, 'task_' + str(self.task_index) + '.csv')

        # self.dataframe.iloc[task_ls_sup].to_csv(store_path)
        # self.dataframe.iloc[task_ls_qry].to_csv(store_path,mode='a')
        
        return task_ls_sup, task_ls_qry        

def choose_scans_test(pdframe, scan_num_sum, store_name):
    all_class_name = np.unique(pdframe["Class"])
    index_lst = []
    for class_name in all_class_name:
        df_class = pdframe[pdframe['Class'] == class_name]# 索引值不会变化
        patient_ls = list(np.unique(df_class['Patient']))

        yu = 0 
        if class_name == 'PVRL': # PIC / others
            per_patient_scan = int(scan_num_sum // len(patient_ls))            
            yu = scan_num_sum % len(patient_ls)
            
            if per_patient_scan == 0:
                patient_ls = random.sample(patient_ls, scan_num_sum)
                    
        else:
            per_patient_scan = int((scan_num_sum / 15) // len(patient_ls))
            
            yu = (scan_num_sum / 15) % len(patient_ls)
            
            # 因为scan_num_sum是大于15的
            if per_patient_scan == 0:
                patient_ls = random.sample(patient_ls,  int((scan_num_sum / 15)))
             
        i = 0
        for patient_name in patient_ls:
            df_patient = df_class[df_class['Patient'] == patient_name]# 索引值不会变化
            df_patient_index = df_patient.index.to_list()
            
            # 健壮性
            if per_patient_scan + 1 > len(df_patient_index):
                index_lst += df_patient_index
                continue
            
            if i < yu:
                # index_lst += df_patient_index[:per_patient_scan+1]
                index_lst += random.sample(df_patient_index, per_patient_scan+1)
                
            else:
                # index_lst += df_patient_index[:per_patient_scan]         
                index_lst += random.sample(df_patient_index, per_patient_scan)           
            i += 1
    
    index_lst.sort() # 思考下顺序要不要sort. 答：没有影响，封装daloader时，参数shuffer可以控制
    pdframe.iloc[index_lst].to_csv('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication/' + store_name)
    return index_lst

# 标签变换
def change_classid(pdframe):
    all_class_name = np.unique(pdframe["Class"])
    
    dict_labels = {}
    for label, class_name in enumerate(all_class_name):
        dict_labels[class_name] = label
    
    
    # # 16个标签。16类
    # label_list = []
    # for class_name in pdframe['Class']:
    #     label_list.append(dict_labels[class_name])
        
        
    # 变成两类：rare, others
    label_list = []
    for class_name in pdframe['Class']:
        if class_name == 'PVRL': # PIC/others
            label_list.append(0)
        else:
            label_list.append(1)
        
    return label_list


# 用于元测试
class Meta_Test_task():
    def __init__(self, args, df_test_ls, test_class='pvrl_pic_rp_common'):
        self.args = args
        self.df_train_data = df_test_ls[0]
        self.df_val_data = df_test_ls[1]
        self.df_test_data = df_test_ls[2]

        self.support_roots = [] # 相当于train   
        self.query_roots = [] # 相当于val
        
        self.test_roots = [] # 用于最终测试
 
        if test_class == 'pvrl_others':
            self.get_pvrl_index()
        elif test_class == 'pvrl_pic_rp_normal':
            self.get_four_index()# rare + normal 四分类
        elif test_class == 'pvrl_pic_rp_common':
            self.get_four_comm()
        elif 'lymph' in test_class or 'thyroid' in test_class:
            self.get_lymph()
        

    # 做PVRL——others分类  
    def get_pvrl_index(self):
        # support set+++++
        paths = self.df_train_data[["Image_path", "Label"]]
        label_lst= change_classid(self.df_train_data)
        paths.loc[paths.index, 'Label'] = label_lst
        # iloc的是行号，不是索引号，行号和索引号可能不对应
        index_ls = self.df_train_data.index.to_list()
        support_idxs = choose_scans_test(self.df_train_data, self.args.test_k_shot, 'train.csv')

        for idx in support_idxs:
            self.support_roots.append((paths.iloc[index_ls.index(idx)][0], paths.iloc[index_ls.index(idx)][1]))   
        
  
        # querry set++++++
        paths = self.df_val_data[["Image_path", "Label"]]
        label_lst= change_classid(self.df_val_data)
        paths.loc[paths.index, 'Label'] = label_lst
        index_ls = self.df_val_data.index.to_list()
        query_idxs = choose_scans_test(self.df_val_data, self.args.test_k_query, 'val.csv')
       
        for idx in query_idxs:
            self.query_roots.append((paths.iloc[index_ls.index(idx)][0], paths.iloc[index_ls.index(idx)][1]))   
        

        # test set++++++
        paths = self.df_test_data[["Image_path", "Label"]]
        label_lst= change_classid(self.df_test_data)
        paths.loc[paths.index,'Label'] = label_lst
        index_ls = self.df_test_data.index.to_list()
        test_idxs = choose_scans_test(self.df_test_data, self.args.test_k_query, 'test.csv')  
        for idx in test_idxs:
            self.test_roots.append((paths.iloc[index_ls.index(idx)][0], paths.iloc[index_ls.index(idx)][1]))    

    # 做pvrl_pic_rp_normal分类
    def get_four_index(self):

        if 'Label' not in self.df_train_data.columns:
            # 增加Label列
            class_id_dict =  {'normal':0, 'PIC':1, 'PVRL':2, 'RP':3}

            label_list= add_label(self.df_train_data, class_id_dict)
            self.df_train_data['Label'] = label_list

            label_list= add_label(self.df_val_data, class_id_dict)
            self.df_val_data['Label'] = label_list

            label_list= add_label(self.df_test_data, class_id_dict)
            self.df_test_data['Label'] = label_list



        # support set+++++
        paths = self.df_train_data[["Image_path", "Label"]]
        # iloc的是行号，不是索引号，行号和索引号可能不对应
        index_ls = self.df_train_data.index.to_list()
        support_idxs = index_ls

        for idx in support_idxs:
            self.support_roots.append((paths.iloc[index_ls.index(idx)][0], paths.iloc[index_ls.index(idx)][1]))   
        
  
        # querry set++++++
        paths = self.df_val_data[["Image_path", "Label"]]
        index_ls = self.df_val_data.index.to_list()
        query_idxs = index_ls
       
        for idx in query_idxs:
            self.query_roots.append((paths.iloc[index_ls.index(idx)][0], paths.iloc[index_ls.index(idx)][1]))   
        

        # test set++++++
        paths = self.df_test_data[["Image_path", "Label"]]
        index_ls = self.df_test_data.index.to_list()
        test_idxs = index_ls
        for idx in test_idxs:
            self.test_roots.append((paths.iloc[index_ls.index(idx)][0], paths.iloc[index_ls.index(idx)][1]))    

    # 做pvrl_pic_rp_ommon分类
    def get_four_comm(self):
        # support set+++++
        paths = self.df_train_data[["Image_path", "Label"]]
        # iloc的是行号，不是索引号，行号和索引号可能不对应
        index_ls = self.df_train_data.index.to_list()
        support_idxs = index_ls

        for idx in support_idxs:
            self.support_roots.append((paths.iloc[index_ls.index(idx)][0], paths.iloc[index_ls.index(idx)][1]))   
        
  
        # querry set++++++
        paths = self.df_val_data[["Image_path", "Label"]]
        index_ls = self.df_val_data.index.to_list()
        query_idxs = index_ls
       
        for idx in query_idxs:
            self.query_roots.append((paths.iloc[index_ls.index(idx)][0], paths.iloc[index_ls.index(idx)][1]))   
        
        # test set++++++
        paths = self.df_test_data[["Image_path", "Label"]]
        index_ls = self.df_test_data.index.to_list()
        test_idxs = index_ls
        for idx in test_idxs:
            self.test_roots.append((paths.iloc[index_ls.index(idx)][0], paths.iloc[index_ls.index(idx)][1]))   

    # 做lymph normal micro分类
    def get_lymph(self):

        # support set+++++
        paths = self.df_train_data[["Image_path", "Label"]]
        # iloc的是行号，不是索引号，行号和索引号可能不对应
        index_ls = self.df_train_data.index.to_list()
        support_idxs = index_ls
        for idx in support_idxs:
            self.support_roots.append((paths.iloc[index_ls.index(idx)][0], paths.iloc[index_ls.index(idx)][1]))   
        
        # querry set++++++
        paths = self.df_val_data[["Image_path", "Label"]]
        index_ls = self.df_val_data.index.to_list()
        query_idxs = index_ls
        for idx in query_idxs:
            self.query_roots.append((paths.iloc[index_ls.index(idx)][0], paths.iloc[index_ls.index(idx)][1]))   
        
        # test set++++++
        paths = self.df_test_data[["Image_path", "Label"]]
        index_ls = self.df_test_data.index.to_list()
        test_idxs = index_ls
        for idx in test_idxs:
            self.test_roots.append((paths.iloc[index_ls.index(idx)][0], paths.iloc[index_ls.index(idx)][1]))   

        


# 增加label列
def add_label(dframe, class_id_dict = None):
    class_ls = np.unique(dframe['Class'])
    if class_id_dict is None:
        class_id_dict ={}
        for i, class_name in enumerate(class_ls):
            class_id_dict[class_name] = i

    allclass = list(dframe['Class'])
    label_list = [class_id_dict.get(item, item) for item in allclass]

    return label_list


def charge(df_train, df_val, df_test):
    all_class = ["normal", "acute CSCR", "acute RAO", "acute RVO", "acute VKH",  "dAMD", "macular-off RRD", "mCNV", "MTM", "nAMD", "nPDR", "PCV", "PDR", "PIC", "PVRL", "RP"]
    for class_name in all_class:
        train_patient= df_train[df_train['Class'] == class_name]['Patient']
        tarin_id = list(np.unique(list(train_patient)))
        
        val_patient= df_val[df_val['Class'] == class_name]['Patient']
        val_id = list(np.unique(list(val_patient)))
        
        test_patient= df_test[df_test['Class'] == class_name]['Patient']
        test_id = list(np.unique(list(test_patient)))
        

        id = tarin_id + val_id + test_id
        
        set_id = set(id)
        
        if len(id)==len(set_id):
            print('列表里的元素互不重复！')
        else:
            print('列表里有重复的元素！')  
            print(class_name)
            print(tarin_id)
            print(val_id)
            print(test_id)


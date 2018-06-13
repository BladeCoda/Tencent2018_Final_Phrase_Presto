# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

#切割训练数据的代码，比赛模型使用的是按照时间切割cutTrainByTime

def userFeatureCut(num=10):
    userFeature_data = []
    with open('data/origin/userFeature.data', mode='r', encoding='utf-8') as f:
        files=f.readlines()
    flen=len(files)
    chunck=flen//num
    for i in range(num):
        print('处理第%d份数据'%(i+1))
        userFeature_data = []
        if i!=num-1:
            cdata=files[i*chunck:(i+1)*chunck]
        else:
            cdata=files[i*chunck:]
        j=1
        for line in cdata:
            line = line[:-1].split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            # print(userFeature_dict)
            userFeature_data.append(userFeature_dict)
            if j%100000 ==0:
                print('当前已完成： %.3f'%(j*1.0/chunck))
            j+=1
        userFeature_pd = pd.DataFrame(userFeature_data)
        userFeature_pd.to_csv('data/origin/chunck/userFeature_%d.csv'%(i+1),index =False)
        del cdata
        del userFeature_data
    print('分割处理完毕')

def concatCutFile(num=10):
    df_all=pd.read_csv('data/origin/chunck/userFeature_1.csv') 
    print('开始合并')
    for i in range(1,num):
        print('合并第%d份数据'%(i+1))
        df_current=pd.read_csv('data/origin/chunck/userFeature_%d.csv'%(i+1))
        df_all=df_all.append(df_current)
        del df_current      
    print('合并完毕')
    df_all.to_csv('data/origin/userFeature.csv',index =False)
    
def userfeatureProcessing(num=10):
    userFeatureCut(num=10)
    concatCutFile(num=10)
    
def featureRegular():
    df_all=pd.read_csv('data/origin/userFeature.csv')
    print('保存细节信息')
    df_detail=df_all.drop(['age', 'gender',
                            'education', 'consumptionAbility', 'LBS',
                            'carrier', 'house'],axis=1)
    df_detail.to_csv('data/origin/userFeature_detail.csv',index=False)
    del df_detail
    print('全部保存完毕')
    
def merge(dtype='train'):
    if dtype=='train':
        df_join=pd.read_csv('data/origin/train.csv')
        out_path='data/merge/merge_train.csv'
    elif dtype=='test1':
        df_join=pd.read_csv('data/origin/test1.csv')
        out_path='data/merge/merge_test1.csv'
    elif dtype=='test2':
        df_join=pd.read_csv('data/origin/test2.csv')
        out_path='data/merge/merge_test2.csv'
    else:
        print('error type')
        return
        
    print('加载特征表')
        
    df_ad=pd.read_csv('data/origin/adFeature.csv')
    df_user=pd.read_csv('data/origin/userFeature.csv')
    
    #拼接信息
    print('开始拼接%s的广告信息'%dtype)
    df_join=pd.merge(df_join,df_ad,how='left',on='aid') #拼接用户信息
    del df_ad
    
    print('开始拼接%s的用户信息'%dtype)
    df_join=pd.merge(df_join,df_user,how='left',on='uid')#拼接position信息
    del df_user
    
    print('拼接完成，开始保存')
    
    df_join.to_csv(out_path,index=False)
    print('保存完成,开始分割保存信息')
    
    del df_join
    
def merge_cut(dtype='train'):
    if dtype=='train':
        df_join=pd.read_csv('data/merge/merge_train.csv')
    elif dtype=='test1':
        df_join=pd.read_csv('data/merge/merge_test1.csv')
    elif dtype=='test2':
        df_join=pd.read_csv('data/merge/merge_test2.csv')
    else:
        print('error type')
        return
    
    print('保存基本信息')
    df_base=df_join.drop(['appIdAction','appIdInstall','ct','interest1',
                   'interest2','interest3','interest4','interest5',
                   'kw1','kw2','kw3','topic1','topic2','topic3',
                   'marriageStatus','os'],axis=1)
    df_base.to_csv('data/merge/%s_m_base.csv'%dtype,index=False)
    del df_base

    del df_join
    
def getTrainLabel():
    print('直接获取Train的Label')
    df_train=pd.read_csv('data/origin/train.csv')
    df_out=pd.DataFrame({'label':df_train['label'].values})  
    print('开始保存')
    df_out.to_csv('data/origin/train_label.csv',index=False)
        
if __name__=='__main__':
    
    #对数据进行切割
    '''userFeatureCut(num=10)
    concatCutFile(num=10)
    userfeatureProcessing(num=10)'''
    
    #拼接数据
    merge(dtype='train')
    merge(dtype='test1')
    merge(dtype='test2')
    
    merge_cut(dtype='train')
    merge_cut(dtype='test1')
    merge_cut(dtype='test2')
    
    featureRegular()
    getTrainLabel()
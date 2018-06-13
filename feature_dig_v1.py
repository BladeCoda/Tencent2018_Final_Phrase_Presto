
#V1版特征，计数统计

import pandas as pd
import numpy as np

def countDetail(x):   
    #对兴趣进行计数
    count_i1=0
    count_i2=0
    count_i3=0
    count_i4=0
    count_i5=0
    
    if not isinstance(x.interest1,float):
        count_i1=len(x.interest1.strip().split(' '))
        
    if not isinstance(x.interest2,float):
        count_i2=len(x.interest2.strip().split(' '))
        
    if not isinstance(x.interest3,float):
        count_i3=len(x.interest3.strip().split(' '))
        
    if not isinstance(x.interest4,float):
        count_i4=len(x.interest4.strip().split(' '))
        
    if not isinstance(x.interest5,float):
        count_i5=len(x.interest5.strip().split(' '))
    
    #对关键词进行计数
    count_k1=0
    count_k2=0
    count_k3=0
    
    if not isinstance(x.kw1,float):
        count_k1=len(x.kw1.strip().split(' '))
        
    if not isinstance(x.kw2,float):
        count_k2=len(x.kw2.strip().split(' '))
        
    if not isinstance(x.kw3,float):
        count_k3=len(x.kw3.strip().split(' '))
    
    #对主题计数
    count_t1=0
    count_t2=0
    count_t3=0
    
    if not isinstance(x.topic1,float):
        count_t1=len(x.topic1.strip().split(' '))
        
    if not isinstance(x.topic2,float):
        count_t2=len(x.topic2.strip().split(' '))
        
    if not isinstance(x.topic3,float):
        count_t3=len(x.topic3.strip().split(' '))
    
    #APP活跃计数
    count_aa=0
    if not isinstance(x.appIdAction,float):
        count_aa=len(x.appIdAction.strip().split(' '))
        
    #APP安装计数
    count_ai=0
    if not isinstance(x.appIdInstall,float):
        count_ai=len(x.appIdInstall.strip().split(' '))
        
    re=str(count_i1)+','+str(count_i2)+','+str(count_i3)+','+str(count_i4)+','+str(count_i5)
    re+=','+str(count_k1)+','+str(count_k2)+','+str(count_k3)
    re+=','+str(count_t1)+','+str(count_t2)+','+str(count_t3)
    re+=','+str(count_aa)+','+str(count_ai)
    
    print(re)  
    return re
    
def get_count_i1(x):
    return x.split(',')[0]
def get_count_i2(x):
    return x.split(',')[1]
def get_count_i3(x):
    return x.split(',')[2]
def get_count_i4(x):
    return x.split(',')[3]
def get_count_i5(x):
    return x.split(',')[4]

def get_count_k1(x):
    return x.split(',')[5]
def get_count_k2(x):
    return x.split(',')[6]
def get_count_k3(x):
    return x.split(',')[7]

def get_count_t1(x):
    return x.split(',')[8]
def get_count_t2(x):
    return x.split(',')[9]
def get_count_t3(x):
    return x.split(',')[10]

def get_count_aa(x):
    return x.split(',')[11]
def get_count_ai(x):
    return x.split(',')[12]

def digCount():

    df_all=pd.read_csv('data/origin/userFeature_detail.csv')     
    df_out=pd.DataFrame({'uid':df_all['uid'].values})  
    
    print('开始挖掘')
    df_out['Detail']=df_all.apply(lambda x:countDetail(x),axis=1)
    del df_all
    
    df_out['count_i1']=df_out.Detail.apply(lambda x:get_count_i1(x))
    print('count_i1统计完成')
    df_out['count_i2']=df_out.Detail.apply(lambda x:get_count_i2(x))
    print('count_i2统计完成')
    df_out['count_i3']=df_out.Detail.apply(lambda x:get_count_i3(x))
    print('count_i3统计完成')
    df_out['count_i4']=df_out.Detail.apply(lambda x:get_count_i4(x))
    print('count_i4统计完成')
    df_out['count_i5']=df_out.Detail.apply(lambda x:get_count_i5(x))
    print('count_i5统计完成')
    
    df_out['count_k1']=df_out.Detail.apply(lambda x:get_count_k1(x))
    print('count_k1统计完成')
    df_out['count_k2']=df_out.Detail.apply(lambda x:get_count_k2(x))
    print('count_k2统计完成')
    df_out['count_k3']=df_out.Detail.apply(lambda x:get_count_k3(x))
    print('count_k3统计完成')
    
    df_out['count_t1']=df_out.Detail.apply(lambda x:get_count_t1(x))
    print('count_t1统计完成')
    df_out['count_t2']=df_out.Detail.apply(lambda x:get_count_t2(x))
    print('count_t2统计完成')
    df_out['count_t3']=df_out.Detail.apply(lambda x:get_count_t3(x))
    print('count_t3统计完成')
    
    df_out['count_aa']=df_out.Detail.apply(lambda x:get_count_aa(x))
    print('count_aa统计完成')
    df_out['count_ai']=df_out.Detail.apply(lambda x:get_count_ai(x))
    print('count_ai统计完成')
    
    df_out=df_out.drop(['Detail'],axis=1)
    print('开始保存')
    
    df_out.to_csv('data/extra/countFeature.csv',index=False)
    
def mergeCount(dtype='train'):
    if dtype=='train':
        df_join=pd.read_csv('data/merge/train_m_base.csv')
        outpath='data/feature/train_v1.csv'
    elif dtype=='test1':
        df_join=pd.read_csv('data/merge/test1_m_base.csv')
        outpath='data/feature/test1_v1.csv'
    elif dtype=='test2':
        df_join=pd.read_csv('data/merge/test2_m_base.csv')
        outpath='data/feature/test2_v1.csv'
    else:
        print('error type')
        return
    
    print('开始拼接特征！')
    df_feature=pd.read_csv('data/extra/countFeature.csv')
    df_join=pd.merge(df_join,df_feature,how='left',on='uid')#拼接position信息
    del df_feature
    
    print('拼接完成，开始保存')
    df_join.to_csv(outpath,index=False)
    print('保存完毕')
    
if __name__=='__main__':
    
    #digCount()
    
    #mergeCount(dtype='train')
    mergeCount(dtype='test1')
    mergeCount(dtype='test2')
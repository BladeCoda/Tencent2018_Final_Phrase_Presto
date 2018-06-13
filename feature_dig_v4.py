#对用户的APP行为和广告特征进行统计

import pandas as pd
import numpy as np
    
#统计交集
def count_Ad(list_a,dic):
    re=[]
    for item in list_a:
        re.append(dic[item])
    return np.array(re)
    
def digAdFeature():
    print('统计广告特征')
    df_ad=pd.read_csv('data/origin/adFeature.csv')   
    df_out=pd.DataFrame({'aid':df_ad['aid'].values})  
    
    dict_adv=df_ad['aid'].groupby(df_ad['advertiserId']).count().to_dict()
    dict_cam=df_ad['aid'].groupby(df_ad['campaignId']).count().to_dict()
    dict_pro=df_ad['aid'].groupby(df_ad['productId']).count().to_dict()
    dict_adc=df_ad['aid'].groupby(df_ad['adCategoryId']).count().to_dict()
    
    df_out['adv_ad']=count_Ad(df_ad['advertiserId'].values,dict_adv)
    df_out['cam_ad']=count_Ad(df_ad['campaignId'].values,dict_cam)
    df_out['pro_ad']=count_Ad(df_ad['productId'].values,dict_pro)
    df_out['adc_ad']=count_Ad(df_ad['adCategoryId'].values,dict_adc)
    
    print('保存文件')
    df_out.to_csv('data/extra/adFeature_c.csv',index=False)
    
def mergeV4(dtype='train'):
    if dtype=='train':
        df_join=pd.read_csv('data/feature/train_V3.csv')
        outpath='data/feature/train_v4.csv'
    elif dtype=='test1':
        df_join=pd.read_csv('data/feature/test1_V3.csv')
        outpath='data/feature/test1_v4.csv'
    elif dtype=='test2':
        df_join=pd.read_csv('data/feature/test2_V3.csv')
        outpath='data/feature/test2_v4.csv'
    else:
        print('error type')
        return
    
    print('开始拼接广告特征！')
    df_feature=pd.read_csv('data/extra/adFeature_c.csv')
    df_join=pd.merge(df_join,df_feature,how='left',on='aid')#拼接position信息
    del df_feature
    
    
    print('拼接完成，开始保存')
    df_join.to_csv(outpath,index=False)
    print('保存完毕')
    
if __name__=='__main__': 
    digAdFeature()
    
    mergeV4(dtype='train')
    mergeV4(dtype='test1')
    mergeV4(dtype='test2')
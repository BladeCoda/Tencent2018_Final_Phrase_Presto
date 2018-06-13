#对广告和用户的一些信息做计数统计

import pandas as pd
import numpy as np
    
def count_ID(flist):
    num_all=len(flist)
    i=1
    dict_re={}
    for item in flist:
        if item in dict_re:
            dict_re[item]+=1
        else:
            dict_re[item]=1
        i+=1
        if i%1000000==0:
            print('统计字典:%.2f%%'%(i*100.0/num_all))
    print('统计完成')  
    return dict_re
    
def count_trans(flist,dict_re):
    re=[]
    for item in flist:
        re.append(dict_re[item])  
    return re


def digIDCount():

    print('开始载入')
    df_train=pd.read_csv('data/merge/train_m_base.csv')   
    df_train=df_train.drop('label',axis=1)  
    df_test1=pd.read_csv('data/merge/test1_m_base.csv') 
    df_test2=pd.read_csv('data/merge/test2_m_base.csv') 
    
    df_all=df_train.append(df_test1)
    df_all=df_all.append(df_test2)
    del df_train
    del df_test1
    del df_test2
    
    print('统计aid记录数')
    dict_aid=count_ID(df_all['aid'].values)
    
    print('统计uid记录数')
    dict_uid=count_ID(df_all['uid'].values)
    
    print('统计advertiserId记录数')
    dict_adid=count_ID(df_all['advertiserId'].values)
    
    print('统计campaignId记录数')
    dict_camid=count_ID(df_all['campaignId'].values)
    
    print('统计creativeId记录数')
    dict_creid=count_ID(df_all['creativeId'].values)
    
    df_all
    
    print('载入trainV2......')
    df_train=pd.read_csv('data/feature/train_v2.csv')
    print('对训练文件生成aid特征')
    df_train['aid_c']=count_trans(df_train['aid'].values,dict_aid)
    print('对训练文件生成uid特征')
    df_train['uid_c']=count_trans(df_train['uid'].values,dict_uid)
    print('对训练文件生成advertiserId特征')
    df_train['adid_c']=count_trans(df_train['advertiserId'].values,dict_adid)
    print('对训练文件生成campaignId特征')
    df_train['camid_c']=count_trans(df_train['campaignId'].values,dict_camid)
    print('对训练文件生成creativeId特征')
    df_train['creid_c']=count_trans(df_train['creativeId'].values,dict_creid)
    print('保存训练文件')
    df_train.to_csv('data/feature/train_V3.csv',index=False)
    del df_train
    
    print('载入test1V2......')
    df_test=pd.read_csv('data/feature/test1_v2.csv')
    print('对测试文件生成aid特征')
    df_test['aid_c']=count_trans(df_test['aid'].values,dict_aid)
    print('对测试生成uid特征')
    df_test['uid_c']=count_trans(df_test['uid'].values,dict_uid)
    print('对测试文件生成advertiserId特征')
    df_test['adid_c']=count_trans(df_test['advertiserId'].values,dict_adid)
    print('对测试文件生成campaignId特征')
    df_test['camid_c']=count_trans(df_test['campaignId'].values,dict_camid)
    print('对测试文件生成creativeId特征')
    df_test['creid_c']=count_trans(df_test['creativeId'].values,dict_creid)
    print('保存测试文件')
    df_test.to_csv('data/feature/test1_V3.csv',index=False)
    del df_test
    
    print('载入test2V2......')
    df_test=pd.read_csv('data/feature/test2_v2.csv')
    print('对测试文件生成aid特征')
    df_test['aid_c']=count_trans(df_test['aid'].values,dict_aid)
    print('对测试生成uid特征')
    df_test['uid_c']=count_trans(df_test['uid'].values,dict_uid)
    print('对测试文件生成advertiserId特征')
    df_test['adid_c']=count_trans(df_test['advertiserId'].values,dict_adid)
    print('对测试文件生成campaignId特征')
    df_test['camid_c']=count_trans(df_test['campaignId'].values,dict_camid)
    print('对测试文件生成creativeId特征')
    df_test['creid_c']=count_trans(df_test['creativeId'].values,dict_creid)
    print('保存测试文件')
    df_test.to_csv('data/feature/test2_V3.csv',index=False)
    del df_test
    
if __name__=='__main__': 
    digIDCount()
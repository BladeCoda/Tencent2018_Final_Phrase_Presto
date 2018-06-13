# -*- coding: utf-8 -*-

#V2版特征，对列表特征的编码

import pandas as pd
import numpy as np

def encoding(flist):
    num_all=len(flist)
    i=1
    j=1
    dict_re={'':0}
    re_list=[]
    for item in flist:
        if isinstance(item,float):
            re_list.append(0)
        else:
            list_t=item.strip().split(' ')
            list_t.sort()
            key=' '.join(list_t)
            if key in dict_re:
                re_list.append(dict_re[key])
            else:
                re_list.append(j)
                dict_re[key]=j
                j+=1
        i+=1
        if i%1000000==0:
            print('转化已完成:%.2f%%'%(i*100.0/num_all))
    print('转化完成')
    return np.array(re_list)
    
def count_encoding(flist):
    num_all=len(flist)
    i=1
    dict_re={0:0}
    for item in flist:
        if item!=0:
            if item in dict_re:
                dict_re[item]+=1
            else:
                dict_re[item]=1
        i+=1
        if i%1000000==0:
            print('统计字典:%.2f%%'%(i*100.0/num_all))
    print('统计完成,开始记录')
    
    re_list=[]
    i=1
    for item in flist:
        re_list.append(dict_re[item])
    return np.array(re_list)


def digEncoding():

    df_all=pd.read_csv('data/origin/userFeature_detail.csv')     
    df_out=pd.DataFrame({'uid':df_all['uid'].values})  
    
    #对兴趣编码
    for i in range(1,6):
        print('统计interest%d'%i)
        e=df_all['interest%d'%i].values
        e=encoding(e)
        df_out['i%d_e'%i]=e
        del e
        
    #对关键词编码
    for i in range(1,4):
        print('统计kw%d'%i)
        e=df_all['kw%d'%i].values
        e=encoding(e)
        df_out['k%d_e'%i]=e
        del e
        
    #对主题编码
    for i in range(1,4):
        print('统计topic%d'%i)
        e=df_all['topic%d'%i].values
        e=encoding(e)
        df_out['t%d_e'%i]=e
        del e
    
    os_e=df_all.os.values
    ct_e=df_all.ct.values
    mar_e=df_all.marriageStatus.values
    
    del df_all
    
    print('开始对os数据进行转化')
    os_e=encoding(os_e)
    df_out['os_e']=os_e
    del os_e
    
    print('开始对ct数据进行转化')
    ct_e=encoding(ct_e)
    df_out['ct_e']=ct_e
    del ct_e
    
    print('开始对婚姻数据进行转化')
    mar_e=encoding(mar_e)
    df_out['mar_e']=mar_e
    del mar_e    

    print('开始保存')
    df_out.to_csv('data/extra/EncodingFeature.csv',index=False)
 
#----分割线，对id进行次数编码，而非直接利用（效果待验证）----
def digEncoding_C():
    print('开始加载特征')
    df_out=pd.read_csv('data/extra/EncodingFeature.csv')
    for i in range(1,6):
        print('计数转化interest%d'%i)
        df_out['i%d_e'%i]=count_encoding(df_out['i%d_e'%i].values)
        
    #对关键词编码
    for i in range(1,4):
        print('计数转化kw%d'%i)
        df_out['k%d_e'%i]=count_encoding(df_out['k%d_e'%i].values)
        
    #对主题编码
    for i in range(1,4):
        print('计数转化topic%d'%i)
        df_out['t%d_e'%i]=count_encoding(df_out['t%d_e'%i].values)
        
    print('开始保存')
    df_out.to_csv('data/extra/EncodingFeature_C.csv',index=False)
    
def mergeEncoding(dtype='train',EC=False):
    if dtype=='train':
        df_join=pd.read_csv('data/feature/train_v1.csv')
        outpath='data/feature/train_v2.csv'
    elif dtype=='test1':
        df_join=pd.read_csv('data/feature/test1_v1.csv')
        outpath='data/feature/test1_v2.csv'
    elif dtype=='test2':
        df_join=pd.read_csv('data/feature/test2_v1.csv')
        outpath='data/feature/test2_v2.csv'
    else:
        print('error type')
        return
    
    print('开始拼接特征！')
    if EC==True:
        df_feature=pd.read_csv('data/extra/EncodingFeature_C.csv')
    else:
        df_feature=pd.read_csv('data/extra/EncodingFeature.csv')
    df_join=pd.merge(df_join,df_feature,how='left',on='uid')#拼接position信息
    del df_feature
    
    print('拼接完成，开始保存')
    df_join.to_csv(outpath,index=False)
    print('保存完毕')
    
if __name__=='__main__': 
    digEncoding()
    digEncoding_C()
    
    mergeEncoding(dtype='train',EC=True)
    mergeEncoding(dtype='test1',EC=True)
    mergeEncoding(dtype='test2',EC=True)
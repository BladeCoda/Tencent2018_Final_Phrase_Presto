#V8版本特征：交叉转化次数

from collections import Counter
import pandas as pd 
import numpy as np
        
#对训练集进行分割以用于后续的窗口计算
def cutTrain():
    print('开始载入') 
    df_train=pd.read_csv('data/merge/train_m_base.csv') 
    
    chunck=len(df_train)//5
    for i in range(5):
        print('处理第%d份数据'%(i+1))
        if i!=4:
            df_t=df_train[i*chunck:(i+1)*chunck]
        else:
            df_t=df_train[i*chunck:]
        print('保存第%d份分割'%(i+1))
        df_t.to_csv('data/merge/window/win_%d.csv'%(i+1),index =False)
        del df_t
    print('分割处理完毕')
 
#对某一类特征进行统计
def windowCount(flist1,all_id):
    print('开始统计')
    dic_zz=dict(Counter(flist1))  
    print('构建种子率')
    dic_re={}
    for fid in all_id:
        if fid not in dic_zz:
            dic_re[fid]=0
        else:
            dic_re[fid]=dic_zz[fid]  
    return dic_re
    
#获取种子率
def getWindow(flist,dic):
    re=[]
    for item in flist:
        re.append(dic[item])
    return np.array(re)
    
#对某一个窗口内进行统计
def digSingleWindow(dfa,dfb,all_uid,all_aid,all_cid,all_pid):
    
    print('获取uid的转化次数')
    flist1=dfa[dfa['label']==1].uid.values
    dic=windowCount(flist1,all_uid)
    uid_wsm=getWindow(dfb.uid.values,dic)
    del flist1,dic
    
    print('获取aid的转化次数')
    flist1=dfa[dfa['label']==1].aid.values
    dic=windowCount(flist1,all_aid)
    aid_wsm=getWindow(dfb.aid.values,dic)
    del flist1,dic
    
    print('获取cid的转化次数')
    flist1=dfa[dfa['label']==1].campaignId.values
    dic=windowCount(flist1,all_cid)
    cid_wsm=getWindow(dfb.campaignId.values,dic)
    del flist1,dic
    
    print('获取pid的转化次数')
    flist1=dfa[dfa['label']==1].productId.values
    dic=windowCount(flist1,all_pid)
    pid_wsm=getWindow(dfb.productId.values,dic)
    del flist1,dic
    
    dft=pd.DataFrame({'uid_w':uid_wsm,'aid_w':aid_wsm,
                      'cid_w':cid_wsm,'pid_w':pid_wsm})
    return dft  
    
#对aid,camid和proid来挖掘转化率
def digWindow():
    
    print('获取全部UID')
    df_user=pd.read_csv('data/origin/userFeature.csv')
    all_uid=df_user.uid.values
    del df_user
    
    print('获取全部 ad ID')
    df_ad=pd.read_csv('data/origin/adFeature.csv')
    all_aid=set(df_ad.aid.values)
    all_cid=set(df_ad.campaignId.values)
    all_pid=set(df_ad.productId.values)
    del df_ad
    
    dft=''
    for i in range (1,6):
        print('处理第%d个窗口'%i)
        dfa=''
        for j in range(1,6):
            if j!=i:
                if isinstance(dfa,str):
                    dfa=pd.read_csv('data/merge/window/win_%d.csv'%j)
                else:
                    dfa=dfa.append(pd.read_csv('data/merge/window/win_%d.csv'%j))
                    
        dfb=pd.read_csv('data/merge/window/win_%d.csv'%i)
        if isinstance(dft,str):
            dft=digSingleWindow(dfa,dfb,all_uid,all_aid,all_cid,all_pid)
        else:
            dft=dft.append(digSingleWindow(dfa,dfb,all_uid,all_aid,all_cid,all_pid))
        del dfa,dfb
    print('保存训练特征')
    dft.to_csv('data/extra/train_window.csv',index=False)
    del dft
    
    print('处理测试数据')
    dfa=''
    for i in range (1,5):
        if isinstance(dfa,str):
            dfa=pd.read_csv('data/merge/window/win_%d.csv'%i)
        else:
            dfa=dfa.append(pd.read_csv('data/merge/window/win_%d.csv'%i))
            
    dfb=pd.read_csv('data/merge/test1_m_base.csv')
    dft=digSingleWindow(dfa,dfb,all_uid,all_aid,all_cid,all_pid)
    print('保存测试特征')
    dft.to_csv('data/extra/test1_window.csv',index=False)  
    
    dfb=pd.read_csv('data/merge/test2_m_base.csv')
    dft=digSingleWindow(dfa,dfb,all_uid,all_aid,all_cid,all_pid)
    print('保存测试特征')
    dft.to_csv('data/extra/test2_window.csv',index=False)  
    
def mergeV8(dtype='train'):
    if dtype=='train':
        df_join=pd.read_csv('data/feature/train_v7.csv')
        dft=pd.read_csv('data/extra/train_window.csv')
        outpath='data/feature/train_v8.csv'
    elif dtype=='test1':
        df_join=pd.read_csv('data/feature/test1_v7.csv')
        outpath='data/feature/test1_v8.csv'
        dft=pd.read_csv('data/extra/test1_window.csv')
    elif dtype=='test2':
        df_join=pd.read_csv('data/feature/test2_v7.csv')
        outpath='data/feature/test2_v8.csv'
        dft=pd.read_csv('data/extra/test2_window.csv')
    else:
        print('error type')
        return
    
    print('开始拼接特征！')
    df_join['uid_w']=dft['uid_w'].values
    df_join['cid_w']=dft['cid_w'].values
    df_join['pid_w']=dft['pid_w'].values
    df_join['aid_w']=dft['aid_w'].values
    del dft
    
    print('拼接完成，开始保存')
    df_join.to_csv(outpath,index=False)
    print('保存完毕')
    
if __name__=='__main__': 
    #cutTrain()
    
    digWindow()
    mergeV8(dtype='train')
    mergeV8(dtype='test1')
    mergeV8(dtype='test2')

# 对提交结果进行处理，包括融合，加权，概率调整等等

import pandas as pd
import numpy as np

def combineResult(blper=0.5):
    print('开始合并结果')
    p1=[]
    with open('result/submission.csv', mode='r', encoding='utf-8') as f:
        lines=f.readlines()
        for item in lines:
            p1.append(float(item.split(',')[2]))
            
    p2=[]
    with open('data/combine/submission_bl.csv', mode='r', encoding='utf-8') as f:
        lines=f.readlines()
        for item in lines:
            p2.append(float(item.split(',')[2]))
            
    prob=[]
    for i in range(len(p1)):
        prob.append(p1[i]*(1-blper)+p2[i]*blper)
        
    df_origin=pd.read_csv('data/origin/test2.csv')
    aid=df_origin.aid.values
    uid=df_origin.uid.values
    del df_origin
    
    output=pd.DataFrame({'aid':aid,'uid':uid,'score':prob})
    output['score']=output.score.apply(lambda x:round(x,8))
    
    columns = ['aid','uid','score']
    output.to_csv('result/submission_c.csv',index=False,columns=columns,header=None)
    
def combineResult2():
    print('开始合并结果')
    p1=[]
    with open('result/submission.csv', mode='r', encoding='utf-8') as f:
        lines=f.readlines()
        for item in lines:
            p1.append(float(item.split(',')[2]))
            
    p2=[]
    with open('data/combine/submission_bl.csv', mode='r', encoding='utf-8') as f:
        lines=f.readlines()
        for item in lines:
            p2.append(float(item.split(',')[2]))
            
    p3=[]
    with open('data/combine/submission_xgb.csv', mode='r', encoding='utf-8') as f:
        lines=f.readlines()
        for item in lines:
            p3.append(float(item.split(',')[2]))
            
    prob=[]
    for i in range(len(p1)):
        prob.append(p1[i]*0.25+p3[i]*0.25+p2[i]*0.5)
        
    df_origin=pd.read_csv('data/origin/test2.csv')
    aid=df_origin.aid.values
    uid=df_origin.uid.values
    del df_origin
    
    output=pd.DataFrame({'aid':aid,'uid':uid,'score':prob})
    output['score']=output.score.apply(lambda x:round(x,8))
    
    columns = ['aid','uid','score']
    output.to_csv('result/submission_c.csv',index=False,columns=columns,header=None)
    
if __name__=='__main__':

   combineResult(0.5)#0.5暂定
   #combineResult2()
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import metrics

#lgbm训练代码

from sklearn.cross_validation import train_test_split
import scipy as sp

def loadCSV(path=''):
    return pd.read_csv(path) 
    
#训练分类器XGB
def trainClassifierLGBM(x_train,y_train):
    
    print('使用LIGHTBGM进行训练')
    lgb_train = lgb.Dataset(x_train, y_train)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        #'max_depth':3,
        'num_leaves': 2500, #2500(暂定)
        #'max_bin':150,  
        'learning_rate': 0.04,#0.04暂定)
        'feature_fraction': 0.4,
        'lambda_l1': 0.5,#0.5暂定
        'lambda_l2': 0.5,#0.5暂定
        #'bagging_fraction': 0.85,
        #'bagging_freq': 5,
        'verbose': 0,
    }
    
    #origin:learning_rate': 0.02，num_boost_round=700,'num_leaves': 1000
    
    lgbm = lgb.train(params,
                lgb_train,
                num_boost_round=700) #700（预测）
    
    print(params)
    
    return lgbm

#训练分类器XGB
def valClassifierLGBM(x_train,y_train,x_val,y_val):
    
    print('使用LIGHTBGM进行训练')
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_val = lgb.Dataset(x_val, y_val)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        #'max_depth':3,
        'num_leaves': 2500, #2500(暂定)
        #'max_bin':150,  
        'learning_rate': 0.04,#0.04(暂定)
        'feature_fraction': 0.4,
        'lambda_l1': 0.5,#0.5暂定
        'lambda_l2': 0.5,#0.5暂定
        #'bagging_fraction': 0.85,
        #'bagging_freq': 5,
        'verbose': 0,
    }
    
    #origin:learning_rate': 0.02，num_boost_round=700,'num_leaves': 1000
    
    lgbm = lgb.train(params,
                lgb_train,valid_sets=lgb_val,
                num_boost_round=700) #700（预测）
    
    print(params)
    
    return lgbm
    
#encoding代表是不是有独热编码信息（需要分解）
def predict_test_prob(lgbm,test_path):
    
    df_origin=loadCSV('data/origin/test2.csv')
    aid=df_origin.aid.values
    uid=df_origin.uid.values
    del df_origin
    
    df_all=loadCSV(test_path)     
    feature_all=df_all.values
    del df_all
    #feature_all=df_all.drop(['uid'],axis=1).values
    
    prob = lgbm.predict(feature_all, num_iteration=lgbm.best_iteration)
      
    output=pd.DataFrame({'aid':aid,'uid':uid,'score':prob})
    output['score']=output.score.apply(lambda x:round(x,8))
    
    columns = ['aid','uid','score']
    output.to_csv('result/submission.csv',index=False,columns=columns,header=None) 

#交叉验证

def cross_validat(train_path,test_size):
    
    print('开始载入')
     
    df_label=loadCSV('data/origin/train_label.csv')    
    label_all=df_label.label.values  
    del df_label
    
    #为了优化，从V6版特征开始，label就是单独载入的了
    df_all=loadCSV(train_path)
    feature_all=df_all.values
    del df_all
    #label_all=df_all.label.values
    #feature_all=df_all.drop(['label','uid'],axis=1).values
    

    print('开始交叉验证')
    
    len_all=len(feature_all)
    inter=int(len_all*(1-test_size))
    
    x_train=feature_all[:inter]
    x_test=feature_all[inter:]
    y_train=label_all[:inter]
    y_test=label_all[inter:]
    
    #x_train,x_test,y_train,y_test=train_test_split(feature_all,label_all,test_size=test_size,random_state=42)
    print('数据集切割完成')
    
    del feature_all
    del label_all
    
    lgbm=valClassifierLGBM(x_train,y_train,x_test,y_test)
    print('训练完成')
    
    prob = lgbm.predict(x_test, num_iteration=lgbm.best_iteration)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, prob,pos_label=1)
    auc=metrics.auc(fpr, tpr)
    print('AUC为:',auc)
    
    return prob
    
#输出预测结果
def predictSubmission(train_path,test_path):
      
    df_label=loadCSV('data/origin/train_label.csv')    
    label_all=df_label.label.values  
    del df_label
      
    df_all=loadCSV(train_path)           
    feature_all=df_all.values
    del df_all
    
    #feature_all=df_all.drop(['label','uid'],axis=1).values
    #label_all=df_all.label.values
   
    bst=trainClassifierLGBM(feature_all,label_all)
    print('分类器训练完成,开始预测')
   
    predict_test_prob(bst,test_path)
    print('结果预测完成')
    
    
#主函数入口
if __name__=='__main__':

   print('LGB')
   #pre=cross_validat('data/feature/train_v8.csv',0.2)
   predictSubmission('data/feature/train_v8.csv','data/feature/test2_v8.csv')

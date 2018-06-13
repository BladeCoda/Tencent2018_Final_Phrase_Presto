# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import metrics
from scipy import sparse

#lgbm训练代码

from sklearn.cross_validation import train_test_split
import scipy as sp

def loadCSV(path=''):
    return pd.read_csv(path) 
    
#训练分类器XGB
def trainClassifierLGBM(x_train,y_train,x_val,y_val,valid):
    
    print('使用LIGHTBGM进行训练')
    lgb_train = lgb.Dataset(x_train, y_train)
    if valid==True:
         lgb_val = lgb.Dataset(x_val, y_val)
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        #'max_depth':3,
        'num_leaves': 2500, #2500(暂定)
        #'max_bin':150,  
        'learning_rate': 0.02,#0.02(暂定)
        'feature_fraction': 0.4,
        'lambda_l1': 0.5,#0.5暂定
        'lambda_l2': 0.5,#0.5暂定
        #'bagging_fraction': 0.85,
        #'bagging_freq': 5,
        'verbose': 0,
    }
    
    #origin:learning_rate': 0.02，num_boost_round=700,'num_leaves': 1000
    
    if valid==True:
        lgbm = lgb.train(params,
                    lgb_train,valid_sets=lgb_val,
                    num_boost_round=700) #700（预测）
    else:
        lgbm = lgb.train(params,
                    lgb_train,num_boost_round=700) #700（预测）
    
    print(params)  
    return lgbm
    
#encoding代表是不是有独热编码信息（需要分解）
def predict_test_prob(lgbm):
    
    df_origin=loadCSV('../data/origin/test2.csv')
    aid=df_origin.aid.values
    uid=df_origin.uid.values
    del df_origin
    
    test_x=sparse.load_npz('data/test2_x.npz') 
    
    prob = lgbm.predict(test_x, num_iteration=lgbm.best_iteration)
      
    output=pd.DataFrame({'aid':aid,'uid':uid,'score':prob})
    output['score']=output.score.apply(lambda x:round(x,8))
    
    columns = ['aid','uid','score']
    output.to_csv('result/submission.csv',index=False,columns=columns,header=None) 
	output.to_csv('../data/combine/submission_bl.csv',index=False,columns=columns,header=None) 

#交叉验证

def cross_validat():
    
    print('loading files')
   
    train_x_val=sparse.load_npz('data/train_x_val.npz')
    test_x_val=sparse.load_npz('data/test_x_val.npz')
    train_y_val=loadCSV('data/train_y_val.csv').label.values
    test_y_val=loadCSV('data/test_y_val.csv').label.values
    
    lgbm=trainClassifierLGBM(train_x_val,train_y_val,test_x_val,test_y_val,True)
    print('训练完成')
    
    prob = lgbm.predict(test_x_val, num_iteration=lgbm.best_iteration)
    fpr, tpr, thresholds = metrics.roc_curve(test_y_val, prob,pos_label=1)
    auc=metrics.auc(fpr, tpr)
    print('AUC为:',auc)
    
    return prob
    
#输出预测结果
def predictSubmission():
      
    df_label=loadCSV('data/train_label.csv')    
    label_all=df_label.label.values  
    del df_label
    
    print('loading file........')
      
    train_x=sparse.load_npz('data/train_x.npz')
    train_y=label_all
   
    bst=trainClassifierLGBM(train_x,train_y,[],[],False)
    print('分类器训练完成,开始预测')
   
    predict_test_prob(bst)
    print('结果预测完成')
    
    
#主函数入口
if __name__=='__main__':

   print('lgb')
   #pre=cross_validat()

   predictSubmission()
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os

def cutDataForOneHot(per=0.3):
    print('开始载入')
    df_train=pd.read_csv('../data/merge/merge_train.csv')   
    print('start cutting....')
    df_train=df_train.sample(frac=per)
    print('saving')
    df_train.to_csv('data/train.csv') 
    
def combine(list1,list2):
    re=[]
    for i in range(len(list1)):
        re.append(str(list1[i])+str(list2[i]))
    return np.array(re)

def digOneHot(per=0.8):

    print('开始载入')
    df_train=pd.read_csv('data/train.csv')
    
    len_train=len(df_train)
    center=int(len_train*per)
    
    train_label=df_train['label'].values
    df_train_label=pd.DataFrame({'label':train_label})
    train_y_val=df_train_label[:center]
    test_y_val=df_train_label[center:]
     
    df_train_label.to_csv('data/train_label.csv',index=False)
    train_y_val.to_csv('data/train_y_val.csv',index=False)
    test_y_val.to_csv('data/test_y_val.csv',index=False)
    
    df_train=df_train.drop('label',axis=1)  
    df_test1=pd.read_csv('../data/merge/merge_test1.csv') 
    df_test2=pd.read_csv('../data/merge/merge_test2.csv') 
    len_test1=len(df_test1)
    
    df_all=df_train.append(df_test1)
    df_all=df_all.append(df_test2)
    del df_train
    del df_test1
    del df_test2
    
    df_all=df_all.fillna('-1')#fix none values
    
    one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender',
                                 'house','os','ct','marriageStatus','advertiserId','campaignId', 
                                 'creativeId','adCategoryId', 'productId', 'productType']
    vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3',
                                'interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
    combineFeature=['age','gender','consumptionAbility','education']
    
    print('begining combine.......')
    for feature in combineFeature:
        print('processing %s.......'%feature)
        df_all['app_'+feature]=combine(df_all['aid'].values,df_all[feature].values)
    
    print('normalize id.........')   
    for feature in combineFeature:
        print('processing %s'%feature)
        try:
            df_all['app_'+feature] = LabelEncoder().fit_transform(df_all['app_'+feature].apply(int))
        except:
            df_all['app_'+feature] = LabelEncoder().fit_transform(df_all['app_'+feature])
            
    for feature in one_hot_feature:
        print('processing %s'%feature)
        try:
            df_all[feature] = LabelEncoder().fit_transform(df_all[feature].apply(int))
        except:
            df_all[feature] = LabelEncoder().fit_transform(df_all[feature])
            
    df_train=df_all[:len_train]
    df_test1=df_all[len_train:len_train+len_test1]
    df_test2=df_all[len_train+len_test1:]
    df_val_train=df_all[:center]
    df_val_test=df_all[center:len_train]
    
    print('transforming OneHot features.........')
    enc = OneHotEncoder()
    train_x=df_train[['creativeSize']]
    test1_x=df_test1[['creativeSize']]
    test2_x=df_test2[['creativeSize']]
    train_x_val=df_val_train[['creativeSize']]
    test_x_val=df_val_test[['creativeSize']]
        
    for feature in one_hot_feature:
         print('processing %s'%feature)
         enc.fit(df_all[feature].values.reshape(-1, 1))
         
         train_a=enc.transform(df_train[feature].values.reshape(-1, 1))
         test1_a = enc.transform(df_test1[feature].values.reshape(-1, 1))
         test2_a = enc.transform(df_test2[feature].values.reshape(-1, 1))
         train_a_val=enc.transform(df_val_train[feature].values.reshape(-1, 1))
         test_a_val = enc.transform(df_val_test[feature].values.reshape(-1, 1))
         
         train_x= sparse.hstack((train_x, train_a))
         test1_x = sparse.hstack((test1_x, test1_a))
         test2_x = sparse.hstack((test2_x, test2_a))
         train_x_val = sparse.hstack((train_x_val, train_a_val))
         test_x_val = sparse.hstack((test_x_val , test_a_val ))
         
    print('one-hot prepared !')
    
    cv=CountVectorizer()
    for feature in vector_feature:
         print('processing %s'%feature)
         cv.fit(df_all[feature])
         
         train_a = cv.transform(df_train[feature])
         test1_a = cv.transform(df_test1[feature])
         test2_a = cv.transform(df_test2[feature])
         train_a_val = cv.transform(df_val_train[feature])
         test_a_val = cv.transform(df_val_test[feature])
         
         train_x = sparse.hstack((train_x, train_a))
         test1_x = sparse.hstack((test1_x, test1_a))
         test2_x = sparse.hstack((test2_x, test2_a))
         train_x_val = sparse.hstack((train_x_val, train_a_val))
         test_x_val = sparse.hstack((test_x_val, test_a_val))
         
    print('cv prepared !')
        
    print('transforming Combine features.........')
        
    for feature in combineFeature:
         print('processing %s'%feature)
         enc.fit(df_all['app_'+feature].values.reshape(-1, 1))
         train_a=enc.transform(df_train['app_'+feature].values.reshape(-1, 1))
         test1_a = enc.transform(df_test1['app_'+feature].values.reshape(-1, 1))
         test2_a = enc.transform(df_test2['app_'+feature].values.reshape(-1, 1))
         train_val_a=enc.transform(df_val_train['app_'+feature].values.reshape(-1, 1))
         test_val_a = enc.transform(df_val_test['app_'+feature].values.reshape(-1, 1))
         
         train_x= sparse.hstack((train_x, train_a))
         test1_x = sparse.hstack((test1_x, test1_a))
         test2_x = sparse.hstack((test2_x, test2_a))
         train_x_val= sparse.hstack((train_x_val,train_val_a))
         test_x_val= sparse.hstack((test_x_val, test_val_a))
    print('one-hot prepared !')     
    
    print('saving......')
    
    sparse.save_npz('data/train_x.npz',train_x)
    sparse.save_npz('data/test1_x.npz',test1_x)
    sparse.save_npz('data/test2_x.npz',test2_x)
    sparse.save_npz('data/train_x_val.npz',train_x_val)
    sparse.save_npz('data/test_x_val.npz',test_x_val)
    
if __name__=='__main__':
    cutDataForOneHot(per=0.5)#0.35
    digOneHot(per=0.8)

#对兴趣，关键词和主题做聚类特征(使用SkipGram聚类)
import logging
from gensim.models.word2vec import Word2Vec
from sklearn.cluster import KMeans
import pickle as pk
import pandas as pd

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)

#训练词向量
def trainingForWords(sentences,filepath,fnum):
    #设定词向量参数

    print('开始训练') 
    num_features=fnum #词向量的维度
    min_word_count=5 #词频数最低阈值
    num_workers=-1 #线程数,想要随机种子生效的话，设为1
    context=10 #上下文窗口大小
    downsampling=1e-3 #与自适应学习率有关
    num_iter=15
    hs=0
    sg=1#是否使用skip-gram模型
    
    model_path=filepath
    model=Word2Vec(sentences,workers=num_workers,hs=hs,
                   size=num_features,min_count=min_word_count,seed=77,iter=num_iter,
                   window=context,sample=downsampling,sg=sg)
    model.init_sims(replace=True)#锁定训练好的word2vec,之后不能对其进行更新 
    model.save(model_path)#讲训练好的模型保存到文件中
    print('训练完成')
    return model
    
#加载word2vec模型
def loadForWord(filepath):
    model=Word2Vec.load(filepath)
    print('word2vec模型读取完毕')
    return model
    
#将flist训练向量
def trainingForFeature(flist,filepath,fnum):
    re=[]
    for item in flist:
        if not isinstance(item,float):
            list_t=item.strip().split(' ')
            list_t.sort()
            re.append(list_t)
            del list_t
    del flist
    trainingForWords(re,filepath,fnum)
    
#将兴趣1，关键词1和主题1做embedding模型
def embeddingProcessing():
    print('开始载入')
    df_user=pd.read_csv('data/origin/userFeature_detail.csv') 
    
    list_k1=df_user['kw1'].values
    list_t1=df_user['topic1'].values 
    
    list_k2=df_user['kw2'].values
    list_t2=df_user['topic2'].values

    del df_user
    
    print('处理kw1')
    trainingForFeature(list_k1,'data/extra/embedding/k1_em.w2v',300)
    del list_k1
    
    print('处理topic1')
    trainingForFeature(list_t1,'data/extra/embedding/t1_em.w2v',300)
    del list_t1
    
    print('处理kw2')
    trainingForFeature(list_k2,'data/extra/embedding/k2_em.w2v',300)
    del list_k2
    
    print('处理topic2')
    trainingForFeature(list_t2,'data/extra/embedding/t2_em.w2v',300)
    del list_t2
    

    
def knnClusterForW2V(filepath,outpath,cluster_num):
    W2Vmodel=loadForWord(filepath)
    vocab=list(W2Vmodel.wv.vocab.keys())
    vectors=[W2Vmodel[vocab[i]] for i in (range(len(vocab)))]
    print('开始聚类')
    clf=KMeans(n_clusters=cluster_num,random_state=77)
    clf.fit(vectors)
    print('聚类完成，开始讲词典转化为类别字典')
    dict_re={vocab[i]:clf.labels_[i] for i in range(len(vocab))}
    print('保存字典。。。。')
    with open(outpath,'wb') as f:
        pk.dump(dict_re,f)
    return dict_re
    
def loadDict(dictpath):
    print('载入字典：%s'%dictpath)
    with open(dictpath,'rb') as f:
        dict_re=pk.load(f)
    return dict_re

def clusteringProcessing():

    print('对关键词1进行聚类')
    knnClusterForW2V('data/extra/embedding/k1_em.w2v',
                     'data/extra/cluster/k1.cl',100)
    
    print('对主题1进行聚类')
    knnClusterForW2V('data/extra/embedding/t1_em.w2v',
                     'data/extra/cluster/t1.cl',100)
    
    print('对关键词2进行聚类')
    knnClusterForW2V('data/extra/embedding/k2_em.w2v',
                     'data/extra/cluster/k2.cl',100)
    
    print('对主题2进行聚类')
    knnClusterForW2V('data/extra/embedding/t2_em.w2v',
                     'data/extra/cluster/t2.cl',100)
    
    
def getMaxCnum(clist):
    dic={}
    max_num=-1
    max_c=-1
    for c in clist:
        if c not in dic:
            dic[c]=1
            cur=1
        else:
            dic[c]+=1
            cur=dic[c]
        if cur>max_num:
            max_num=cur
            max_c=c
    return max_c    
    
def flist2clusterFeature(flist,dic,c_num):
    re=[]
    print('开始进行转化')
    for item in flist:
        if isinstance(item,float):
            re.append(c_num)
        else:
            clist=[]
            for i in item.strip().split(' '):
                if i not in dic:
                    clist.append(c_num)
                else:
                    clist.append(dic[i])
            re.append(getMaxCnum(clist))
    print('转化完成')
    return re
    
#目前验证i1,i2和t3,k3没有什么作用
def digEmCluster():
    print('开始载入')
    df_user=pd.read_csv('data/origin/userFeature_detail.csv') 
    df_out=pd.DataFrame({'uid':df_user['uid'].values})  
    
    list_k1=df_user['kw1'].values
    list_t1=df_user['topic1'].values

    list_k2=df_user['kw2'].values
    list_t2=df_user['topic2'].values

    del df_user
    
    print('处理kw1')
    dic=loadDict('data/extra/cluster/k1.cl')
    df_out['k1_ec']=flist2clusterFeature(list_k1,dic,100)
    del list_k1,dic
    
    print('处理topic1')
    dic=loadDict('data/extra/cluster/t1.cl')
    df_out['t1_ec']=flist2clusterFeature(list_t1,dic,100)
    del list_t1,dic
    
    print('处理kw2')
    dic=loadDict('data/extra/cluster/k2.cl')
    df_out['k2_ec']=flist2clusterFeature(list_k2,dic,100)
    del list_k2,dic
    
    print('处理topic2')
    dic=loadDict('data/extra/cluster/t2.cl')
    df_out['t2_ec']=flist2clusterFeature(list_t2,dic,100)
    del list_t2,dic
    
    print('开始保存')
    df_out.to_csv('data/extra/EmCluster.csv',index=False)

def mergeV5(dtype='train'):
    if dtype=='train':
        df_join=pd.read_csv('data/feature/train_v4.csv')
        outpath='data/feature/train_v5.csv'
    elif dtype=='test1':
        df_join=pd.read_csv('data/feature/test1_v4.csv')
        outpath='data/feature/test1_v5.csv'
    elif dtype=='test2':
        df_join=pd.read_csv('data/feature/test2_v4.csv')
        outpath='data/feature/test2_v5.csv'
    else:
        print('error type')
        return
    
    print('开始拼接聚类特征！')
    df_feature=pd.read_csv('data/extra/EmCluster.csv')
    df_join=pd.merge(df_join,df_feature,how='left',on='uid')#拼接position信息
    del df_feature
    
    print('拼接完成，开始保存')
    df_join.to_csv(outpath,index=False)
    print('保存完毕')
    
def reduceFile(dtype='train'):
    print('开始删减特征')
    if dtype=='train':
        df_all=pd.read_csv('data/feature/train_v5.csv')
        df_all=df_all.drop(['label','uid'],axis=1)
        outpath='data/feature/train_v5.csv'
    elif dtype=='test1':
        df_all=pd.read_csv('data/feature/test1_v5.csv')
        df_all=df_all.drop(['uid'],axis=1)
        outpath='data/feature/test1_v5.csv'
    elif dtype=='test2':
        df_all=pd.read_csv('data/feature/test2_v5.csv')
        df_all=df_all.drop(['uid'],axis=1)
        outpath='data/feature/test2_v5.csv'
    
    df_all=df_all.drop(['advertiserId','campaignId','pro_ad','adc_ad'],axis=1)
    print('删减完成，开始保存')
    df_all.to_csv(outpath,index=False)
             
if __name__=='__main__': 
    '''embeddingProcessing()
    clusteringProcessing()
    
    digEmCluster()'''
    
    mergeV5(dtype='train')
    mergeV5(dtype='test1')
    mergeV5(dtype='test2')
    
    reduceFile(dtype='train')
    reduceFile(dtype='test1')
    reduceFile(dtype='test2')
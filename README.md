qdrQA
===

[![Build Status](https://api.travis-ci.org/seomoz/qdr.png)](https://travis-ci.org/seomoz/qdr)

Query-Document Relevance ranking functions


承接[练习题 - 基于快速文本标题匹配的知识问答实现（一，基础篇）](https://blog.csdn.net/sinat_26917383/article/details/82224413)，前篇主要把qdr这个项目解剖了一下，现在开始应用做一下问答。

可以看到qdr这个项目的特点是：可以快速比对两个文本之间的相似性，而且计算tfidf、bm25、lm三款模型的速度很快。

那么本轮知识问答的设计源于此：

 - 先储备一批问答语料，一问一答比较合适；
 - 把问题进行分词,变为文本序列；
 - 载入qdr模型之中，进行训练；
	 - 先trainer，统计词条频次 / 单词存在的文档数量两个数据；
	 - Scoring，把trainer的统计数据`QueryDocumentRelevance`载入，变为文本集合；
 - 新查询句，分词；
 - 分词之后的查询句在model中比对，得到最大相似的query对，
 - 从query找到对应的answer

接下来会演示：一个极其简单的demo + 一部分百度问答语料的demo.


----------

## 目录
[toc]

----------


## 1 知识问答简单demo


```
import os
import unittest
import numpy as np
from qdr import ranker,trainer
from tempfile import mkstemp 
from qdr import QueryDocumentRelevance

class qdrQA(object):
    def __init__(self, query, document):
        self.query = query
        self.document = document
        assert len(self.query)==len(self.document), "Inconsistent length on both query and document."
        self.qd = self.TrainModel()
        self.scorer = QueryDocumentRelevance(self.qd._counts,self.qd._total_docs)
        
    def TrainModel(self):
    # 模型统计词条频次 / 单词存在的文档数量两个数据
        qd = trainer.Trainer()
        qd.train(self.query)
        return qd

    def update(self,query_update,document_update):
        # 模型update
        qd2 = trainer.Trainer()
        qd2.train(query_update)
        self.qd.update_counts_from_trained(qd2)   # 合并两个容器的训练集
        self.query = self.query + query_update
        self.document = self.document + document_update

    def QueryAnswer(self,input_sentence,select_model = 'tfidf',limit = 0):
    # 查询语句
        #query_scores = np.array([self.scorer.score(input_sentence,qu)[select_model] for qu in self.query])
        query_scores = np.array([qu[select_model] for qu in self.scorer.score_batch(input_sentence,self.query)])
        
        if query_scores.max() > limit:
            answer = self.document[query_scores.argmax()]
        else:
            answer = 'sorry,no match sentence.'
        return answer
```

以上就是基于qdr进行简单的封装，其中

 - `TrainModel()`是训练模块；
 - `update()`是如果有新的语料可以随机更新（非常方便！）；
 - `QueryAnswer()`，问答匹对。

进行简单测试：

```
# 数据集
query = [['信用积分','在','哪里','查询'],['蚂蚁积分','可以','兑换','什么','东西'],['信用积分','兑换','什么','性价比','比较','高']]
document = ['可以在首页查询','蚂蚁积分可以兑换商城中很多东西','信用积分性价比最高兑换物品是苹果手机']
# 建模
qdr = qdrQA(query,document)
# 问答
select_model = 'tfidf'
input_sentence = ['信用积分','查询']
limit = 0
print qdr.QueryAnswer(input_sentence,limit = 0)
>>> 可以在首页查询
```
以上是输入文字序列，其中query对需要分词，因为这样可以增加匹配概率。
那么如果新加语料，如何训练：

```
# 模型更新
query_update = [['信用积分','与','蚂蚁积分','的','区别']]
document_update = ['区别主要集中在商城兑换品']
qdr.update(query_update,document_update)

# 问答
select_model = 'tfidf'
input_sentence = ['信用积分','与','蚂蚁积分','区别']
print(qdr.QueryAnswer(input_sentence))
>>> 区别主要集中在商城兑换品
```
很方便的直接更新，只要与训练语料格式保持一致。


----------


## 2 部分百度问答语料的问答

语料在github中可加载得到：
```
import json 
import jieba


# 问答
def qaPrint(input_sentence,select_model = 'tfidf',limit = 0):
    query_scores = np.array([qu[select_model] for qu in qdr.scorer.score_batch(input_sentence,qdr.query)])
    similar_answer = ''.join(qdr.query[query_scores.argmax()])
    print 'query is : ', ''.join(input_sentence) 
    print 'most similar query is : ', similar_answer
    print 'answer is :',qdr.QueryAnswer(input_sentence,limit = 0) 
    
def QueryJieba(input_sen):
    return [i.encode('utf-8') for i in list(jieba.cut(input_sen))]

# 数据准备
qa = open("/mnt/qdr/me_test.ann.json", "r").read()
qa = eval(qa)

query_bd = []
answer_bd = []
for qa_ in qa.values():
    if (qa_['question']!='') and (qa_['evidences'].values()[0]['evidence']!=''):
        query_bd.append(qa_['question'])
        answer_bd.append(qa_['evidences'].values()[0]['evidence'])
        
# jieba
query_bd_jieba = [list(jieba.cut(wo.decode('unicode-escape'))) for wo in query_bd]

# format processing
query_bd_jieba = [[i.encode('utf-8') for i in q]  for q in query_bd_jieba]
answer_bd = [q.encode('utf-8')  for q in answer_bd]

# 模型训练
qdr = qdrQA(query_bd_jieba,answer_bd)

# 提问
input_sen = '沙漠最大的叫什么？'
qaPrint(QueryJieba(input_sen))

>>> query is :  沙漠最大的叫什么？
>>> most similar query is :  世界上最大的沙漠叫什么名字?
>>> answer is : 撒哈拉沙漠撒哈拉沙漠面积为860万平方公里，是世界上最大的沙漠，占据了北非大部分地区。

# 提问2
input_sen = '最浅的海是哪里'
qaPrint(QueryJieba(input_sen))

>>> query is :  最浅的海是哪里
>>> most similar query is :  世界上最浅的海叫什么？
>>> answer is : 亚速海平均深度8米，最深处也只有14米，是世界上最浅的海记得采纳啊

```
加载数据，把问题数据进行jieba分词，其中，qdr模型接受utf-8格式,需要转化一下格式。
这边简单写了一下，提问之后，返回给你最相似的问题以及对应的答案。


----------


## 延伸：单独来看，一些小模块的应用：
### 1、 获得该批语料单词的idf值

```
qdr.scorer.get_idf('沙漠')
>>> 7.321188556739478
```


### 2、单独的文本匹配模块

```
qdr.scorer.score_batch(QueryJieba('沙漠最大的叫什么？'),[QueryJieba('最浅的海是哪里')])
>>> [{'bm25': 0.43801802356073943,
  'lm_ad': -28.28692876400435,
  'lm_dirichlet': -27.58677603082954,
  'lm_jm': -33.64719347947683,
  'tfidf': 0.014088049093832688}]
```

# Installing

```
sudo pip install -r requirements.txt
sudo make install
```

# Contributing
Contributions welcome!  Fork, commit, then open a pull request.



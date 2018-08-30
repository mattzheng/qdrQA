# -*- coding: utf-8 -*-

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
        #query_scores = np.array([self.scorer.score(input_sentence,qu)[select_model] for qu in self.query])
        query_scores = np.array([qu[select_model] for qu in self.scorer.score_batch(input_sentence,self.query)])
        
        if query_scores.max() > limit:
            answer = self.document[query_scores.argmax()]
        else:
            answer = 'sorry,no match sentence.'
        return answer

if __name__ == '__main__':
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

    # 模型更新
    query_update = [['信用积分','与','蚂蚁积分','的','区别']]
    document_update = ['区别主要集中在商城兑换品']
    qdr.update(query_update,document_update)
    
    # 问答
    select_model = 'tfidf'
    input_sentence = ['信用积分','与','蚂蚁积分','区别']
    print(qdr.QueryAnswer(input_sentence))


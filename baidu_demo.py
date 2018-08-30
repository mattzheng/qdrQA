# -*- coding: utf-8 -*-

import json 
import jieba
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



def qaPrint(input_sentence,select_model = 'tfidf',limit = 0):
    query_scores = np.array([qu[select_model] for qu in qdr.scorer.score_batch(input_sentence,qdr.query)])
    similar_answer = ''.join(qdr.query[query_scores.argmax()])
    print 'query is : ', ''.join(input_sentence) 
    print 'most similar query is : ', similar_answer
    print 'answer is :',qdr.QueryAnswer(input_sentence,limit = 0).decode('unicode-escape')
    
def QueryJieba(input_sen):
    return [i.encode('utf-8') for i in list(jieba.cut(input_sen))]   

if __name__ == '__main__':
    #load data
    qa = open("me_test.ann.json", "r").read()
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
    
    # 建模
    qdr = qdrQA(query_bd_jieba,answer_bd)
    # 问答1
    qaPrint(query_bd_jieba[0])
    # 问答2
    input_sen = '沙漠最大的叫什么？'
    qaPrint(QueryJieba(input_sen))
    # 问答3
    input_sen = '最浅的海是哪里'
    qaPrint(QueryJieba(input_sen))
    # 求idf
    qdr.scorer.get_idf('沙漠')
    # 文本匹配
    qdr.scorer.score_batch(QueryJieba('沙漠最大的叫什么？'),[QueryJieba('最浅的海是哪里')])

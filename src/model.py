#!/usr/bin/env python
# _*_ coding:utf-8 _*_


# from src.Roberta import MultiHeadAttention
from transformers import AutoModel, AutoConfig
from src.common import MultiHeadAttention

import torch
import torch.nn as nn
from itertools import accumulate
import pdb
import numpy as np
class BertWordPair(nn.Module):
    def __init__(self, cfg):
        super(BertWordPair, self).__init__()
        self.bert = AutoModel.from_pretrained(cfg.bert_path)
        bert_config = AutoConfig.from_pretrained(cfg.bert_path)

        self.dense_layers = nn.ModuleDict({
            'ent': nn.Linear(bert_config.hidden_size, cfg.inner_dim * 4 * 6),
            'rel': nn.Linear(bert_config.hidden_size, cfg.inner_dim * 4 * 3),
            'pol': nn.Linear(bert_config.hidden_size, cfg.inner_dim * 4 * 4)
        })

        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        att_head_size = int(bert_config.hidden_size / bert_config.num_attention_heads)

        self.reply_attention = MultiHeadAttention(bert_config.num_attention_heads, bert_config.hidden_size, att_head_size, att_head_size, bert_config.attention_probs_dropout_prob)
        self.speaker_attention = MultiHeadAttention(bert_config.num_attention_heads, bert_config.hidden_size, att_head_size, att_head_size, bert_config.attention_probs_dropout_prob)
        self.thread_attention = MultiHeadAttention(bert_config.num_attention_heads, bert_config.hidden_size, att_head_size, att_head_size, bert_config.attention_probs_dropout_prob)

        self.cfg = cfg 

    def thread_classify_matrix(self, kwargs, thread_sequence_output, Train,mat_name='ent'):
        thread_utterance_index, thread_token_index, thread_lengths = [kwargs[w] for w in ['thread_utterance_index', 'thread_token_index','thread_sentence_spans']]
        thread_utterance_spans  = kwargs['thread_utterance_spans']
        thread_max_lens = thread_sequence_output.shape[2] 
        thread_nums = [len(sublist) for sublist in thread_utterance_spans] #[4,] thread的个数
        thread_masks = kwargs['thread_sentence_masks'] if mat_name == 'ent' else kwargs['thread_full_masks']
        batch = self.cfg.batch_size
        # pdb.set_trace()
        thread_lengths = thread_lengths.squeeze()# 去除第二维度
        loss = 0.0
        pred_logits = []
        for i in range(batch):
            for j in range(thread_nums[i]):
                thread_output = thread_sequence_output[i][j]
                dense_layer = self.dense_layers[mat_name]
                thread_output = dense_layer(thread_output) 
                thread_output = torch.split(thread_output,self.cfg.inner_dim*4, dim=-1)
                thread_output = torch.stack(thread_output,dim=-2) 
                t_q_token,t_q_utterance,t_k_token,t_k_utterance = torch.split(thread_output,self.cfg.inner_dim,dim=-1) 
                thread_pred_logits = torch.einsum('bmh,xnh->bxn', t_q_token, t_k_token) 
                t_s,t_e = thread_utterance_spans[i][j][0].item(),thread_utterance_spans[i][j][1].item()
                if(t_e>=thread_max_lens or t_s>=thread_max_lens):
                    t_e = t_e-t_s
                    t_s = 0
                thread_pred_logits = thread_pred_logits[t_s:t_e+1,t_s:t_e+1,:] #([11, 11, 6])
                thread_input_labels,thread_active_labels,thread_active_loss = self.ThreadMetrix(kwargs[f'thread_{mat_name}_matrix'], thread_masks,thread_utterance_spans,i,j,)
               
                thread_active_logits = thread_pred_logits.reshape(-1, thread_pred_logits.shape[-1])[thread_active_loss] 
                nums = thread_pred_logits.shape[-1]
                criterion = nn.CrossEntropyLoss(thread_output.new_tensor([1.0] + [self.cfg.loss_weight[mat_name]] * (nums - 1)))
                thread_loss = criterion(thread_active_logits,thread_active_labels)
                loss += thread_loss
        return loss
    def ThreadMetrix(self,matrix,thread_mask,thread_utterance_spans,BatchIndex,ThreadIndex):
        t_s,t_e = thread_utterance_spans[BatchIndex][ThreadIndex][0].item(),thread_utterance_spans[BatchIndex][ThreadIndex][1].item()
        input_labels = matrix[BatchIndex, t_s:t_e+1, t_s:t_e+1]
        active_loss = thread_mask[BatchIndex, t_s:t_e+1, t_s:t_e+1].reshape(-1) == 1
        active_labels = input_labels.reshape(-1)[active_loss] #(125*125)
        return input_labels,active_labels,active_loss
    def WindowsMetrix(self,matrix,windows_mask,windows_utterance_spans,BatchIndex,WindowsIndex,length):
        w_span = windows_utterance_spans[BatchIndex][WindowsIndex] #若干坐标组成的，如（1，11，14，62）
        root = windows_utterance_spans[0][0] #第一句话的索引

        input_labels = []
        for idx1 in range(0,len(w_span),2):
            region = []
            for idx2 in range(0,len(w_span),2):
                region1 = matrix[BatchIndex,w_span[idx1]:w_span[idx1+1]+1, w_span[idx2]:w_span[idx2+1]+1]
                region.append(region1)
            region = torch.cat(region, axis=1)
            input_labels.append(region)
        input_labels = torch.cat(input_labels, axis=0)
        
        active_loss = windows_mask[BatchIndex,WindowsIndex,1:length-1,1:length-1].reshape(-1) == 1 #全1的半角矩阵 不包括cls之类的
        active_labels = input_labels.reshape(-1)[active_loss] #(125*125)
        return input_labels,active_labels,active_loss
    def windows_classify_matrix(self, kwargs, windows_sequence_output, Train,mat_name='ent'):
        windows_utterance_spans =  kwargs['windows_utterance_spans']
        windows_masks = kwargs['windows_sentence_masks'] if mat_name == 'ent' else kwargs['windows_full_masks']
        windows_lengths = kwargs['windows_lengths']
        windows_max_lens = windows_sequence_output.shape[2] 
        windows_nums = [len(sublist) for sublist in windows_utterance_spans] #[26]
        batch = self.cfg.batch_size
        windows_loss = 0.0
        losses = 0.0
        pred_logits = []
        input_labels = []
        active_labels = []
        for i in range(batch):
            pred_logit = []
            input_label = []
            for j in range(windows_nums[i]):
                windows_output = windows_sequence_output[i][j] 
                dense_layer = self.dense_layers[mat_name]
                windows_output = dense_layer(windows_output)
                windows_output = torch.split(windows_output,self.cfg.inner_dim*4, dim=-1) 
                windows_output = torch.stack(windows_output,dim=-2) 
                w_q_token,w_q_utterance,w_k_token,w_k_utterance = torch.split(windows_output,self.cfg.inner_dim,dim=-1) 
                windows_pred_logits = torch.einsum('bmh,xnh->bxn', w_q_token, w_k_token)
                windows_pred_logits = windows_pred_logits[1:windows_lengths[i][j]-1,1:windows_lengths[i][j]-1,:] 
                windows_input_labels,windows_active_labels,windows_active_loss = self.WindowsMetrix(kwargs[f'{mat_name}_matrix'], windows_masks,windows_utterance_spans,i,j,windows_lengths[i][j])
                
                windows_active_logits = windows_pred_logits.reshape(-1, windows_pred_logits.shape[-1])[windows_active_loss] #[121,6]
                nums = windows_pred_logits.shape[-1]
                criterion = nn.CrossEntropyLoss(windows_output.new_tensor([1.0] + [self.cfg.loss_weight[mat_name]] * (nums - 1)))
                w_loss = criterion(windows_active_logits,windows_active_labels)
                windows_loss += w_loss
                pred_logit.append(windows_pred_logits)
                input_label.append(windows_input_labels)
            pred_logits.append(pred_logit)
            input_labels.append(input_label)
            active_labels.append(windows_active_labels)
       
        dialogue_loss,thread_loss,tags = self.inference(pred_logits,input_labels,kwargs['windows_utterance_spans_raw'],windows_lengths,criterion,Train) 
        """--------------------------Ablation----------------------------------------"""
        # losses = dialogue_loss + thread_loss  
        # losses = dialogue_loss + windows_loss
        # losses = thread_loss + windows_loss
        losses = windows_loss + dialogue_loss + thread_loss
        return losses,tags
    
    def inference(self,pred_logits,gold_logits,spans,lengths,criterion,Train):
        """
        对一个线程{A,B,C,D}四句话. f()为预测的答案  g()为推理后的答案
            g(A) = f(A)
            g(A,B) = f(A,B) + g(A) + g(B)
            g(A,B,C) = f(A,B,C) + g(A,B) + g(B,C)    
        """
        thread_windows_nums = self.calWindowNumsInThread(spans)
        window_len,pred_thread_logits,gold_thread_logits,thread_spans = self.windows2thread(pred_logits,gold_logits,lengths,thread_windows_nums,spans)
        
        final_thread_gold_logits = self.LabelCombineWindows2Thread(gold_thread_logits) #用于后面计算对话级的loss
        final_thread_pred_logits = self.PredCombineWindows2Thread(pred_thread_logits,gold_thread_logits,window_len,criterion,Train,dynamic=True) 
        thread_loss = self.threadLoss(final_thread_pred_logits,criterion,gold_thread_logits)
        final_preds = self.thread2dialogue(final_thread_pred_logits,window_len,thread_spans)
        final_labels = self.thread2dialogue(final_thread_gold_logits,window_len,thread_spans,False)
       
        dialogue_loss = self.dialogueLoss(final_preds,final_labels,criterion)
        final_loss = dialogue_loss + thread_loss 
        return dialogue_loss,thread_loss,final_preds
    
    def calWindowNumsInThread(self,spans):
        thread_windows_nums = []
        for span in spans:
            lens = 0
            nums = []
            for i,sp in enumerate(span):
                # pdb.set_trace()
                if sp == span[0] and i>0:
                    nums.append(lens)
                    lens = 1
                else:
                    lens += 1
                if i == len(span)-1:
                    nums.append(lens)
            thread_windows_nums.append(nums)
        return thread_windows_nums
    
    def dialogueLoss(self,pred,labels,criterion):
        loss = 0.0
        for i,logits in enumerate(pred):
            active_logits = logits.reshape(-1, logits.shape[-1])
            active_labels = labels[i].reshape(-1) 
            loss += criterion(active_logits,active_labels)
        return loss
    def LabelCombineWindows2Thread(self,logits):
        final_logits = []
        for i,thread_logit in enumerate(logits): #batch
            final_logit = []
            for j,single_thread in enumerate(thread_logit): #thread
                final_logit.append(single_thread[-1])
            final_logits.append(final_logit)
        return final_logits
    

    def PredCombineWindows2Thread(self,pred_logits,gold_logits,window_len,criterion,Train,dynamic=True):
        final_logits = []
        for i,thread_logit in enumerate(pred_logits): #batch
            final_logit = []
            for j,single_thread in enumerate(thread_logit): #thread
                fusion = 0 #初始化 0代表sum 1代表替代 2代表不变
                windows_nums = len(single_thread)
                st = window_len[i][j]
                if windows_nums == 3: #root+1sents
                    #g(A,B) = g(A)+g(B)+f(A,B)
                    if dynamic and Train:
                        tmp1 = single_thread[-1][:st[0],:st[0],:]
                        tmp2 = single_thread[-1][st[0]:st[0]+st[1],st[0]:st[0]+st[1],:]
                        f1,f2 = self.CalFusionScore(tmp1, single_thread[0],gold_logits[i][j][0],criterion) #(1,1)代表加法 (1,0)代表不变，(0,1)代表小窗口替换
                        f3,f4 = self.CalFusionScore(tmp2, single_thread[1],gold_logits[i][j][1],criterion)
                        single_thread[-1][:st[0],:st[0],:] = f1*tmp1 + f2*single_thread[0]
                        single_thread[-1][st[0]:st[0]+st[1],st[0]:st[0]+st[1],:] = f3*tmp2 + f4*single_thread[1]
                    else:#不执行动态选择窗口
                        single_thread[-1][:st[0],:st[0],:] += single_thread[0]
                        single_thread[-1][st[0]:st[0]+st[1],st[0]:st[0]+st[1],:] += single_thread[1]
                
                elif windows_nums == 6: #root+2sents ABC     
                    if dynamic and Train: #只处理窗口<2的
                        """rep3和tem3表示同一个位置，但不同窗口"""
                        tmp1 = single_thread[3][:st[0],:st[0],:] #A
                        tmp2 = single_thread[3][st[0]:st[0]+st[1],st[0]:st[0]+st[1],:] #B
                        rep2 = single_thread[4][:st[1],:st[1],:] #又是一个B
                        tmp3 = single_thread[4][st[1]:st[1]+st[2],st[1]:st[1]+st[2],:] #C
                        tmp4 = single_thread[5][:st[3],:st[3],:] #AB
                        tmp5 = single_thread[5][st[0]:st[0]+st[4],st[0]:st[0]+st[4],:]#BC
                        f1,f2 = self.CalFusionScore(tmp1, single_thread[0],gold_logits[i][j][0],criterion)
                        f3,f4 = self.CalFusionScore(tmp2, single_thread[1],gold_logits[i][j][1],criterion)
                        g3,g4 = self.CalFusionScore(rep2,single_thread[1],gold_logits[i][j][1],criterion)
                        f5,f6 = self.CalFusionScore(tmp3,single_thread[2],gold_logits[i][j][2],criterion)
                        f11,f22 =  self.CalFusionScore(tmp4,single_thread[3],gold_logits[i][j][3],criterion)
                        f33,f44 =  self.CalFusionScore(tmp5,single_thread[4],gold_logits[i][j][4],criterion)
                        single_thread[3][:st[0],:st[0],:] = f1*tmp1 + f2*single_thread[0]
                        single_thread[3][st[0]:st[0]+st[1],st[0]:st[0]+st[1],:] = f3*tmp2 + f4*single_thread[1]
                        single_thread[4][:st[1],:st[1],:] = g3*rep2 + g4*single_thread[1] 
                        single_thread[4][st[1]:st[1]+st[2],st[1]:st[1]+st[2],:] = f5*tmp3 + f6*single_thread[2]
                        single_thread[5][:st[3],:st[3],:] = f11*tmp4 + f22*single_thread[3]
                        single_thread[5][st[0]:st[0]+st[4],st[0]:st[0]+st[4],:] = f33*tmp5 + f44*single_thread[4]
                    else:
                        single_thread[3][:st[0],:st[0],:] += single_thread[0]
                        single_thread[3][st[0]:st[0]+st[1],st[0]:st[0]+st[1],:] += single_thread[1]
                        #BC = B + C
                        single_thread[4][:st[1],:st[1],:] += single_thread[1] 
                        single_thread[4][st[1]:st[1]+st[2],st[1]:st[1]+st[2],:] += single_thread[2]
                        # ABC = AB + BC 
                        single_thread[5][:st[3],:st[3],:] += single_thread[3]
                        single_thread[5][st[0]:st[0]+st[4],st[0]:st[0]+st[4],:] += single_thread[4]
                    # pdb.set_trace()
                elif windows_nums == 10: #root+3sents
                    if dynamic and Train:
                        tmp1 = single_thread[4][:st[0],:st[0],:]
                        tmp2 = single_thread[4][st[0]:st[0]+st[1],st[0]:st[0]+st[1],:]
                        rep2 = single_thread[5][:st[1],:st[1],:]
                        tmp3 = single_thread[5][st[1]:st[1]+st[2],st[1]:st[1]+st[2],:]
                        rep3 = single_thread[6][:st[2],:st[2],:]
                        tmp4 = single_thread[6][st[2]:st[2]+st[3],st[2]:st[2]+st[3],:]
                        f1,f2 = self.CalFusionScore(tmp1, single_thread[0],gold_logits[i][j][0],criterion) #A
                        f3,f4 = self.CalFusionScore(tmp2, single_thread[1],gold_logits[i][j][1],criterion) #B
                        g3,g4 = self.CalFusionScore(rep2, single_thread[1],gold_logits[i][j][1],criterion) #B
                        f5,f6 = self.CalFusionScore(tmp3, single_thread[2],gold_logits[i][j][2],criterion) #C
                        g5,g6 = self.CalFusionScore(rep3, single_thread[2],gold_logits[i][j][2],criterion) #C
                        f7,f8 = self.CalFusionScore(tmp4, single_thread[3],gold_logits[i][j][3],criterion) #D
                        single_thread[4][:st[0],:st[0],:] = f1*tmp1 + f2*single_thread[0] 
                        single_thread[4][st[0]:st[0]+st[1],st[0]:st[0]+st[1],:] = f3*tmp2 + f4*single_thread[1]
                        #BC = B+C
                        single_thread[5][:st[1],:st[1],:] = g3*rep2 + g4*single_thread[1] 
                        single_thread[5][st[1]:st[1]+st[2],st[1]:st[1]+st[2],:] = f5*tmp3 + f6*single_thread[2]
                        #CD= C+D
                        single_thread[6][:st[2],:st[2],:] = g5*rep3 +g6*single_thread[2] 
                        single_thread[6][st[2]:st[2]+st[3],st[2]:st[2]+st[3],:] = f7*tmp4 + f8*single_thread[3]

                        tmp5 = single_thread[7][:st[4],:st[4],:] #AB
                        tmp6 = single_thread[7][st[0]:st[0]+st[5],st[0]:st[0]+st[5],:] #BC
                        rep6 = single_thread[8][:st[5],:st[5],:] #BC
                        tmp7 = single_thread[8][st[1]:st[1]+st[6],st[1]:st[1]+st[6],:] #CD
                        f11,f22 = self.CalFusionScore(tmp5, single_thread[4],gold_logits[i][j][4],criterion) #A
                        f33,f44 = self.CalFusionScore(tmp6, single_thread[5],gold_logits[i][j][5],criterion) #B
                        g33,g44 = self.CalFusionScore(rep6, single_thread[5],gold_logits[i][j][5],criterion) #B
                        f55,f66 = self.CalFusionScore(tmp7, single_thread[6],gold_logits[i][j][6],criterion) #C
                        #ABC = AB +BC
                        single_thread[7][:st[4],:st[4],:] = f11*tmp5 +  f22*single_thread[4]
                        single_thread[7][st[0]:st[0]+st[5],st[0]:st[0]+st[5],:] = f33*tmp6 + f44*single_thread[5]
                        #BCD = BC+CD
                        single_thread[8][:st[5],:st[5],:] = g33*rep6+ g44*single_thread[5]
                        single_thread[8][st[1]:st[1]+st[6],st[1]:st[1]+st[6],:] =  f55*tmp7 + f66*single_thread[6]
                    else:   
                        single_thread[4][:st[0],:st[0],:] += single_thread[0] 
                        single_thread[4][st[0]:st[0]+st[1],st[0]:st[0]+st[1],:] += single_thread[1]
                        #BC = B+C
                        single_thread[5][:st[1],:st[1],:] += single_thread[1] 
                        single_thread[5][st[1]:st[1]+st[2],st[1]:st[1]+st[2],:] += single_thread[2]
                        #CD = C+D
                        single_thread[6][:st[2],:st[2],:] += single_thread[2] 
                        single_thread[6][st[2]:st[2]+st[3],st[2]:st[2]+st[3],:] += single_thread[3]
                        #ABC = AB+BC
                        single_thread[7][:st[4],:st[4],:] += single_thread[4]
                        single_thread[7][st[0]:st[0]+st[5],st[0]:st[0]+st[5],:] += single_thread[5]
                        #BCD = BC+CD
                        single_thread[8][:st[5],:st[5],:] += single_thread[5]
                        single_thread[8][st[1]:st[1]+st[6],st[1]:st[1]+st[6],:] += single_thread[6]
                    #ABCD = ABC + BCD
                    single_thread[9][:st[7],:st[7],:] += single_thread[7]
                    single_thread[9][st[0]:st[0]+st[8],st[0]:st[0]+st[8],:] += single_thread[8]
                else:
                    print(f"only one root! and windows nums = {windows_nums}")
                    continue
                final_logit.append(single_thread[-1]) #保存一个thread中的最后一个窗口的logits 即完整的
            final_logits.append(final_logit)#batch级   
        return final_logits
    def CalFusionScore(self,mat1,mat2,gold_mat,criterion):
        """ mat1 代表大窗口中对应的小窗口索引的内容， mat2代表小窗口的内容。"""
        f1,f2 = 0,0
        m1 = mat1.reshape(-1,mat1.shape[-1])
        m2 = mat2.reshape(-1,mat2.shape[-1])
        gold_m = gold_mat.reshape(-1)
        loss1 = criterion(m1,gold_m).item()
        loss2 = criterion(m2,gold_m).item() #
        
        loss3 = criterion(m1+m2,gold_m).item()
        # pdb.set_trace()
        if loss3<=min(loss2,loss1):
            return (1,1) #0代表加法融合
        elif loss2<=min(loss1,loss3):
            return (0,1) #1表示小窗口覆盖
        elif loss1<=min(loss2,loss3):
            return (1,0) #表示不对大窗口进行任何处理
    def windows2thread(self,pred_logits,gold_logits,lengths,thread_windows_nums,spans):
        window_len = []
        w_len = []
        t_logits = []
        g_logits = []
        t_span = []
        thread_logits = []
        gold_thread_logits = []
        thread_spans = []
        for i,nums in enumerate(thread_windows_nums):
            start = 0
            for size in nums:
                end = start + size
                # pdb.set_trace()
                w_len.append(lengths[i][start:end])
                t_logits.append(pred_logits[i][start:end])
                g_logits.append(gold_logits[i][start:end])
                t_span.append(spans[i][start:end])
                start = end
            window_len.append(w_len)
            thread_logits.append(t_logits)
            gold_thread_logits.append(g_logits)
            thread_spans.append(t_span)
        window_len = [[length - 2 for length in sublist] for sublist in window_len]    
        
        return window_len,thread_logits,gold_thread_logits,thread_spans



    def thread2dialogue(self,final_logits,window_len,thread_spans,pred=True):
        """
        将t个线程合并到一个matrix
        """
        loss = 0.0
        tags = []
        max_lens=0
        thread_nums = len(window_len[0])
        for i,lens in enumerate(window_len):
            sum = 0
            for j,w_len in enumerate(lens):
                sum += w_len[-1]
            sum -=(thread_nums-1)*lens[0][0]
            if sum > max_lens:
                max_lens = sum
        if pred==False: #gold
            matrix = torch.zeros([len(final_logits), max_lens, max_lens],dtype=torch.long).to(self.cfg.device)
        else:
            matrix = torch.zeros([len(final_logits), max_lens, max_lens,final_logits[0][0].shape[-1]]).to(self.cfg.device)
        for i,logits in enumerate(final_logits):
            for j,logit in enumerate(logits):
                span = thread_spans[i][j][-1]
                start = 0
                if(j==0): #ABCD 可以直接划分完整区域
                    matrix[i,span[0]:span[-1]+1,span[0]:span[-1]+1] = logit
                else: #默认线程至少2个句子
                    matrix[i,:span[1]+1,span[2]:span[-1]+1] = logit[:span[1]+1,span[1]+1:] #行坐标为root的范围，列坐标为DE的范围
                    matrix[i,span[2]:span[-1]+1,:span[1]+1] = logit[span[1]+1:,:span[1]+1] #行坐标为其余句子的范围，列坐标为root的范围
                    matrix[i,span[2]:span[-1]+1,span[2]:span[-1]+1] = logit[span[1]+1:,span[1]+1:] # DE的正方形面积 
        return matrix
    def threadLoss(self,pred_logits,criterion,gold_thread_logits):
        """ 计算推理后得到的线程级别的loss"""
        thread_loss = 0.0
        t_loss = 0.0
        for i,pred_thread_logits in enumerate(pred_logits):
            for j,thread_logit in enumerate(pred_thread_logits): #thread_logit: [100,100,6]    
                active_thread_logits = thread_logit.reshape(-1, thread_logit.shape[-1])
                active_labels = gold_thread_logits[i][j][-1].reshape(-1) #label最后一个窗口，代表thread线程级的[abcd]
                t_loss += criterion(active_thread_logits,active_labels)
            thread_loss +=t_loss
        return thread_loss
    
    def forward(self,Train=True, **kwargs):
        windows_input_ids, windows_input_masks, windows_input_segments = [kwargs[w] for w in ['windows_input_ids', 'windows_input_masks', 'windows_input_segments']]
        windows_sequence_outputs = self.bert(windows_input_ids, token_type_ids=windows_input_segments, attention_mask=windows_input_masks)[0] #([26, 165, 1024])
        windows_sequence_outputs = windows_sequence_outputs.unsqueeze(0) 
        windows_sequence_outputs = self.dropout(windows_sequence_outputs)
        
        w_loss0, w_tags0 = self.windows_classify_matrix(kwargs,windows_sequence_outputs,Train,'ent')
        w_loss1, w_tags1 = self.windows_classify_matrix(kwargs,windows_sequence_outputs,Train,'rel')
        w_loss2, w_tags2 = self.windows_classify_matrix(kwargs,windows_sequence_outputs,Train,'pol')
        
        return (w_loss0,w_loss1,w_loss2),(w_tags0,w_tags1,w_tags2)
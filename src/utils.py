#!/usr/bin/env python

import torch
import numpy as np
from attrdict import AttrDict
from scipy.linalg import block_diag
from collections import defaultdict
from attrdict import AttrDict 
import pdb
from torch.utils.data import Dataset, DataLoader
import os
import pickle as pkl
import random
from loguru import logger
import json

from src.common import WordPair
from src.preprocess import Preprocessor
# from src.run_eval1 import Template as Run_eval
from src.run_eval import Template as Run_eval

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class MyDataLoader:
    def __init__(self, cfg):
        path = os.path.join(cfg.preprocessed_dir, '{}_{}.pkl'.format(cfg.lang, cfg.bert_path.replace('/', '-')))
        preprocessor = Preprocessor(cfg)
        
        data = None
        if not os.path.exists(path):
            logger.info('Preprocessing data...')
            data = preprocessor.forward()
            logger.info('Saving preprocessed data to {}'.format(path))
            if not os.path.exists(cfg.preprocessed_dir):
                os.makedirs(cfg.preprocessed_dir)
            pkl.dump(data, open(path, 'wb'))
        
        logger.info('Loading preprocessed data from {}'.format(path))
        self.data = pkl.load(open(path, 'rb')) if data is None else data

        self.kernel = WordPair()
        self.config = cfg 

    def worker_init(self, worked_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    def collate_fn(self, lst):
        # doc_id, input_ids, input_masks, input_segments,sentence_length, token2sents, utterance_index, \
        #     token_index, thread_length, token2speaker, reply_mask, speaker_mask, thread_mask, pieces2words, new2old, \
        #         triplets, pairs, entity_list, rel_list, polarity_list = zip(*lst)
        doc_id, input_ids, input_masks, input_segments,sentence_length,sentence_length_raw,token2sents, token2sents_raw,utterance_index,\
        token_index, thread_length, token2speaker, reply_mask, speaker_mask, thread_mask, pieces2words, new2old, \
        triplets, pairs, entity_list, rel_list, polarity_list,\
        thread_input_ids,thread_input_masks,thread_input_segments,thread_utterance_spans,\
        thread_triplets, thread_pairs, thread_entity_list, thread_rel_list, thread_polarity_list,thread_utterance_index,thread_sentence_spans,\
        flatten_all_windows_tokens_spans,flatten_all_windows_tokens_spans_raw,windows_input_ids,windows_input_masks,windows_input_segments,windows_token2sents1D,windows_token2sents,windows_length = zip(*lst)
        """
        windows_sentence_length:[batch,len] 2层list[[136, 40, 138, 80, 174, 176, 216, 310, 254, 388, 136, 7, 20, 44, 141, 25, 62, 159, 67, 201, 136, 16, 150]]
        windows_token2sents:[batch,thread_nums,windows_num] 3层list [ [[ ]],[[ ]] ]
        triplets和pairs和各种list都是加了<s>和<\s>后对应的坐标
        """
        # pdb.set_trace()
        windows_sentence_length = [] #[[136, 40, 138, 80, 174, 176, 216, 310, 254, 388, 136, 7, 20, 44, 141, 25, 62, 159, 67, 201, 136, 16, 150]]
        for batch in windows_length:
            thread_data = []
            for window in batch:
                # window_data = [item for sublist in window for item in sublist]
                thread_data.extend(window)
            windows_sentence_length.append(thread_data)

       

        dialogue_length = list(map(len, input_ids))#[9,10]

        max_lens = max(map(lambda line: max(map(len, line)), input_ids))
        padding = lambda input_batch: [w + [0] * (max_lens - len(w)) for line in input_batch for w in line]
        input_ids, input_masks, input_segments = map(padding, [input_ids, input_masks, input_segments])
        
        max_lens = max(map(len, token2sents))
        padding = lambda input_batch: [w + [0] * (max_lens - len(w)) for w in input_batch]
        token2sents, utterance_index, token_index, token2speaker = map(padding, [token2sents, utterance_index, token_index, token2speaker])#用于RoPE算法

        padding_list = lambda input_batch : [list(map(list, w)) + [[0, 0, 0]] * (max(map(len, input_batch)) - len(w)) for w in input_batch]
        entity_lists, rel_lists, polarity_lists = map(padding_list, [entity_list, rel_list, polarity_list])
        # pdb.set_trace()
        max_tri_num = max(map(len, triplets))
        triplet_masks = [[1] * len(w) + [0] * (max_tri_num - len(w)) for w in triplets]
        triplets = [list(map(list, w)) + [[0] * 7] * (max_tri_num - len(w)) for w in triplets]
        
        
        sentence_masks = np.zeros([len(token2sents), max_lens, max_lens], dtype=int)
        for i in range(len(sentence_length)):
            masks = [np.triu(np.ones([lens, lens], dtype=int)) for lens in sentence_length[i]]
            masks = block_diag(*masks)
            sentence_masks[i, :len(masks), :len(masks)] = masks
        sentence_masks = sentence_masks.tolist()

        flatten_length = list(map(sum, sentence_length))
        cur_masks = (np.expand_dims(np.arange(max(flatten_length)), 0) < np.expand_dims(flatten_length, 1)).astype(np.int64)
        full_masks = (np.expand_dims(cur_masks, 2) * np.expand_dims(cur_masks, 1)).tolist()
        
        max_lens2 = max(map(len, token2sents_raw))
        sentence_masks_raw = np.zeros([len(token2sents_raw), max_lens2, max_lens2], dtype=int)
        for i in range(len(sentence_length_raw)):
            masks = [np.triu(np.ones([lens, lens], dtype=int)) for lens in sentence_length_raw[i]]
            masks = block_diag(*masks)
            sentence_masks_raw[i, :len(masks), :len(masks)] = masks
        sentence_masks_raw = sentence_masks_raw.tolist()

        flatten_length = list(map(sum, sentence_length_raw))
        cur_masks = (np.expand_dims(np.arange(max(flatten_length)), 0) < np.expand_dims(flatten_length, 1)).astype(np.int64)
        full_masks_raw = (np.expand_dims(cur_masks, 2) * np.expand_dims(cur_masks, 1)).tolist()
        # pdb.set_trace()

        entity_matrix = self.kernel.list2rel_matrix4batch(entity_lists, max_lens)
        rel_matrix = self.kernel.list2rel_matrix4batch(rel_lists, max_lens)
        polarity_matrix = self.kernel.list2rel_matrix4batch(polarity_lists, max_lens)

        new_reply_masks = np.zeros([len(reply_mask), max_lens, max_lens])
        for i in range(len(new_reply_masks)):
            lens = len(reply_mask[i])
            new_reply_masks[i, :lens, :lens] = reply_mask[i]

        new_speaker_masks = np.zeros([len(speaker_mask), max_lens, max_lens])
        for i in range(len(new_speaker_masks)):
            lens = len(speaker_mask[i])
            new_speaker_masks[i, :lens, :lens] = speaker_mask[i]

        new_thread_masks = np.zeros([len(thread_mask), max_lens, max_lens])
        for i in range(len(new_thread_masks)):
            lens = len(thread_mask[i])
            new_thread_masks[i, :lens, :lens] = thread_mask[i]
#================================windows窗口=================================#
        windows_dialogue_length = list(map(len, windows_input_ids))#[23,] 窗口的个数
    
        windows_max_lens = max(map(lambda line: max(map(len, line)), windows_input_ids))
        windows_padding = lambda input_batch: [w + [0] * (windows_max_lens - len(w)) for line in input_batch for w in line]

        # 23个窗口,每个窗口被padding成最大388
        windows_input_ids, windows_input_masks, windows_input_segments = map(windows_padding, [windows_input_ids, windows_input_masks, windows_input_segments])
        windows_token2sents = [item for sublist in windows_token2sents for item in sublist ] #4维降为3层list维:[batch,windows,win_seq_len] [ [  [] [],[],[] ]  ]
        # windows_max_lens = max(map(len, windows_token2sents))
        windows_max_lens = max(len(item) for sublist in windows_token2sents for item in sublist) #388
        # padding = lambda input_batch: [w + [0] * (max_lens - len(w)) for w in input_batch]
        # token2sents, utterance_index, token_index, token2speaker = map(padding, [token2sents, utterance_index, token_index, token2speaker])#用于RoPE算法
        windows_max_nums = max(map(len,windows_sentence_length))
        # padding_list = lambda input_batch : [list(map(list, w)) + [[0, 0, 0]] * (max(map(len, input_batch)) - len(w)) for w in input_batch]
        # entity_lists, rel_lists, polarity_lists = map(padding_list, [entity_list, rel_list, polarity_list])
        # pdb.set_trace()
        """
        windows_sentence_masks: 比sentence_masks多一维，mask是每个windows的mask    
                                [batch, max_windows_num, windows_max_lens, windows_max_lens] 如[1,26,165,165]  
        windows_full_masks: [batch,max_windows_num, windows_max_lens,max_lens]:   填充的窗口所在的<行>为0  如windows_full_masks[0][0][13]所在的行是全0的。
        
        sentence_masks = np.zeros([len(token2sents), max_lens, max_lens], dtype=int)
        for i in range(len(sentence_length)):
            masks = [np.triu(np.ones([lens, lens], dtype=int)) for lens in sentence_length[i]]
            masks = block_diag(*masks)
            sentence_masks[i, :len(masks), :len(masks)] = masks
        
        """
        windows_sentence_masks = []
        windows_mask= []
        for w_lengths in windows_sentence_length: #batch级 [[136, 40, 138, 80, 174, 176, 216, 310, 254, 388, 136, 7, 20, 44, 141, 25, 62, 159, 67, 201, 136, 16, 150]]
            #每个windows的内容
            w_sentence_mask = np.zeros([windows_max_nums, windows_max_lens, windows_max_lens], dtype=int)
            # pdb.set_trace()
            for j, length in enumerate(w_lengths):
                #FIXME 不知道下面这个w_mask改的对不对，不再是对全部对话进行mask，而是以windows的长度为单位
                w_mask = [np.triu(np.ones([length , length ], dtype=int))]
                w_mask = block_diag(*w_mask)
                # pdb.set_trace()
                w_sentence_mask[j, :len(w_mask), :len(w_mask)] = w_mask
            w_sentence_mask = w_sentence_mask.tolist()
            windows_sentence_masks.append(w_sentence_mask) #([1, 23, 388, 388]) [batch, 最大窗口数，最长窗口]
        # windows_sentence_masks = windows_sentence_masks.tolist()
        # pdb.set_trace()
        
        windows_full_masks = [] #以窗口为单位
        for w_lengths in windows_sentence_length: #[[136, 40, 138, 80, 174, 176, 216, 310, 254, 388, 136, 7, 20, 44, 141, 25, 62, 159, 67, 201, 136, 16, 150]]
            cur_masks = np.zeros((windows_max_nums, windows_max_lens, windows_max_lens), dtype=np.int64)
            for i, length in enumerate(w_lengths):
                cur_mask = np.zeros((length, length), dtype=np.int64)
                cur_mask.fill(1)
                cur_masks[i, :length, :length] = cur_mask
            windows_full_masks.append(cur_masks.tolist())
       
#=============================================================================#
#==================================part1==============================================================#
        """
        thread_dialogue_length = list(map(len,thread_input_ids)) #[4,4]由句子的个数变成线程的个数

        thread_max_lens = max(map(lambda line: max(map(len, line)), thread_input_ids))
        thread_padding = lambda input_batch: [w + [0] * (thread_max_lens - len(w)) for line in input_batch for w in line]
        
        # ✔️下面的thread_input_ids 要处理成thread和线程的格式作为输入!!!!
        
        thread_input_ids, thread_input_masks, thread_input_segments = map(thread_padding, [thread_input_ids, thread_input_masks, thread_input_segments])
        thread_max_lens = max(map(len, thread_token2sents))
        thread_padding = lambda input_batch: [w + [0] * (thread_max_lens - len(w)) for w in input_batch]
        # token2sents, utterance_index, token_index, token2speaker = map(padding, [token2sents, utterance_index, token_index, token2speaker])

        padding_list = lambda input_batch : [list(map(list, w)) + [[0, 0, 0]] * (max(map(len, input_batch)) - len(w)) for w in input_batch]
        thread_entity_lists, thread_rel_lists, thread_polarity_lists = map(padding_list, [thread_entity_list, thread_rel_list, thread_polarity_list])

        max_tri_num = max(map(len, thread_triplets))
        thread_triplet_masks = [[1] * len(w) + [0] * (max_tri_num - len(w)) for w in thread_triplets]
        thread_triplets = [list(map(list, w)) + [[0] * 7] * (max_tri_num - len(w)) for w in thread_triplets]

        thread_flatten_length = list(map(sum, thread_sentence_length))
        cur_masks = (np.expand_dims(np.arange(max(thread_flatten_length)), 0) < np.expand_dims(thread_flatten_length, 1)).astype(np.int64)
        thread_full_masks = (np.expand_dims(cur_masks, 2) * np.expand_dims(cur_masks, 1)).tolist()
        # entity_matrix = self.kernel.list2rel_matrix4batch(entity_lists, max_lens)
        thread_entity_matrix = self.kernel.list2rel_matrix4batch(thread_entity_lists,thread_max_lens)
        thread_rel_matrix = self.kernel.list2rel_matrix4batch(thread_rel_lists, thread_max_lens)
        thread_polarity_matrix = self.kernel.list2rel_matrix4batch(thread_polarity_lists,thread_max_lens)
       """
#----------------------------------------------------------------------------------------------------#
        
       
        res = {
            "doc_id": doc_id,
			"input_ids": input_ids, "input_masks": input_masks, "input_segments": input_segments,
            'ent_matrix': entity_matrix, 'rel_matrix': rel_matrix, 'pol_matrix': polarity_matrix,
            'sentence_masks': sentence_masks, 'full_masks': full_masks,   
            'sentence_masks_raw': sentence_masks_raw, 'full_masks_raw': full_masks_raw,             
            'triplets': triplets, 'triplet_masks': triplet_masks, 'pairs': pairs,
            'token2sents': token2sents, 'token2sents_raw':token2sents_raw,'dialogue_length': dialogue_length,
            'utterance_index': utterance_index, 'token_index': token_index,
            'thread_lengths': thread_length, 'token2speakers': token2speaker,
            'reply_masks': new_reply_masks, 'speaker_masks': new_speaker_masks, 'thread_masks': new_thread_masks,
            'pieces2words': pieces2words, 'new2old': new2old,
            'windows_input_ids':windows_input_ids, 'windows_input_masks':windows_input_masks, 'windows_input_segments':windows_input_segments,
            'windows_utterance_spans':flatten_all_windows_tokens_spans,'windows_utterance_spans_raw':flatten_all_windows_tokens_spans_raw,'windows_sentence_masks':windows_sentence_masks,'windows_full_masks':windows_full_masks,'windows_lengths':windows_sentence_length
        }
        # pdb.set_trace()
        """
        res = {
            "doc_id": doc_id,
			"input_ids": input_ids, "input_masks": input_masks, "input_segments": input_segments,
            'ent_matrix': entity_matrix,   'rel_matrix': rel_matrix, 'pol_matrix': polarity_matrix,
            'sentence_masks': sentence_masks, 'full_masks': full_masks,             
            'triplets': triplets, 'triplet_masks': triplet_masks, 'pairs': pairs,
            'token2sents': token2sents, 'dialogue_length': dialogue_length,
            'utterance_index': utterance_index, 'token_index': token_index,
            'thread_lengths': thread_length, 'token2speakers': token2speaker,
            'reply_masks': new_reply_masks, 'speaker_masks': new_speaker_masks, 'thread_masks': new_thread_masks,
            'pieces2words': pieces2words, 'new2old': new2old,
            
        }
        """
        nocuda = ['thread_lengths', 'pairs', 'doc_id', 'pieces2words', 'new2old','windows_utterance_spans','windows_utterance_spans_raw']
        res = {k: v if k in nocuda else torch.tensor(v).to(self.config.device) for k, v in res.items()}
        # pdb.set_trace()
        return res
       
    def getdata(self):
        
        load_data = lambda mode: DataLoader(MyDataset(self.data[mode]), num_workers=0, worker_init_fn=self.worker_init, 
                                                shuffle=(mode == 'train'),  batch_size=self.config.batch_size, collate_fn=self.collate_fn)
        
        train_loader, valid_loader, test_loader = map(load_data, 'train valid test'.split())

        line = 'polarity_dict target_dict aspect_dict opinion_dict entity_dict relation_dict'.split()
        for w, z in zip(line, self.data['label_dict']):
            self.config[w] = z

        res = (train_loader, valid_loader, test_loader, self.config)

        return res
    
class RelationMetric:
    def __init__(self, config):
        self.clear()
        self.kernel = WordPair()
        self.predict_result = defaultdict(list) #用于存储预测结果
        self.config = config
    
    def trans2position(self, triplet,  pieces2words):
        res = []
        """
        recover the position of entities in the original sentence

        new2old: transfer position from index with CLS and SEP to index without CLS and SEP
        pieces2words: transfer position from index of wordpiece to index of original words 

        Example:
        list0 (original sentence):"London is the capital of England"
        list1 (tokenized sentence): "Lon ##don is the capital of England"
        list2 (packed sentence): "[CLS] Lon #don is the capital of England [SEP]"
        predicted entity: (1, 2), denotes "Lon #don" in list2

        new2old: list2->list1
          = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, ...}
        pieces2words: list1->list0
          = {'0': 0, '1': 0, '2': 1, '3': 2, '4': 3, ...}

        input  -> entity in list2: "Lon #don" (1, 2)
        middle -> entity in list1: "Lon #don" (0, 1)
        output -> entity in list0: "London"   (0, 0)
        """

        # head = lambda x : pieces2words[new2old[x]]
        # tail = lambda x : pieces2words[new2old[x]]
        #这里不需要加new2old映射了，因为构造的pred matrix是不包括cls的
        head = lambda x : pieces2words[x] 
        tail = lambda x : pieces2words[x]
        triplet = list(triplet)
        for s0, e0, s1, e1, s2, e2, pol in triplet:
            ns0, ns1, ns2 = head(s0), head(s1), head(s2)
            ne0, ne1, ne2 = tail(e0), tail(e1), tail(e2)
            res.append([ns0, ne0, ns1, ne1, ns2, ne2, pol])
        return res
    
    def trans2pair(self, pred_pairs, pieces2words):
        new_pairs = {}
        # new_pos = lambda x : pieces2words[new2old[x]]
        new_pos = lambda x : pieces2words[x]
        for k, line in pred_pairs.items():
            new_line = []
            for s0, e0, s1, e1 in line:
                s0, e0, s1, e1 = new_pos(s0), new_pos(e0), new_pos(s1), new_pos(e1)
                new_line.append([s0, e0, s1, e1])
            new_pairs[k] = new_line
        return new_pairs

    def filter_entity(self, ent_list, pieces2words):
        res = []

        # If the entity is a sub-string of another entity, remove it
        # ent_list = sorted(ent_list, key=lambda x: (x[0], -x[1]))
        # ent_list = [w for i, w in enumerate(ent_list) if i == 0 or w[0] != ent_list[i-1][0]]
        for s, e, pol in ent_list:
            # pdb.set_trace()
            # ns, ne = pieces2words[new2old[s]], pieces2words[new2old[e]]
            ns, ne = pieces2words[s], pieces2words[e]
            res.append([ns, ne, pol])
        return res

    def add_instance(self, data, pred_ent_matrix, pred_rel_matrix, pred_pol_matrix):
        """
        input_matrix: [B, Seq, Seq]
        pred_matrix: [B, Seq, Seq, 6]
        input_masks: [B, Seq]
        """
        pred_ent_matrix = pred_ent_matrix.argmax(-1) * data['sentence_masks_raw']
        pred_rel_matrix = pred_rel_matrix.argmax(-1) * data['full_masks_raw']
        pred_pol_matrix = pred_pol_matrix.argmax(-1) * data['full_masks_raw'] 
        # token2sents = data['token2sents'].tolist()
        token2sents = data['token2sents_raw'].tolist() #不包括cls的token到句子的映射
        # new2old = data['new2old']
        pieces2words = data['pieces2words']
        doc_id = data['doc_id']

        pred_rel_matrix = np.array(pred_rel_matrix.tolist()) #(1,304,304) [batch,max_len,max_len]
        pred_ent_matrix = np.array(pred_ent_matrix.tolist())
        pred_pol_matrix = np.array(pred_pol_matrix.tolist())
        # pdb.set_trace()
        for i in range(len(pred_ent_matrix)): #batch级
            ent_matrix, rel_matrix, pol_matrix = pred_ent_matrix[i], pred_rel_matrix[i], pred_pol_matrix[i]
            pred_triplet, pred_pairs = self.kernel.get_triplets(ent_matrix, rel_matrix, pol_matrix, token2sents[i])
            pred_ents = self.kernel.rel_matrix2list(ent_matrix)

            """
            将上面解码得到的结果 重定向到原始sentence的index（既去除掉<s>和</s>的）
            pred_pairs
                前：{'ta': [(96, 98, 112, 113), (165, 167, 173, 173), (287, 287, 281, 282), (287, 287, 295, 298),...}
                后：{'ta': [[84, 84, 95, 95], [140, 140, 145, 145], [236, 236, 231, 231], [236, 236, 243, 245],...}
            pred_triplets:
                前：[(165, 167, 173, 173, 169, 172, 1), (287, 287, 281, 282, 290, 293, 2), (287, 287, 295, 298, 290, 293, 2), (287, 287, 297, 298, 290, 293, 2),...]
                后：
            
            对于我自己构建的matrix，因为本身就是不包含cls的，所以在下面的解码中不需要new2old这一步映射，直接拼接pieces2word就可以。
            
            """
            """上面的还是带着cls的"""
            pred_ents = self.filter_entity(pred_ents, pieces2words[i]) #[[3, 5, 3], [11, 11, 1], [17, 17, 3], [20, 20, 1], [24, 24, 2], ...]
            pred_pairs = self.trans2pair(pred_pairs,pieces2words[i])
            # pdb.set_trace()
            pred_triplet = self.trans2position(pred_triplet, pieces2words[i])
            self.predict_result[doc_id[i]].append(pred_ents)
            self.predict_result[doc_id[i]].append(pred_pairs)
            self.predict_result[doc_id[i]].append(pred_triplet)
            # pdb.set_trace()
    def clear(self):
        self.predict_result = defaultdict(list)

    def save2file(self, gold_file, pred_file):
        # pol_dict = {"O": 0, "pos": 1, "neg": 2, "other": 3}
        pol_dict = self.config.polarity_dict
        reverse_pol_dict = {v: k for k, v in pol_dict.items()}
        reverse_ent_dict = {v: k for k, v in self.config.entity_dict.items()}
                
        gold_file = open(gold_file, 'r', encoding='utf-8')

        data = json.load(gold_file)

        res = []
        for line in data:
            doc_id, sentence = line['doc_id'], line['sentences']
            if doc_id not in self.predict_result:
                continue
            doc = ' '.join(sentence).split()
            new_triples = []

            prediction = self.predict_result[doc_id]
            entities = defaultdict(list)
            for head, tail, tp in prediction[0]:
                tp = reverse_ent_dict[tp]
                head, tail = head, tail + 1
                tp_dict = {'ENT-T': 'targets', 'ENT-A': 'aspects', 'ENT-O': 'opinions'}
                entities[tp_dict[tp]].append([head, tail])

            pairs = defaultdict(list)
            for key in ['ta', 'to', 'ao']:
                for s0, e0, s1, e1 in prediction[1][key]:
                    e0, e1 = e0 + 1, e1 + 1
                    pairs[key].append([s0, e0, s1, e1])

            new_triples = []
            for s0, e0, s1, e1, s2, e2, pol in prediction[2]:
                pol = reverse_pol_dict[pol]
                e0, e1, e2 = e0 + 1, e1 + 1, e2 + 1
                new_triples.append([s0, e0, s1, e1, s2, e2, pol, ' '.join(doc[s0:e0]), ' '.join(doc[s1:e1]), ' '.join(doc[s2:e2])])

            res.append({'doc_id': doc_id, 'triplets': new_triples, \
                        'targets': entities['targets'], 'aspects': entities['aspects'], 'opinions': entities['opinions'],\
                        'ta': pairs['ta'], 'to': pairs['to'], 'ao': pairs['ao']})
        logger.info('Save prediction results to {}'.format(pred_file))
        json.dump(res, open(pred_file, 'w', encoding='utf-8'), ensure_ascii=False)
    
    def compute(self, name='valid'):
        # action: pred, make prediction, save to file 
        # action: eval, make prediction, save to file and evaluate 

        args = AttrDict({
            'pred_file': os.path.join(self.config.target_dir, 'pred_{}_{}.json'.format(self.config.lang, name)),
            'gold_file': os.path.join(self.config.json_path, '{}.json'.format(name))
            # 'gold_file': os.path.join(self.config.json_path, '{}_gold.json'.format(name))
        })
        self.save2file(args.gold_file, args.pred_file)

        micro, iden, res = Run_eval(args).forward()
        self.clear()
        return micro[2], res
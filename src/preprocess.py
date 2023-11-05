
from src.utils import WordPair
import os
import re
import json

import numpy as np
import copy
from collections import defaultdict
from itertools import accumulate
from transformers import AutoTokenizer
from typing import List, Dict
from loguru import logger
from tqdm import tqdm
import pdb

class Preprocessor:
    def __init__(self, config):
        self.config = config 
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
        self.wordpair = WordPair()
        self.entity_dict = self.wordpair.entity_dic
    
    def get_dict(self):
        self.polarity_dict = self.config.polarity_dict

        self.aspect_dict = {}
        for w in self.config.bio_mode:
            self.aspect_dict['{}{}'.format(w, '' if w == 'O' else '-' + self.config.asp_type)] = len(self.aspect_dict)

        self.target_dict = {}
        for w in self.config.bio_mode:
            self.target_dict['{}{}'.format(w, '' if w == 'O' else '-' + self.config.tgt_type)] = len(self.target_dict)

        self.opinion_dict = {'O': 0}
        for p in self.polarity_dict:
            if p == 'O': continue
            for w in self.config.bio_mode[1:]:
                self.opinion_dict['{}-{}_{}'.format(w, self.config.opi_type, p)] = len(self.opinion_dict)
        
        self.relation_dict = {'O': 0, 'yes': 1}
        return self.polarity_dict, self.target_dict, self.aspect_dict, self.opinion_dict, self.entity_dict, self.relation_dict
    
    def get_neighbor(self, utterance_spans, replies, max_length, speaker_ids, thread_nums):
        # utterance_mask = np.zeros([max_length, max_length], dtype=int)
        reply_mask = np.eye(max_length, dtype=int)
        for i, w in enumerate(replies):
            s1, e1 = utterance_spans[i]
            s0, e0 = utterance_spans[w + (1 if w == -1 else 0)] #第0句话即root
            reply_mask[s0 : e0 + 1, s1 : e1 + 1] = 1
            reply_mask[s1 : e1 + 1, s0 : e0 + 1] = 1
            reply_mask[s0 : e0 + 1, s0 : e0 + 1] = 1
            reply_mask[s1 : e1 + 1, s1 : e1 + 1] = 1
        
        speaker_mask = np.zeros([max_length, max_length], dtype=int)
        for i, idx in enumerate(speaker_ids):
            # utterance_ids = [j for j, w in enumerate(speaker_ids) if w == idx]
            s0, e0 = utterance_spans[i]
            for j, idx1 in enumerate(speaker_ids):
                if idx != idx1: continue
                s1, e1 = utterance_spans[j] 
                speaker_mask[s0 : e0 + 1, s1 : e1 + 1] = 1
                speaker_mask[s1 : e1 + 1, s0 : e0 + 1] = 1
                speaker_mask[s0 : e0 + 1, s0 : e0 + 1] = 1
                speaker_mask[s1 : e1 + 1, s1 : e1 + 1] = 1
        
        thread_mask = np.eye(max_length, dtype=int)
        thread_ends = accumulate(thread_nums)
        thread_spans = [(w - z, w) for w, z in zip(thread_ends, thread_nums)]#[0,1],[1,4],[4,7],[7,10]
        for i, (s, e) in enumerate(thread_spans):
            if i == 0: continue
            head_start, head_end = utterance_spans[0]
            thread_mask[head_start : head_end + 1, head_start : head_end + 1] = 1
            for j in range(s, e):
                s0, e0 = utterance_spans[j]
                thread_mask[s0:e0 + 1, head_start:head_end+1] = 1
                thread_mask[head_start:head_end+1, s0:e0 + 1] = 1
                for k in range(s, e):
                    s1, e1 = utterance_spans[k]
                    thread_mask[s0 + 1 : e0, s1 + 1: e1] = 1
                    thread_mask[s1 + 1 : e1, s1 + 1: e1] = 1
                    thread_mask[s0 + 1 : e0, s0 + 1: e0] = 1
                    thread_mask[s1 + 1 : e1, s0 + 1: e0] = 1
        return reply_mask.tolist(), speaker_mask.tolist(), thread_mask.tolist()
    
    def find_utterance_index(self, replies, sentence_lengths):
        utterance_collections = [i for i, w in enumerate(replies) if w == 0]
        zero_index = utterance_collections[1]
        for i in range(len(replies)):
            if i < zero_index: continue
            if replies[i] == 0:
                zero_index = i
            replies[i] = (i - zero_index)

        sentence_index = [w + 1 for w in replies]

        utterance_index = [[w] * z for w, z in zip(sentence_index, sentence_lengths)]
        utterance_index = [w for line in utterance_index for w in line]

        token_index = [list(range(sentence_lengths[0]))]
        lens = len(token_index[0])
        for i, w in enumerate(sentence_lengths):
            if i == 0: continue
            if sentence_index[i] == 1:
                distance = lens
            token_index += [list(range(distance, distance + w))]
            distance += w
        token_index = [w for line in token_index for w in line]

        utterance_collections = np.split(sentence_index, utterance_collections)

        thread_nums = list(map(len, utterance_collections))
        thread_ranges = [0] + list(accumulate(thread_nums))
        thread_lengths = [sum(sentence_lengths[thread_ranges[i]:thread_ranges[i+1]]) for i in range(len(thread_ranges)-1)]

        return utterance_index, token_index, thread_lengths, thread_nums
    
    def get_pair(self, full_triplets):
        pairs = {'ta': set(), 'ao': set(), 'to': set()}
        for i in range(len(full_triplets)):
            st, et, sa, ea, so, eo, p = full_triplets[i][:7]
            if st != -1 and sa != -1:
                pairs['ta'].add((st, et, sa, ea))

            if st != -1 and so != -1:
                pairs['to'].add((st, et, so, eo))

            if sa != -1 and eo != -1:
                pairs['ao'].add((sa, ea, so, eo))

        return pairs

    def transfer_polarity(self, pol):
        res = {'pos': 'pos', 'neg': 'neg'}
        return res.get(pol, 'other')
    
    def read_data(self, mode):
        path = os.path.join(self.config.json_path, '{}.json'.format(mode))

        if not os.path.exists(path):
            raise FileNotFoundError('File {} not found! Please check your input and data path.'.format(path))

        content = json.load(open(path, 'r', encoding='utf-8'))
        res = []
        for line in tqdm(content, desc='Processing dialogues for {}'.format(mode)):
            new_dialog = self.parse_dialogue(line, mode)
            res.append(new_dialog)
        return res
    
    def check_text(self, tokenized_text, source_text):
        if self.config.bert_path in ['roberta-large', 'roberta-base']: #you only need to change 'roberta-large' to your own path.
            t0 = tokenized_text.lower()
            roberta_chars = 'â ī ¥ Ġ ð ł ĺ ħ Ł ŀ į Ŀ Į ĵ © ĵ ĳ ¶ ã'.split()
            unused = [self.config.unk, '##']
            if self.config.bert_path in ['roberta-large', 'roberta-base']:
                unused += roberta_chars
            for u in unused:
                t0 = t0.replace(u.lower(), '')
            t1 = source_text.replace(' ', '').lower()
            for k in self.config.unkown_tokens:
                t1 = t1.replace(k, '')
            if self.config.bert_path in ['roberta-large', 'roberta-base']:
                t1 = t1.replace('×', '').replace('≥', '')
            if t0 != t1:
                logger.info(t1 + '||' + t1)
                logger.info(tokenized_text + '||' + source_text)
                t2 = t0
                for u in unused:
                    t2 = t2.replace(u, '')
                raise AssertionError("--{}-- != --{}--".format(t0, t1))
            return t0 == t1

        t0 = tokenized_text.replace('##', '').replace(self.config.unk, '').replace('[UNK]','').lower()
        t1 = source_text.replace(' ', '').lower()
        # pdb.set_trace()
        for k in self.config.unkown_tokens:
            t1 = t1.replace(k, '')
        if t0 != t1:
            logger.info(t0 + '||' + t1)
            logger.info(tokenized_text + '||' + source_text)
        return t0 == t1
    
    def parse_dialogue(self, dialogue, mode):
        sentences = dialogue['sentences']
        new_sentences, pieces2words = self.align_index_with_list(sentences)

        word2pieces = defaultdict(list) 
        for p, w in enumerate(pieces2words):
            word2pieces[w].append(p)

        dialogue['pieces2words'] = pieces2words
        dialogue['sentences'] = new_sentences

        # get target, aspect and opinion respectively, and align to the new index

        if mode != 'train':
            return dialogue
        targets, aspects, opinions = [dialogue[w] for w in ['targets', 'aspects', 'opinions']]
        targets = [(word2pieces[x][0], word2pieces[y-1][-1] + 1, z) for x, y, z in targets]
        aspects = [(word2pieces[x][0], word2pieces[y-1][-1] + 1, z) for x, y, z in aspects]
        opinions = [(word2pieces[x][0], word2pieces[y-1][-1] + 1, z, self.transfer_polarity(w)) for x, y, z, w in opinions]
        
        # Put the elements into the dialogue object after converting the elements to the new index
        dialogue['targets'], dialogue['aspects'], dialogue['opinions'] = targets, aspects, opinions

        # Flatten the two-dimensional list and put the entire dialogue in a list
        news = [w for line in new_sentences for w in line]

        # Confirm the index again
        for ts, te, t_t in targets:
            assert self.check_text(''.join(news[ts:te]), t_t)
        for ts, te, t_t in aspects:
            assert self.check_text(''.join(news[ts:te]), t_t)
        for ts, te, t_t,_ in opinions:
            assert self.check_text(''.join(news[ts:te]), t_t)

        triplets = []

        # polarity transfer and index transfer

        for t_s, t_e, a_s, a_e, o_s, o_e, polarity, t_t, a_t, o_t in dialogue['triplets']:
            polarity = self.transfer_polarity(polarity)
            nts, nas, nos = [word2pieces[w][0] if w != -1 else -1 for w in [t_s, a_s, o_s]]
            nte, nae, noe = [word2pieces[w - 1][-1] + 1 if w != -1 else -1 for w in [t_e, a_e, o_e]]
            self.check_text(''.join(news[nts:nte]), t_t)
            self.check_text(''.join(news[nas:nae]), a_t) or nas == -1
            if not self.check_text(''.join(news[nos:noe]), o_t) and nos != -1:
                logger.info(''.join(news[nos:noe]) + '||' + o_t)
            self.check_text(''.join(news[nos:noe]), o_t) or nos == -1

            triplets.append((nts, nte, nas, nae, nos, noe, polarity, t_t, a_t, o_t)) #前部分数字，后部分字符
            # pdb.set_trace()
        dialogue['triplets'] = triplets
        return dialogue #返回切片后的

    
    def align_index_with_list(self, sentences):
        """_summary_
        align the index of the original elements according to the tokenization results
        Args:
            sentences (_type_): List<str>
            e.g., xiao mi 12x is my favorite
        """
        pieces2word = []
        word_num = 0
        all_pieces = []
        for sentence in sentences:
            sentence = sentence.split()
            tokens = [self.tokenizer.tokenize(w) for w in sentence]
            cur_line = []
            for token in tokens:
                for piece in token:
                    pieces2word.append(word_num)
                word_num += 1
                cur_line += token
            all_pieces.append(cur_line)
        
        return all_pieces, pieces2word
    
    def align_index(self, sentences):
        res, char2token = [], {}
        source_lens, token_lens = 0, 0
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            if self.config.bert_path in ['roberta-large', 'roberta-base']:
                c2t, tokens = self.alignment_roberta(sentence, tokens)
            else:
                c2t, tokens = self.alignment(sentence, tokens)
            res.append(tokens)
            for k, v in c2t.items():
                char2token[k + source_lens] = v + token_lens
            source_lens, token_lens = source_lens + len(sentence) + 1, token_lens + len(tokens)

        return res, char2token
    
    def alignment(self, source_sequence, tokenized_sequence: List[str], align_type: str = 'one2many') -> Dict:
        """[summary]
        # this is a function that to align sequcences  that before tokenized and after.
        Parameters
        ----------
        source_sequence : [type]
            this is the original sequence, whose type either can be str or list
        tokenized_sequence : List[str]
            this is the tokenized sequcen, which is a list of tokens.
        index_type : str, optional, default: str
            this indicate whether source_sequence is str or list, by default 'str'
        align_type : str, optional, default: one2many
            there may be several kinds of tokenizer style, 
            one2many: one word in source sequence can be split into multiple tokens 
            many2one: many word in source sequence will be merged into one token
            many2many: both contains one2many and many2one in a sequence, this is the most complicated situation.
        
        useage:
        source_sequence = "Here, we investigate the structure and dissociation process of interfacial water"
        tokenized_sequence = ['here', ',', 'we', 'investigate', 'the', 'structure', 'and', 'di', '##sso', '##ciation', 'process', 'of', 'inter', '##fa', '##cial', 'water']
        char2token = alignment(source_sequence, tokenized_sequence)
        print(char2token)
        for c, t in char2token.items():
            print(source_sequence[c], tokenized_sequence[t])
        """
        char2token = {}
        if isinstance(source_sequence, str) and align_type == 'one2many':
            source_sequence = source_sequence.lower()
            i, j = 0, 0
            while i < len(source_sequence) and j < len(tokenized_sequence):
                cur_token, length = tokenized_sequence[j], len(tokenized_sequence[j])
                if source_sequence[i] == ' ':
                    i += 1
                elif source_sequence[i: i + length] == cur_token:
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
                elif tokenized_sequence[j] == self.config.unk:
                    lens = 1
                    if j + 1 == len(tokenized_sequence):
                        lens = len(source_sequence) - i
                    else:
                        while i + lens < len(source_sequence):
                            if source_sequence[i + lens] == tokenized_sequence[j + 1].strip('#')[0] or tokenized_sequence[j+1] == self.config.unk:
                                break
                            lens += 1
                    new_token = self.repack_unknow(source_sequence[i:i+lens])
                    tokenized_sequence = tokenized_sequence[:j] + new_token + tokenized_sequence[j+1:]
                    if tokenized_sequence[j] == self.config.unk:
                        char2token[i] = j
                        i += 1
                        j += 1
                else:
                    assert tokenized_sequence[j].startswith('#')
                    length = len(tokenized_sequence[j].lstrip('#'))
                    assert source_sequence[i: i + length] == tokenized_sequence[j].lstrip('#')
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
        return char2token, tokenized_sequence
    
    def alignment_roberta(self, source_sequence, tokenized_sequence: List[str]) -> Dict:
        # For English dataset
        char2token = {}
        if isinstance(source_sequence, str):
            source_sequence = source_sequence.lower()
            i, j = 0, 0
            while i < len(source_sequence) and j < len(tokenized_sequence):
                cur_token, length = tokenized_sequence[j], len(tokenized_sequence[j].strip('Ġ'))
                if source_sequence[i] == ' ':
                    i += 1
                elif source_sequence[i: i + length].lower() == cur_token.strip('Ġ').lower():
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
                elif tokenized_sequence[j] == self.config.unk:
                    lens = 1
                    if j + 1 == len(tokenized_sequence):
                        lens = len(source_sequence) - i
                    else:
                        while i + lens < len(source_sequence):
                            if source_sequence[i + lens] == tokenized_sequence[j + 1].strip('#')[0] or tokenized_sequence[j+1] == self.config.unk:
                                if tokenized_sequence[j+1].strip('#')[0] == 'i' and j + 1 < len(tokenized_sequence) and len(tokenized_sequence[j+1].strip()) > 1:
                                    if i + lens + 1 < len(source_sequence) and source_sequence[i+lens+1] == tokenized_sequence[j+1].strip('#')[1]: 
                                        break
                                else:
                                    break
                            lens += 1
                    new_token = self.repack_unknow(source_sequence[i:i+lens])
                    tokenized_sequence = tokenized_sequence[:j] + new_token + tokenized_sequence[j+1:]
                    if tokenized_sequence[j] == self.config.unk:
                        char2token[i] = j
                        i += 1
                        j += 1
                else:
                    assert tokenized_sequence[j].startswith('#')
                    length = len(tokenized_sequence[j].lstrip('#'))
                    assert source_sequence[i: i + length] == tokenized_sequence[j].lstrip('#')
                    for k in range(length):
                        char2token[i + k] = j
                    i, j = i + length, j + 1
        return char2token, tokenized_sequence
    
    def repack_unknow(self, source_sequence):
        '''
        # sentence='🍎12💩', Bert can't recognize two contiguous emojis, so it recognizes the whole as '[UNK]'
        # We need to manually split it, recognize the words that are not in the bert vocabulary as UNK, 
        and let BERT re-segment the parts that can be recognized, such as numbers
        # The above example processing result is: ['[UNK]', '12', '[UNK]']
        '''
        lst = list(re.finditer('|'.join(self.config.unkown_tokens), source_sequence))
        start, i = 0, 0
        new_tokens = []
        while i < len(lst):
            s, e = lst[i].span()
            if start < s:
                token = self.tokenizer.tokenize(source_sequence[start:s]) 
                new_tokens += token
                start = s
            else:
                new_tokens.append(self.config.unk)
                start = e
            i += 1
        if start < len(source_sequence):
            token = self.tokenizer.tokenize(source_sequence[start:]) 
            new_tokens += token
        return new_tokens
    
    def generate_sliding_windows(self,sublist, window_size):
        windows = []
        for i in range(len(sublist) - window_size + 1):
            window = sublist[i:i+window_size]
            windows.append(window)
        return windows

    def generate_all_windows(self,index_list):
        all_windows = []
        for sublist in index_list:
            sublist_windows = []
            sublist_size = len(sublist)
            for window_size in range(1, sublist_size + 1):
                sublist_windows.extend(self.generate_sliding_windows(sublist, window_size))
            all_windows.append(sublist_windows)
        return all_windows
    def find_targets_in_window(self,windows, targets):
        windows_targets = []
        for window in windows:
            window_start = window[0]
            window_end = window[-1] if len(window) > 2 else window[1]
            window_targets = []

            for target_start, target_end, target_text in targets:
                # Check if the target is completely within the window
                if window_start <= target_start and target_end <= window_end:
                    window_targets.append((target_start, target_end))
            windows_targets.append(window_targets)
        return windows_targets

    def check_target_existence(self,targets, raw_spans):
        for target_start, target_end, _ in targets:
            target_found = False
        window_targets = self.find_targets_in_window(raw_spans, targets)
        for window_target in window_targets:    
            for window_target_start, window_target_end, _ in window_target:
                if target_start == window_target_start and target_end == window_target_end:
                    target_found = True
                    break
            if target_found:
                break
            if not target_found:
                print(f"目标索引 ({target_start}, {target_end}) 不存在于任何窗口")


    def transform2indices(self, dataset, mode='train'):
        res = []
        for document in dataset:
            
            sentences, speakers, replies, pieces2words = [document[w] for w in ['sentences', 'speakers', 'replies', 'pieces2words']] #分词后的sentence:'real', '##ly', '?',
            if mode == 'train':
                triplets, targets, aspects, opinions = [document[w] for w in ['triplets', 'targets', 'aspects', 'opinions']]
            doc_id = document['doc_id']
            
            # pdb.set_trace()
           
            # sentence_length = list(map(lambda x : len(x) + 2, sentences))
            sentence_length = list(map(lambda x : len(x) + 2, sentences))
            sentence_length_raw = list(map(lambda x : len(x), sentences))
            # token2sentid = [[i] * len(w) for i, w in enumerate(sentences)]
            token2sentid = [[i] * len(w) for i, w in enumerate(sentences)]
            token2sentid = [w for line in token2sentid for w in line]

            token2speaker = [[11] + [w] * len(z) + [10] for w, z in zip(speakers, sentences)]
            token2speaker = [w for line in token2speaker for w in line]

            # New token indices (with CLS and SEP) to old token indices (without CLS and SEP)
            new2old = {}
            cur_len = 0
            for i in range(len(sentence_length)):
                for j in range(sentence_length[i]):
                    if j == 0 or j == sentence_length[i] - 1:
                        new2old[len(new2old)] = -1 
                    else:
                        new2old[len(new2old)] = cur_len
                        cur_len += 1
           
            
            tokens = [[self.config.cls] + w + [self.config.sep] for w in sentences]
            
            # sentence_ids of each token (new token)
            nsentence_ids = [[i] * len(w) for i, w in enumerate(tokens)]
            nsentence_ids = [w for line in nsentence_ids for w in line]
            nsentence_ids_raw = [[i] * len(w) for i, w in enumerate(sentences)]
            nsentence_ids_raw = [w for line in nsentence_ids_raw for w in line]
            flatten_tokens = [w for line in tokens for w in line]
            sentence_end = [i - 1 for i, w in enumerate(flatten_tokens) if w == self.config.sep]
            sentence_start = [i + 1 for i, w in enumerate(flatten_tokens) if w == self.config.cls]

            utterance_spans = list(zip(sentence_start, sentence_end))
            
            utterance_index, token_index, thread_length, thread_nums = self.find_utterance_index(replies, sentence_length) 
            
            reply_mask, speaker_masks, thread_masks = self.get_neighbor(utterance_spans, replies, sum(sentence_length), speakers, thread_nums)
            
            input_ids = list(map(self.tokenizer.convert_tokens_to_ids, tokens))
            #pdb.set_trace()
            input_masks = [[1] * len(w) for w in input_ids]
            input_segments = [[0] * len(w) for w in input_ids]
          
            thread_sentence = []
            thread_sentence_index = []
            thread_sentence_spans = utterance_spans.copy()
            thread_sentence_index.append(list(thread_sentence_spans[0])) #root的token开始和结束位置
            thread_sentence.append([sentences[0]]) 
            windows_sentence = []
            windows_sentence_index = []
            
            for i,reply in enumerate(replies):
                if reply == -1:
                    continue
                elif reply == 0:
                    if len(windows_sentence_index)!=0:
                        windows_sentence = [ tw for tw in  windows_sentence]
                        thread_sentence.append(windows_sentence)
                        thread_sentence_index.append(windows_sentence_index)
                    windows_sentence_index = []
                    windows_sentence = []
                    windows_sentence_index.append(thread_sentence_spans[i])
                    windows_sentence.append(sentences[i])
                else:
                    for j in range(i,len(replies)):
                        thread_sentence_spans[j] = (thread_sentence_spans[j][0]-2,thread_sentence_spans[j][1]-2)
                    windows_sentence.append(sentences[i])
                    windows_sentence_index.append(thread_sentence_spans[i])
                if i == len(replies) - 1:
                    windows_sentence = [ tw for tw in  windows_sentence]
                    thread_sentence.append(windows_sentence)
                    windows_sentence_index = [index for index in windows_sentence_index]
                    thread_sentence_index.append(windows_sentence_index)
            thread_sentence = [[item for item in sublist] for sublist in thread_sentence ]

            origion_utterance_span = [] 
            start_index = 0
            for sentence in sentences:
                end_index = start_index + len(sentence) - 1
                origion_utterance_span.append((start_index, end_index))
                start_index = end_index + 1
            
            dialogue_threads = []
            dialogue_threads_raw = []
            thread_idx = []
            thread_idx_raw = []
            #由origion_utterance_span改为utterance_spans
            for i in range(len(origion_utterance_span)): 
                if replies[i] == -1:
                    thread_idx_raw.append(origion_utterance_span[i])
                    thread_idx.append(utterance_spans[i])
                elif replies[i] == 0:
                    if thread_idx:
                        dialogue_threads.append(thread_idx)
                        dialogue_threads_raw.append(thread_idx_raw)
                    thread_idx_raw = [origion_utterance_span[i]]
                    thread_idx = [utterance_spans[i]]
                else:
                    thread_idx_raw.append(origion_utterance_span[i])
                    thread_idx.append(utterance_spans[i])
            if thread_idx:
                dialogue_threads.append(thread_idx)
                dialogue_threads_raw.append(thread_idx_raw)
           
            dialogue_sentid = [[i]*(y-x+1) for i,(x,y) in enumerate(origion_utterance_span)]
            thread_sentid = []
            thread_index = []
            thread_index_raw = []
            idx = 0
            for i,v in enumerate(dialogue_threads[1:]):
                thread_index.append([dialogue_threads[0],v])
                thread_id = [dialogue_sentid[0]]
                for j in range(len(v)):
                    idx += 1
                    thread_id.append(dialogue_sentid[idx])
                thread_sentid.append(thread_id) 
            for i,v in enumerate(dialogue_threads_raw[1:]):
                thread_index_raw.append([dialogue_threads_raw[0],v])
                

            true_thread_utterance_span =  [[tuple for sublist in inner_list for tuple in sublist] for inner_list in thread_index] 
            true_thread_utterance_span_raw =   [[tuple for sublist in inner_list for tuple in sublist] for inner_list in thread_index_raw] 
            root_sentence = thread_sentence[0]
            thread_contexts = [root_sentence + thread_sentence[i] for i in range(1,len(thread_sentence))]
            
            all_windows_span = self.generate_all_windows(true_thread_utterance_span) 
            all_windows_span_raw  = self.generate_all_windows(true_thread_utterance_span_raw) 
            all_windows_sentid = self.generate_all_windows(thread_sentid)
            all_windows_context = self.generate_all_windows(thread_contexts)
            #pdb.set_trace()
            new_list = []
            for sublist1 in all_windows_context:
                new_sublist1 = []
                for sublist2 in sublist1:
                    new_sublist2 = []
                    for inner_list in sublist2:
                        new_sublist2.extend(inner_list)
                    new_sublist1.append(new_sublist2)
                new_list.append(new_sublist1)
            all_windows_context = new_list #示例的3层结构

            new_list_id = []
            for sublist1 in all_windows_sentid:
                new_sublist1 = []
                for sublist2 in sublist1:
                    new_sublist2 = []
                    for inner_list in sublist2:
                        new_sublist2.extend(inner_list)
                    new_sublist1.append(new_sublist2)
                new_list_id.append(new_sublist1)
            all_windows_sentid = new_list_id
            
            all_windows_tokens = [[[self.config.cls] + w + [self.config.sep] for w in sent] for sent in all_windows_context]
            windows_tokens = [token for sublist in all_windows_tokens for token in sublist]
            windows_new2old = {}
            

            all_windows_tokens_sentid = copy.deepcopy(all_windows_sentid) 
            for i in range(len(all_windows_sentid)):
                for j in range(len(all_windows_sentid[i])):
                    all_windows_tokens_sentid[i][j] = [all_windows_sentid[i][j][0]] + all_windows_sentid[i][j] + [all_windows_sentid[i][j][-1]]
           
            all_windows_tokens_spans = all_windows_span
            all_windows_tokens_spans_raw = all_windows_span_raw
            merged_all_windows_tokens_spans = [[tuple(item for subtuple in sublist2 for item in subtuple)] for sublist1 in all_windows_tokens_spans  for sublist2 in sublist1] 
            merged_all_windows_tokens_spans_raw = [[tuple(item for subtuple in sublist2 for item in subtuple)] for sublist1 in all_windows_tokens_spans_raw  for sublist2 in sublist1]
            flatten_all_windows_tokens_spans = [item for l1 in merged_all_windows_tokens_spans for item in l1] 
            flatten_all_windows_tokens_spans_raw = [item for l1 in merged_all_windows_tokens_spans_raw for item in l1]
            flatten_all_windows_tokens = [tokens for l1 in all_windows_tokens for l2 in l1 for tokens in l2]
            
            flatten_all_windows_tokens_sentid = [ids for l1 in all_windows_tokens_sentid for l2 in l1 for ids in l2] 
            flatten_all_windows_tokens_sentid_raw = [ids for l1 in all_windows_sentid for l2 in l1 for ids in l2] 
            windows_length = [[len(sublist) + 2 for sublist in sublist1]for sublist1 in all_windows_context] 
            windows_token2sentid_in = [[i] * len(w) for i, w in enumerate(thread_sentence)]
           
            windows_input_ids = list(map(self.tokenizer.convert_tokens_to_ids, windows_tokens))#2维，和input_ids格式一致
            windows_input_masks = [[1] * len(w) for w in windows_input_ids]
            windows_input_segments = [[0] * len(w) for w in windows_input_ids] 
           
            thread_sentence = [sent for st in thread_sentence for sent in st]
            thread_token2sentid_in = [[i] * len(w) for i, w in enumerate(thread_sentence)]
            thread_token2sentid_in = [w for line in thread_token2sentid_in for w in line]
            thread_token2sentid = []
            tn_ids = [[0]]
            prev_end=0
            for index,(start, end) in enumerate(thread_sentence_spans):
                if prev_end != start and start !=1:
                    
                    tn_ids.append([index-1])
                    tn_ids.append( [index]*(end-start+2))   
                else:
                    tn_ids.append( [index]*(end-start+1))  
                prev_end = end+1
                thread_token2sentid.append([index]*(end-start+1))
            tn_ids.append([len(thread_sentence_spans)-1])
            thread_nsentence_ids = [i for idss in tn_ids for i in idss]
            thread_nsentence_ids_in = [[i] * len(w)  for windows_tokens in all_windows_tokens for i, w in enumerate(windows_tokens)] 
            thread_nsentence_ids_in = [w for line in thread_nsentence_ids_in for w in line]#属于哪个thread, [0,0,0,0,0,1,1,1,1,1,2,2,2,2,3,3,3,3,3]

            thread_token2sentid  = [tids for t_id in thread_token2sentid for tids in t_id]
            # pdb.set_trace()
            thread_utterance_index, token_index, thread_length, thread_nums = self.find_utterance_index(replies, sentence_length)
          
            thread_tokens = [[self.config.cls] + w + [self.config.sep] for w in thread_sentence]         
            thread_flatten_tokens = [w for line in thread_tokens for w in line] #len()比flatten_tokens长了12，因为少了（10-4）*2个标识符
            thread_sentence_end = [i - 1 for i, w in enumerate(thread_flatten_tokens) if w == self.config.sep] #[20, 113, 203, 307]
            thread_sentence_start = [i + 1 for i, w in enumerate(thread_flatten_tokens) if w == self.config.cls] #[1, 23, 116, 206]
           
            thread_utterance_spans = list(zip(thread_sentence_start, thread_sentence_end))#[(1, 20), (23, 113), (116, 203), (206, 307)]

            thread_input_ids = list(map(self.tokenizer.convert_tokens_to_ids, thread_tokens))
            thread_input_masks = [[1] * len(w) for w in thread_input_ids]
            thread_input_segments = [[0] * len(w) for w in thread_input_ids]
             
         
            if mode == 'train':
              
                t_targets = [(s + 2 * thread_token2sentid_in[s] + 1, e + 2 * thread_token2sentid_in[s]) for s, e, t in targets]#映射到thread_flatten_token中的
                t_aspects = [(s + 2 * thread_token2sentid_in[s] + 1, e + 2 * thread_token2sentid_in[s]) for s, e, t in aspects]
                t_opinions = [(s + 2 * thread_token2sentid_in[s] + 1, e + 2 * thread_token2sentid_in[s]) for s, e, t, p in opinions]
                t_opinions = list(set(t_opinions))
           
                targets = [(s + 2 * token2sentid[s] + 1, e + 2 * token2sentid[s]) for s, e, t in targets]#由前闭后开变成闭区间
                aspects = [(s + 2 * token2sentid[s] + 1, e + 2 * token2sentid[s]) for s, e, t in aspects]
                opinions = [(s + 2 * token2sentid[s] + 1, e + 2 * token2sentid[s]) for s, e, t, p in opinions]
                opinions = list(set(opinions))
                full_triplets, new_triplets = [], []
                thread_full_triplets, thread_new_triplets = [],[]
          
                for t_s, t_e, a_s, a_e, o_s, o_e, polarity, t_t, a_t, o_t in triplets:
                    new_index = lambda start, end : (-1, -1) if start == -1 else (start + 2 * token2sentid[start] + 1, end + 2 * token2sentid[start])
                    thread_new_index = lambda start, end : (-1, -1) if start == -1 else (start + 2 * thread_token2sentid_in[start] + 1, end + 2 * thread_token2sentid_in[start])
                    o_t_s, o_t_e = new_index(t_s, t_e)
                    o_a_s, o_a_e = new_index(a_s, a_e)
                    o_o_s, o_o_e = new_index(o_s, o_e)
                    line = (o_t_s, o_t_e, o_a_s,o_a_e, o_o_s, o_o_e, self.polarity_dict[polarity])
                    full_triplets.append(line)
                    if all(w != -1 for w in [o_t_s, o_a_s, o_o_s]):
                        new_triplets.append(line)

                    t_t_s, t_t_e = thread_new_index(t_s, t_e)
                    t_a_s, t_a_e = thread_new_index(a_s, a_e)
                    t_o_s, t_o_e = thread_new_index(o_s, o_e)
                    line = (t_t_s, t_t_e, t_a_s,t_a_e, t_o_s, t_o_e, self.polarity_dict[polarity])
                    thread_full_triplets.append(line)
            
                    if all(w != -1 for w in [t_t_s, t_a_s, t_o_s]):
                        thread_new_triplets.append(line)
                relation_lists = self.wordpair.encode_relation(full_triplets) 
                thread_relation_lists = self.wordpair.encode_relation(thread_full_triplets)
                
                pairs = self.get_pair(full_triplets)
                thread_pairs = self.get_pair(thread_full_triplets)

                target_lists = self.wordpair.encode_entity(targets, 'ENT-T')
                aspect_lists = self.wordpair.encode_entity(aspects, 'ENT-A')
                opinion_lists = self.wordpair.encode_entity(opinions, 'ENT-O')

                thread_target_lists = self.wordpair.encode_entity(t_targets, 'ENT-T')
                thread_aspect_lists = self.wordpair.encode_entity(t_aspects, 'ENT-A')
                thread_opinion_lists = self.wordpair.encode_entity(t_opinions, 'ENT-O')
                
                entity_lists = target_lists + aspect_lists + opinion_lists
                polarity_lists = self.wordpair.encode_polarity(new_triplets)

                thread_entity_lists =  thread_target_lists +  thread_aspect_lists +  thread_opinion_lists
                thread_polarity_lists = self.wordpair.encode_polarity(thread_new_triplets)
                
                # pdb.set_trace()
            else:
                new_triplets, pairs, entity_lists, relation_lists, polarity_lists = [], [], [], [], []
                thread_new_triplets,  thread_pairs,  thread_entity_lists,  thread_relation_lists, thread_polarity_lists = [], [], [], [], []
            
            res.append((doc_id, input_ids, input_masks, input_segments, sentence_length,sentence_length_raw,nsentence_ids, nsentence_ids_raw,utterance_index, token_index, 
                        thread_length, token2speaker, reply_mask, speaker_masks, thread_masks, pieces2words, new2old, 
                        new_triplets, pairs, entity_lists, relation_lists, polarity_lists,
                        thread_input_ids,thread_input_masks,thread_input_segments,thread_utterance_spans, 
                        thread_new_triplets, thread_pairs, thread_entity_lists, thread_relation_lists, thread_polarity_lists,
                        thread_utterance_index,thread_sentence_spans,
                        flatten_all_windows_tokens_spans,flatten_all_windows_tokens_spans_raw,windows_input_ids,windows_input_masks,windows_input_segments,
                        flatten_all_windows_tokens_sentid,all_windows_tokens_sentid,windows_length))
        return res
    
    def forward(self):
        modes = 'train valid test'
        datasets = {}

        for mode in modes.split():
            data = self.read_data(mode)
            datasets[mode] = data

        label_dict = self.get_dict()

        res = {}
        for mode in modes.split():
            res[mode] = self.transform2indices(datasets[mode], mode)
        res['label_dict'] = label_dict
        return res
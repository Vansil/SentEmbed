import re
import pickle
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from collections import Counter

'''
Used to read SNLI data, GloVe vectors and to build a Vocabulary
'''


CLASS_ID = {'neutral': 0,
            'contradiction': 1,
            'entailment': 2,
            '-': -1}

class Vocabulary(object):
    '''Vocabulary all used words'''

    def __init__(self):
        self.counter = Counter()
        self.w2i = {}
        self.i2w = []

    def __len__(self):
        '''Number of words in the vocab'''
        return len(self.i2w)

    def count_word(self, word):
        '''Count a word, before building the vocab'''
        self.counter[word] += 1
    
    def add_word(self, word):
        '''Add a word to the vocab'''
        self.w2i[word] = len(self.w2i)
        self.i2w.append(word)    
        
    def build(self, min_freq=0):
        '''Build the vocab based on previously counted words'''
        self.add_word("<unk>")  # reserve 0 for <unk>
        self.add_word("<pad>")  # reserve 1 for <pad>
        
        word_freq = list(self.counter.items())
        word_freq.sort(key=lambda x: x[1], reverse=True)
        for word, freq in word_freq:
            if freq >= min_freq:
                self.add_word(word)

    def count_snli(self, dataset):
        '''Count all words in an snli dataset'''
        for _, (sent1, sent2) in dataset:
            for word in sent1+sent2:
                self.count_word(word)
    
    def count_glove(self, embedding):
        '''Count all words in an EmbeddingGlove object'''
        for word in embedding.i2w:
            if word not in ['<unk>','<pad>']:
                self.count_word(word)


class DatasetSnli(Dataset):
    '''SNLI dataset'''
    
    def __init__(self, data_path):
        '''
        Load the dataset
        Args:
            data_path (string): Path to data file
        '''
        self.data = self.snli_read_file(data_path)
        
    def __len__(self):
        '''Returns number of samples in dataset'''
        return len(self.data)
    
    def __getitem__(self, idx):
        '''Returns item at index'''
        return self.data[idx]
    
    @classmethod
    def split_sentence(cls, sent):
        '''
        Returns list of lowercase words
        Discards punctuation marks
        '''
        return re.sub(r"[^A-Za-z ]+", '', sent).lower().split()

    def snli_read_line(self, line):
        '''Read line from the raw SNLI dataset'''
        line_data = line.split("\t")
        cid = CLASS_ID[line_data[0]]
        sent1 = self.split_sentence(line_data[5])
        sent2 = self.split_sentence(line_data[6])
        if cid == -1:
            return None
        else:
            return (cid, (sent1, sent2))

    def snli_read_file(self, path):
        '''Read full raw SNLI dataset'''
        f = open(path, "r")
        f.readline()
        data = [t for t in [self.snli_read_line(l) for l in f] if t is not None]
        f.close()
        return data


class DataLoaderSnli(object):
    '''Dataloader for Snli data'''

    def __init__(self, dataset, vocab):
        # Store data and vocab
        self.sentence_pairs = [pair  for label,pair in dataset]
        self.labels         = [label for label,pair in dataset]
        self.vocab = vocab
        # Vars for sampling
        self._index_in_epoch = 0
        self._num_examples = len(dataset)
        self._epochs_completed = 0
        # Determine random order for sampling
        self._sample_order = np.arange(self._num_examples)
        np.random.shuffle(self._sample_order)

    def sentence2index(self, sent):
        '''Convert list of words to vocab indices'''
        return [self.vocab.w2i[w] if w in self.vocab.w2i.keys() else 0 for w in sent]

    def prepare_sent(self, sent, length):
        '''Converts words in a sentence to indices and applies padding'''
        return [self.vocab.w2i[word] if word in self.vocab.w2i.keys() else 0 for word in sent] \
            + [1]*(length-len(sent))

    def prepare_batch(self, batch):
        '''
        Converts words to indices and adds padding
        Args:
            batch: list of sentence pairs
        '''
        maxlen = max([max(len(s1),len(s2)) for (s1,s2) in batch])
        sent1_matrix = torch.LongTensor([self.prepare_sent(s1,maxlen) for (s1,s2) in batch])
        sent2_matrix = torch.LongTensor([self.prepare_sent(s2,maxlen) for (s1,s2) in batch])
        return sent1_matrix, sent2_matrix

    def prepare_sentences(self, batch):
        '''
        Same as prepare_batch, but for list of sentences instead of sentence pairs
        '''
        maxlen = max([len(s) for s in batch])
        sent_matrix = torch.LongTensor([self.prepare_sent(s,maxlen) for s in batch])
        return sent_matrix

    def prepare_manual(self, sent1, sent2):
        '''Returns a mini-batch based on two sentences'''
        s1 = DatasetSnli.split_sentence(sent1)
        s2 = DatasetSnli.split_sentence(sent2)
        maxlen = max(len(s1),len(s2))
        return torch.LongTensor([self.prepare_sent(s1,maxlen)]), torch.LongTensor([self.prepare_sent(s2,maxlen)]), torch.LongTensor([[-1]])

    def next_batch(self, batch_size):
        '''Returns a batch of certain size'''
        # Determine start and end data index
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            
            self._sample_order = np.arange(self._num_examples)
            np.random.shuffle(self._sample_order)

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples, "batch size {} is larger than data size {}".format(batch_size, self._num_examples)
        end = self._index_in_epoch
        # select data
        indices = [self._sample_order[i] for i in range(start,end)]
        sentence_pairs = [self.sentence_pairs[i] for i in indices]
        labels = torch.LongTensor([self.labels[i] for i in indices])
        # Words to index and padding
        sent1_matrix, sent2_matrix = self.prepare_batch(sentence_pairs)
        return sent1_matrix, sent2_matrix, labels


class EmbeddingGlove(object):
    '''Reads and contains a GloVe embedding'''

    def __init__(self, glove_file):
        '''Reads a Glove file'''
        self.i2w = ["<unk>","<pad>"] # index to word
        emb = [[0 for i in range(300)] for j in range(2)] # embedding for unkown and padding is zeros
        # Read lines one by one, checking for invalid line format
        with open(glove_file,"r") as f:
            countTot = countAdd = 0
            for line in f:
                word = line.split()[0]
                # Add line to filtered file if the word is in the vocab
                try:
                    embed = [float(x) for x in line.split()[1:]]
                    if len(embed) == 300:
                        self.i2w.append(word)
                        emb.append(embed)
                        countAdd += 1
                    else:
                        print('Unexpected line length (not 300): "{}"'.format(line))
                except:
                    print('Unexpected line (multiple words?): "{} ..."'.format(line[:20]))
                countTot += 1
                if countTot % 25000 == 0:
                    print("{} words loaded ({} invalid format)".format(countTot, countTot-countAdd))
            print("{} words loaded ({} invalid format)".format(countTot, countTot-countAdd))
        # Convert embedding to pytorch Tensor
        self.embedding = torch.Tensor(emb)


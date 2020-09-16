from io import open
import random
import math
import os
import re
from functools import reduce
import pdb
import torch
import numpy as np
from collections import defaultdict
import time
from model import device
from nltk.tokenize import word_tokenize

class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    hparams_grid = {
        # 'embedding_size': [50, 80],
        # 'batch_size': [16, 32, 48, 64],
        # 'hidden_size': [100, 200, 300, 350],
        # 'lr': [.00001, .00005, .0001, .0002, .001],
        # 'embedding_size': [116],
        'batch_size': [256],
        'hidden_size': [116],
        'lr': [.001],
        # 'l2': [.001, .0015, .002]
    }
    max_epoch = 28
    early_stopping = 14


class Vocab(object):

    def __init__(self):
        """
        Load SICK dataset and extract text and read the stories
        For a particular task, the class will read both train and test files
        and combine the vocabulary.
        """
        path = './SICK'
        # train_filename = 'SICK_train.txt'
        # dev_filename = 'SICK_trial.txt'
        # test_filename = 'SICK_test_annotated.txt'

        train_filename = 'snli_train.txt'
        dev_filename = 'snli_dev.txt'
        test_filename = 'snli_test.txt'
        self.w2v = os.path.exists('./vectors.txt')
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)

        self.compute_statistics(self.load_data(path, train_filename))
        self.idx = self.vocab_size

        self.train = [self.encode(sen_a, sen_b, label) for sen_a, sen_b, label in self.load_data(path, train_filename)]
        self.tunable_idx = self.idx

        self.dev = [self.encode(sen_a, sen_b, label) for sen_a, sen_b, label in self.load_data(path, dev_filename)]
        self.test = [self.encode(sen_a, sen_b, label) for sen_a, sen_b, label in self.load_data(path, test_filename)]

        # self.train = sorted(self.train_data, key=lambda z: (len(z[0]), len(z[1]), z[2]), reverse=True)

        self.train_size = self.tunable_idx - self.vocab_size + 1 #reserve extra one position for zero in RNN model to gen zeros
        self.infer_size = self.idx - self.tunable_idx + 1

    def load_data(self, path=".", file='SICK.txt'):
        """
        Fetch the SICK dataset and load it to memory.
        Returns:
            tuple: training and test files are returned
        """        
        if not os.path.isdir(path):
            raise ValueError("path: {0} is not a valid directory".format(path))

        file = path + '/' + file
        if os.path.exists(file) is False:
            raise ValueError("file: {0} does not exist".format(file))

        lines = open(file, encoding='utf-8').read().strip().split('\n')
        # data = []
        # ans =  {'NEUTRAL':0, 'ENTAILMENT':1, 'CONTRADICTION':2}
        ans = {'neutral':0, 'entailment':1, 'contradiction':2}

        for line in lines:
            row = line.split('\t')
            # data.append((Vocab.tokenize(row[0]), Vocab.tokenize(row[1]), ans[row[-1]]))
            yield Vocab.tokenize(row[0]+' ::'), Vocab.tokenize(row[1]), ans[row[-1]]
        # return data

    @staticmethod
    def tokenize(text):
        """
        Split a sentence into tokens including punctuation.
        """
        # return word_tokenize(text)
        # text = re.sub(r"[^a-zA-Z-/,]+", r" ", text)
        # text = re.sub(r"[^A-Za-z0-9,!.\/'-<>]", " ", text)
        # text = re.sub(r"\'ll", " will ", text)
        # text = re.sub(r"what's", "what is ", text)
        # text = re.sub(r"\'s", " ", text)
        # text = re.sub(r"\'ve", " have ", text)
        # text = re.sub(r"can't", "cannot ", text)
        # text = re.sub(r"n't", " not ", text)
        # text = re.sub(r"i'm", "i am ", text)
        # text = re.sub(r"\'re", " are ", text)
        # text = re.sub(r"\'d", " would ", text)
        # text = re.sub(r"\'ll", " will ", text)
        # text = re.sub(r",", " ", text)
        # text = re.sub(r"\.", " ", text)
        return [x.strip().lower() for x in re.split('(\W+)?', text) if x.strip()]

    def words_to_vector(self, words):
        """
        Convert a list of words into vector form.
        Arguments:
            words (list) : List of words.

        Returns:
            list : Vectorized list of words.
        """
        for w in words:
            if w not in self.word_to_index:
                self.word_to_index[w] = self.idx
                self.idx += 1
        return torch.LongTensor([self.word_to_index[w] for w in words])

    def encode(self, sa, sb, ans):
        """
        Convert (sentence a, sentence b, answer) word into vectors.

        Arguments:
            data (tuple) : Tuple of sentence a, sentence b, answer vectors.

        Returns:
            tuple : Tuple of sentence a, sentence b, answer vectors.
        """
        return (self.words_to_vector(sa), self.words_to_vector(sb), ans)

    def compute_statistics(self, tokens, cnt=1):
        """
        现在统一计算所有train/dev/test data
        Compute vocab, word index, and max length of sentence a and b.
        """
        if self.w2v:
            print("Loading Word2vec Model ...")
            lines = open('./vectors.txt', encoding='utf-8').read().split('\n')
            nwords, nsize = lines[0].split(maxsplit=1)
            self.vocab_size = int(nwords) + 1 #reserve the first one for padding zeros
            self.input_size = int(nsize)
            

            lines.pop(0)
            self.vocab = set() # set of words in train set and word_to_vec vocabulary

            self.word2vector_weights = torch.zeros(self.vocab_size, self.input_size).cuda()


            for nid, line in enumerate(lines): #还是先读入word2vec
                word, line = line.split(maxsplit=1)
                vec = line.split()
                # pdb.set_trace()
                self.word2vector_weights[nid+1, :] =  torch.FloatTensor([float(i) for i in vec])

                self.word_to_index[word] = nid+1
                self.index_to_word[nid+1] = word
                # self.vocab.add(word)
        else:
            raise ValueError("Word2vec model doesn't exist ...")

        # Reserve 0 for masking via pad_sequences and self.vocab_size - 1 for <UNK> token

def data_iterator(d, bsz):
    """
    Generator that can be used to iterate over this dataset.
    Yields:
        tuple : the next minibatch of data.
    """
    length = len(d)
    data = list(d)
    random.shuffle(data)

    nbatches = math.ceil(length / bsz)
    buf = list(zip(*data))
    a = buf[0]
    b = buf[1]
    ans = buf[2]

    for batch_index in range(nbatches):
        start = (batch_index * bsz) % length
        stop = ((batch_index + 1) * bsz) % length
        if start < stop:
            sa_tensor = pad_sentences(a[start:stop], True, bsz)   # batch_size × length × features
            sb_tensor = pad_sentences(b[start:stop], False, bsz)
            answer_tensor = torch.LongTensor(ans[start:stop])
            
            # la = [len(s)-1 for s in a[start:stop]]
            lb = [len(s)-1 for s in b[start:stop]]
        else:
            sa_tensor = pad_sentences(a[start:], True, bsz)
            sb_tensor = pad_sentences(b[start:], False, bsz)
            answer_tensor = torch.LongTensor(ans[start:] + (0,)*stop)

            # la = [len(s)-1 for s in a[start:]] + [-1 for s in range(stop)]
            lb = [len(s)-1 for s in b[start:]] + [-1 for s in range(stop)]

        yield sa_tensor.to(device), sb_tensor.to(device), answer_tensor.to(device), lb


def pad_sentences(sentences, pad_direct, batch_size, sentence_length=None):
    """
    Pad zeros to subjects
    """
    lengths = [len(sent) for sent in sentences]

    # nsamples = len(sentences)
    nsamples = batch_size
    if sentence_length is None:
        sentence_length = max(lengths)

    X = torch.LongTensor(batch_size, sentence_length).zero_() #not a tuple! single element!

    if pad_direct is True:
        for i, sent in enumerate(sentences):
            trunc = sent[-sentence_length:] #sentence_length is max, trunc is for each sentence
            X[i, -len(trunc):] = trunc
    else:
        for i, sent in enumerate(sentences):
            trunc = sent[-sentence_length:]
            X[i, :len(trunc)] = trunc
    return X


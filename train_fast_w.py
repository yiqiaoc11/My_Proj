'''
baseline model:
    standard intra-atten
    share parameters by default
'''
import logging
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import time
import numpy as np
import sys
from models.baseline_snli_w import encoder
from models.baseline_snli_w import atten
import argparse
from models.snli_data import snli_data
from models.snli_data import w2v
from random import shuffle
import pdb
from torch.nn.utils.rnn import pad_sequence
import torchtext
from torchtext import data
from torchtext.data import Dataset
from torchtext import datasets
import torchtext.vocab as vocab
from nltk import word_tokenize
import os
from os.path import join
import dill



class SNLIDataset(Dataset):
    @staticmethod
    def sort_key(idx):
        return data.interleave_keys(
            len(idx.premise), len(idx.hypothesis))

# def LCS(a, b):
#     DP = [[ 0 for i in range(len(a)+1)] for j in range(len(b)+1)]
#     DR = [[ '' for i in range(len(a)+1)] for j in range(len(b)+1)]
#     # for i in range(len(a)):
#     #   for j in range(len(b)):
#     #       if a[i] == b[j]:
#     #           DP[j+1][i+1] = DP[j][i] + ' ' + a[i]
#     #       else:
#     #           DP[j+1][i+1] = DP[j+1][i] if len(DP[j+1][i]) > len(DP[j][i+1]) else DP[j][i+1]
#     # return DP[len(b)][len(a)]

#     x, y = len(b), len(a)
#     for i in range(y):
#         for j in range(x):
#             if a[i] == b[j]:
#                 DP[j+1][i+1] = DP[j][i] + 1
#                 DR[j+1][i+1] = '↖'
#             else:
#                 if DP[j+1][i] > DP[j][i+1]:
#                     DP[j+1][i+1], DR[j+1][i+1] = DP[j+1][i], '←'
#                 else:
#                     DP[j+1][i+1], DR[j+1][i+1] = DP[j][i+1], '↑'

#     # res = []
#     while x > 0 and y > 0:
#         if DR[x][y] == '↑':
#             x -= 1
#         elif DR[x][y] == '←':
#             y -= 1
#         else:
#             x -= 1
#             # res.append(b[x])
#             b.pop(x)
#             y -= 1
#             a.pop(y)
#     if not a or not b:
#         return a+['#'], b+['#']
#     return a, b

def LCS(a, b, purge_all = False):
    # if a == b:
    #     return torch.LongTensor(a), torch.LongTensor(b), torch.LongTensor([2]), torch.tensor(len_a), torch.tensor(len_b)

    res = []
    len_a, len_b = len(a)+1, len(b)+1

    while bool(set(a[:len_a-1]) & set(b[:len_b-1])):
        DP = [[ 0 for i in range(len_a)] for j in range(len_b)]
        DR = [[ '' for i in range(len_a)] for j in range(len_b)]
        x, y = len(b), len(a)
        for i in range(y):
            for j in range(x):
                if a[i] == b[j]:
                    DP[j+1][i+1] = DP[j][i] + 1
                    DR[j+1][i+1] = '↖'
                else:
                    if DP[j+1][i] > DP[j][i+1]:
                        DP[j+1][i+1], DR[j+1][i+1] = DP[j+1][i], '←'
                    else:
                        DP[j+1][i+1], DR[j+1][i+1] = DP[j][i+1], '↑'

        while x > 0 and y > 0:
            if DR[x][y] == '↑':
                x -= 1
            elif DR[x][y] == '←':
                y -= 1
            else:
                x -= 1
                res.append(b[x])
                b.pop(x)
                y -= 1
                a.pop(y)
        # len_a -= len(res)
        # len_b -= len(res)
        if purge_all is False:
            break
    if not a or not b:
        return a+['#'], b+['#']
    res.append(37172)
    return a, b
    # return torch.LongTensor(a), torch.LongTensor(b), torch.LongTensor(res), torch.tensor(len_a), torch.tensor(len_b)



def train(args):
    inputs = data.Field(batch_first=True, include_lengths=True, tokenize=word_tokenize, lower=True)
    answers = data.Field(sequential=False, unk_token=None)
    

    def load_split_datasets(path, fields):
        # 加载examples
        with open(join(path, 'train_examples'), 'rb') as f:
            train_examples = dill.load(f)
        with open(join(path, 'dev_examples'), 'rb') as f:
            dev_examples = dill.load(f)
        with open(join(path, 'test_examples'), 'rb') as f:
            test_examples = dill.load(f)

        # 恢复数据集
        train = SNLIDataset(examples=train_examples, fields=fields)
        dev = SNLIDataset(examples=dev_examples, fields=fields)
        test = SNLIDataset(examples=test_examples, fields=fields)
        return train, dev, test


    if not os.path.exists(join(args.path, 'train_examples')):
        train, dev, test = datasets.SNLI.splits(inputs, answers, answers, root='./data1')
        os.mkdir(args.path)
        with open(join(args.path, 'train_examples'), 'wb') as f:
            dill.dump(train.examples, f)
        with open(join(args.path, 'dev_examples'), 'wb') as f:
            dill.dump(dev.examples, f)
        with open(join(args.path, 'test_examples'), 'wb') as f:
            dill.dump(test.examples, f)

    
    fields = {'premise': inputs, 'hypothesis': inputs, 'label': answers}
    train, dev, test = load_split_datasets(args.path, fields)

    # test[0].__dict__['premise'].pop()
    for i in range(train.__len__()):
        train[i].__dict__['premise'], train[i].__dict__['hypothesis'] = LCS(train[i].__dict__['premise'], train[i].__dict__['hypothesis'])
    for i in range(dev.__len__()):
        dev[i].__dict__['premise'], dev[i].__dict__['hypothesis'] = LCS(dev[i].__dict__['premise'], dev[i].__dict__['hypothesis'])
    for i in range(test.__len__()):
        test[i].__dict__['premise'], test[i].__dict__['hypothesis'] = LCS(test[i].__dict__['premise'], test[i].__dict__['hypothesis'])   
    # 检查分类结果用test[0].__dict__

    inputs.build_vocab(train, dev, test, vectors="glove.840B.300d", unk_init=torch.Tensor.normal_)
    answers.build_vocab(train)
        

        # buf = NumbersDataset(train.__len__())

        # make splits for data
        # if args.max_length < 0:
        #     args.max_length = 9999

    # initialize the logger
    # create logger
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # file handler
    if not os.path.exists(join(args.log_dir, args.log_fname)):
        os.mkdir(args.log_dir)
    fh = logging.FileHandler(args.log_dir + args.log_fname)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # stream handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # torch.cuda.set_device(args.gpu_id) # i.
    device = torch.device('cuda:'+args.gpu_id if torch.cuda.is_available() else 'cpu')

    for arg in vars(args):
        logger.info(str(arg) + ' ' + str(getattr(args, arg)))

    # load train/dev/test data
    # train data
    # logger.info('loading data...')
    # train_data = snli_data(args.train_file, args.max_length)
    # train_batches = train_data.batches

    train_lbl_size = len(answers.vocab.stoi)
    best_dev = []   # (epoch, dev_acc)
    # mask = torch.ones(args.batch_size, args.max_length, args.max_length, requires_grad=False).to(device)
    # build the model
    input_encoder = encoder(len(inputs.vocab.itos), args.embedding_size, args.hidden_size, args.para_init).to(device)
    input_encoder.embedding = nn.Embedding.from_pretrained(inputs.vocab.vectors).to(device)

    # inter_atten = atten(args.hidden_size, args.rbf_output_size, train_lbl_size, args.para_init) #rbf备份
    inter_atten = atten(args.hidden_size, train_lbl_size, args.batch_size, args.max_length, args.para_init).to(device)
    
    # if torch.cuda.is_available():  #ii.
    #     input_encoder.cuda() 
    #     inter_atten.cuda()

    para1 = filter(lambda p: p.requires_grad, input_encoder.parameters())
    para2 = inter_atten.parameters()

    if args.optimizer == 'Adagrad':
        input_optimizer = optim.Adagrad(para1, lr=args.lr, weight_decay=args.weight_decay)
        inter_atten_optimizer = optim.Adagrad(para2, lr=args.lr, weight_decay=args.weight_decay)
        # input_optimizer = optim.Adagrad(para1, lr=args.lr)
        # inter_atten_optimizer = optim.Adagrad(para2, lr=args.lr)
    elif args.optimizer == 'Adadelta':
        input_optimizer = optim.Adadelta(para1, lr=args.lr)
        inter_atten_optimizer = optim.Adadelta(para2, lr=args.lr)
    else:
        logger.info('No Optimizer.')
        sys.exit()

    # criterion1 = nn.NLLLoss(size_average=True)
    # criterion2 = nn.MSELoss(size_average=True)
    # criterion = nn.MultiMarginLoss(size_average=True)
    criterion = nn.CrossEntropyLoss()

    logger.info('start to train...')
    for k in range(args.epoch):

        total = 0.
        correct = 0.
        loss_data = 0.
        train_sents = 0.

        # shuffle()
        timer = time.time()
        
        train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=args.batch_size,
                                                                    sort_key=lambda x: data.interleave_keys(len(x.premise), len(x.hypothesis)), shuffle=False)



        for i, batch in enumerate(train_iter):

            train_src_batch, train_tgt_batch, train_lbl_batch = batch.premise[0], batch.hypothesis[0], batch.label
            mask = torch.zeros(batch.label.size(0), batch.premise[0].size(1), batch.hypothesis[0].size(1)).to(device) # feed mask to data blk in objects
            
            for idx in range(batch.label.size(0)):
                mask[idx, :batch.premise[1][idx], :batch.hypothesis[1][idx]] = 1.0

            # if torch.cuda.is_available():   #iii.
            #     train_src_batch = train_src_batch.cuda()
            #     train_tgt_batch = train_tgt_batch.cuda()
            #     train_lbl_batch = train_lbl_batch.cuda()
            #     # train_src_lens = train_src_lens.cuda()
            #     # train_tgt_lens = train_tgt_lens.cuda()

            batch_size = args.batch_size

            train_src_batch = train_src_batch.to(device)
            train_tgt_batch = train_tgt_batch.to(device)
            train_lbl_batch = train_lbl_batch.to(device)

            train_sents += batch_size

            input_optimizer.zero_grad()
            inter_atten_optimizer.zero_grad()

            # initialize the optimizer
            pdb.set_trace()
            if k == 0 and optim == 'Adagrad':
                for group in input_optimizer.param_groups:
                    for p in group['params']:
                        state = input_optimizer.state[p]
                        state['sum'] += args.Adagrad_init
                for group in inter_atten_optimizer.param_groups:
                    for p in group['params']:
                        state = inter_atten_optimizer.state[p]
                        state['sum'] += args.Adagrad_init


            input_encoder.train()
            inter_atten.train()

            train_src_linear, train_tgt_linear = input_encoder(train_src_batch, train_tgt_batch)
            log_prob = inter_atten(train_src_linear, train_tgt_linear, batch.premise[1], batch.hypothesis[1], mask)

            loss = criterion(log_prob, train_lbl_batch)
            loss.backward()

            # log_prob, x1, x2, y1, y2 = inter_atten(train_src_linear, train_tgt_linear)
            # loss = criterion1(log_prob, train_lbl_batch) + .08*(criterion2(x1, y1.detach()) + criterion2(x2, y2.detach()))
            # loss.backward()

            grad_norm = 0.
            para_norm = 0.

            for m in input_encoder.modules():

                if isinstance(m, nn.Linear):
                    grad_norm += m.weight.grad.data.norm() ** 2
                    para_norm += m.weight.data.norm() ** 2
                    if m.bias:
                        grad_norm += m.bias.grad.data.norm() ** 2
                        para_norm += m.bias.data.norm() ** 2

            for m in inter_atten.modules():
                # print(m, "<<<<<<<<<<<<<<<<<<<<<<<<<<<")                
                if isinstance(m, nn.Linear):                    
                    grad_norm += m.weight.grad.data.norm() ** 2
                    para_norm += m.weight.data.norm() ** 2
                    if m.bias is not None:
                        grad_norm += m.bias.grad.data.norm() ** 2
                        para_norm += m.bias.data.norm() ** 2

            grad_norm ** 0.5
            para_norm ** 0.5

            shrinkage = args.max_grad_norm / grad_norm
            if shrinkage < 1 :
                for m in input_encoder.modules():
                    # print m
                    if isinstance(m, nn.Linear):
                        m.weight.grad.data = m.weight.grad.data * shrinkage
                for m in inter_atten.modules():
                    # print m
                    if isinstance(m, nn.Linear):
                        m.weight.grad.data = m.weight.grad.data * shrinkage
                        if m.bias is not None:
                            m.bias.grad.data = m.bias.grad.data * shrinkage

            input_optimizer.step()
            inter_atten_optimizer.step()
            

            _, predict = log_prob.data.max(dim=1)
            total += train_lbl_batch.size(0)
            correct += torch.sum(predict == train_lbl_batch.data)
            loss_data += (loss.data.item() * batch_size)  # / train_lbl_batch.data.size()[0])


            if (i + 1) % args.display_interval == 0:
                logger.info('epoch %d, batches %d|%d, train-acc %.3f, loss %.3f, para-norm %.3f, grad-norm %.3f, time %.2fs, ' %
                            (k, i + 1, len(train_iter), correct.item() / total,
                             loss_data / train_sents, para_norm, grad_norm, time.time() - timer))
                train_sents = 0.
                timer = time.time()
                loss_data = 0.
                correct = 0.
                total = 0.
            if i == len(train_iter) - 1:
                logger.info('epoch %d, batches %d|%d, train-acc %.3f, loss %.3f, para-norm %.3f, grad-norm %.3f, time %.2fs, ' %
                            (k, i + 1, len(train_iter), correct.item() / total,
                             loss_data / train_sents, para_norm, grad_norm, time.time() - timer))
                train_sents = 0.
                timer = time.time()
                loss_data = 0.
                correct = 0.
                total = 0.

        # evaluate
        if (k + 1) % args.dev_interval == 0:
            input_encoder.eval()
            inter_atten.eval()
            correct = 0.
            total = 0.
            for i, batch in enumerate(dev_iter):

                dev_src_batch, dev_tgt_batch, dev_lbl_batch = batch.premise[0], batch.hypothesis[0], batch.label
                mask = torch.zeros(batch.label.size(0), batch.premise[0].size(1), batch.hypothesis[0].size(1)).to(device)

                for idx in range(batch.label.size(0)):
                    mask[idx, :batch.premise[1][idx], :batch.hypothesis[1][idx]] = 1.0

                # if torch.cuda.is_available():
                
                dev_src_batch = dev_src_batch.to(device)
                dev_tgt_batch = dev_tgt_batch.to(device)
                dev_lbl_batch = dev_lbl_batch.to(device)

                # if dev_lbl_batch.data.size(0) == 1:
                #     # simple sample batch
                #     dev_src_batch=torch.unsqueeze(dev_src_batch, 0)
                #     dev_tgt_batch=torch.unsqueeze(dev_tgt_batch, 0)
                
                dev_src_linear, dev_tgt_linear=input_encoder(dev_src_batch, dev_tgt_batch)
                log_prob =inter_atten(dev_src_linear, dev_tgt_linear, batch.premise[1], batch.hypothesis[1], mask)

                # log_prob, _, _, _, _, =inter_atten(dev_src_linear, dev_tgt_linear)
 
             # inter_atten(train_src_linear, train_tgt_linear)

            # loss = criterion1(log_prob, train_lbl_batch)
            # loss1 =  criterion2(x1, y1.detach()) + criterion2(x2, y2.detach())


                _, predict=log_prob.data.max(dim=1)
                total += dev_lbl_batch.data.size(0)
                correct += torch.sum(predict == dev_lbl_batch.data)

            dev_acc = correct.item() / total
            logger.info('dev-acc %.3f' % (dev_acc))

            if (k + 1) / args.dev_interval == 1:
                model_fname = '%s%s_epoch-%d_dev-acc-%.3f' %(args.model_path, args.log_fname.split('.')[0], k, dev_acc)
                torch.save(input_encoder.state_dict(), model_fname + '_input-encoder.pt')
                torch.save(inter_atten.state_dict(), model_fname + '_inter-atten.pt')
                best_dev.append((k, dev_acc, model_fname))
                logger.info('current best-dev:')
                for t in best_dev:
                    logger.info('\t%d %.3f' %(t[0], t[1]))
                logger.info('save model!') 
            else:
                if dev_acc > best_dev[-1][1]:
                    model_fname = '%s%s_epoch-%d_dev-acc-%.3f' %(args.model_path, args.log_fname.split('.')[0], k, dev_acc)
                    torch.save(input_encoder.state_dict(), model_fname + '_input-encoder.pt')
                    torch.save(inter_atten.state_dict(), model_fname + '_inter-atten.pt')
                    best_dev.append((k, dev_acc, model_fname))
                    logger.info('current best-dev:')
                    for t in best_dev:
                        logger.info('\t%d %.3f' %(t[0], t[1]))
                    logger.info('save model!')



    logger.info('training end!')
    # test
    best_model_fname = best_dev[-1][2]
    input_encoder.load_state_dict(torch.load(best_model_fname + '_input-encoder.pt'))
    inter_atten.load_state_dict(torch.load(best_model_fname + '_inter-atten.pt'))

    input_encoder.eval()
    inter_atten.eval()

    correct = 0.
    total = 0.

    for i, batch in enumerate(test_iter):
        test_src_batch, test_tgt_batch, test_lbl_batch = batch.premise[0], batch.hypothesis[0], batch.label

        mask = torch.zeros(batch.label.size(0), batch.premise[0].size(1), batch.hypothesis[0].size(1)).to(device)

        for idx in range(batch.label.size(0)):
            mask[idx, :batch.premise[1][idx], :batch.hypothesis[1][idx]] = 1.0

        test_src_batch = test_src_batch.to(device)
        test_tgt_batch = test_tgt_batch.to(device)
        test_lbl_batch = test_lbl_batch.to(device)

        test_src_linear, test_tgt_linear=input_encoder(
            test_src_batch, test_tgt_batch)
        log_prob=inter_atten(test_src_linear, test_tgt_linear, batch.premise[1], batch.hypothesis[1], mask)

        _, predict=log_prob.data.max(dim=1)
        total += test_lbl_batch.data.size()[0]
        correct += torch.sum(predict == test_lbl_batch.data)

    test_acc = correct.item() / total
    logger.info('test-acc %.3f' % (test_acc)) 

1
if __name__ == '__main__':
    parser=argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--path', help='training/dev/test data file',
                        type=str, default='data1/snli/snli_1.0/.dll/')

    parser.add_argument('--batch_size', help='batch size for snli datasets',
                        type=int, default=32)

    # parser.add_argument('--w2v_file', help='pretrained word vectors file (hdf5)',
    #                     type=str, default='data1/snli/baseline/glove.hdf5')

    parser.add_argument('--log_dir', help='log file directory',
                        type=str, default='data1/snli/experiment_struc/')

    parser.add_argument('--log_fname', help='log file name',
                        type=str, default='log54.log')

    parser.add_argument('--gpu_id', help='GPU device id',
                        type=str, default='7')

    parser.add_argument('--embedding_size', help='word embedding size',
                        type=int, default=300)

    parser.add_argument('--epoch', help='training epoch',
                        type=int, default=250)

    parser.add_argument('--dev_interval', help='interval for development',
                        type=int, default=1)

    parser.add_argument('--optimizer', help='optimizer',
                        type=str, default='Adagrad')

    # parser.add_argument('--optimizer', help='optimizer',
    #                     type=str, default='Adadelta')

    parser.add_argument('--Adagrad_init', help='initial accumulating values for gradients',
                        type=float, default=0.)

    parser.add_argument('--lr', help='learning rate',
                        type=float, default=0.05)

    # parser.add_argument('--lr', help='learning rate',
    #                     type=float, default=0.02)

    parser.add_argument('--hidden_size', help='hidden layer size',
                        type=int, default=300)

    parser.add_argument('--rbf_output_size', help='rbf output size',
                        type=int, default=700)

    parser.add_argument('--max_length', help='maximum length of sentences,\
                        -1 means no length limit',
                        type=int, default=64)

    parser.add_argument('--display_interval', help='interval of display',
                        type=int, default=1000)

    parser.add_argument('--max_grad_norm', help='If the norm of the gradient vector exceeds this renormalize it\
                               to have the norm equal to max_grad_norm',
                        type=float, default=5)

    parser.add_argument('--para_init', help='parameter initialization gaussian',
                        type=float, default=0.01)

    parser.add_argument('--weight_decay', help='l2 regularization',
                        type=float, default=1e-5)

    parser.add_argument('--model_path', help='path of model file (not include the name suffix',
                        type=str, default='data1/snli/experiment_struc/')

    # parser.add_argument('--max_length', help='max length of sequence',
    #                     type=int, default=128)

    args=parser.parse_args()
    # args.max_lenght = 10   # args can be set manually like this
    train(args)

else:
    pass
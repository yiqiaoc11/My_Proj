from utils import Config, Vocab, data_iterator
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from model import RNN, device
import itertools
import copy
import pdb
import random
import time
import math


# time profiling
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train(network, premise, hypothesis, h_len, target_variables, optimizer, criterion, bsz, vocab_size, infer_size):
    hidden = network.initHidden()

    optimizer.zero_grad()
    correct = 0
    loss = 0
    output = network(premise, hypothesis, h_len, hidden, vocab_size, infer_size)

    loss += criterion(output, target_variables[:output.size()[0]])

    loss.backward()
    optimizer.step()

    # if random.random() < .2:
    #     print('-'*20)
    #     for i, p in network.named_parameters():
    #         print(i)
    #         print(p)s
    #         print('GRAD', (p.grad.data.numpy()**2).sum())    

    topv, topi = output.data.topk(1)
    correct += int(torch.eq(topi.squeeze(1), target_variables[:output.size()[0]]).sum())

    return loss / bsz, correct

def run_epoch(input_data, model, optimizer, criterion, batch_size, vocab_size, infer_size, print_every=20):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    ndata = len(input_data[0])
    total_batchs = ndata // batch_size
    total_correct = 0
    total_data = 1

    batch = 1

    for a, b, ans, lb in data_iterator(input_data, batch_size): # adjust Weights
        _p = torch.t(Variable(a))
        _h = torch.t(Variable(b))
        _a = Variable(ans)

        loss, correct = train(model.to(device), _p.to(device), _h.to(device), lb, _a.to(device), optimizer, criterion, batch_size, vocab_size, infer_size)
        print_loss_total += loss

    #     # if batch % print_every == 0:
    #     #     print_loss_avg = print_loss_total / print_every
    #     #     print_loss_total = 0
    #     #     print('%s (%d %d%%) %.4f' % (timeSince(start, batch / totasl_batchs), 
    #     #                                           batch, batch / total_batchs * 100, print_loss_avg))
    #     #     plot_losses.append(print_loss_avg)
        batch += 1
        total_correct += correct
        total_data += batch_size

    return total_correct / total_data * 100


def eval(input_data, network, batch_size, vocab_size, infer_size):
    correct = 0
    total = 0

    for a, b, ans, lb in data_iterator(input_data, batch_size):
        _p = torch.t(Variable(a))
        _h = torch.t(Variable(b))
        _a = Variable(ans)

        hidden = network.initHidden()
        output = network(_p.to(device), _h.to(device), lb, hidden, vocab_size, infer_size)

        topv, topi = output.data.topk(1)
 
        correct += int(torch.eq(topi.squeeze(1), _a[:output.size()[0]].cuda('cuda:1').data).sum())
        total += _a.data.size()[0]

    return correct / total * 100

def test_TXTMAT():
    start = time.time()
    voc = Vocab()
    vocab_size = voc.vocab_size
    embed_size = voc.train_size
    infer_size = voc.infer_size
    input_size = voc.input_size    
    gen_config = Config()
    
    early_stop = gen_config.early_stopping
    best_val_epoch = 0
    keys = gen_config.hparams_grid.keys()
    
    for param in itertools.product(*gen_config.hparams_grid.values()):
        h_param = dict(zip(keys, param))

        # with open('results.txt', 'a') as f:
        #     f.write('h_param = {}\n'.format(h_param))
        best_val_acc = float(0)
        best_val_epoch = 0

        # input_size = h_param['embedding_size']
        hidden_size = h_param['hidden_size']
        # dp = h_param['dropout']
        batch_size = h_param['batch_size']
        lr = h_param['lr']
        # l2 = h_param['l2']

        model = RNN(voc, vocab_size, embed_size, infer_size, batch_size, input_size, hidden_size).to(device)

        print('Total loading time: {:.2f}s'.format(time.time() - start))    # 102s

        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(parameters, lr)
        criterion = nn.NLLLoss()

        for epoch in range(gen_config.max_epoch):
            print('Epoch {}'.format(epoch))
            
             
            train_acc = run_epoch(voc.train, model, optimizer, criterion, batch_size, vocab_size, embed_size)
            print('Train Acc.: {:.2f}'.format(train_acc))
            print('Total time: {:.2f}s'.format(time.time() - start))    # 117s w/o moving to cuda
            raise EnvironmentError
            # print('Total time: {}'.format(time.time() - start))
            # val_acc = eval(voc.dev, model, batch_size, vocab_size)

            # print('Valid Acc.: {:.2f}'.format(val_acc))

            val_acc = eval(voc.test, model, batch_size, vocab_size, embed_size)
            print('Valid Acc.: {:.2f}'.format(val_acc))

            # with open('results.txt', 'a') as f:
            #     f.write('Epoch {}\n'.format(epoch))
            #     f.write('Validation Acc.: {:.2f}\n'.format(val_acc))
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch
                # torch.save(model.state_dict(), './data')

            if epoch - best_val_epoch > gen_config.early_stopping:
                # model.load_state_dict(torch.load('./data'))
                break

        # test_acc = eval(voc.test, model, batch_size, vocab_size)
        # # with open('results.txt', 'a') as f:
        # #     f.write('=-='*5 + '\n')
        # #     f.write('Test Acc.: {:.2f}\n'.format(test_acc))
        # #     f.write('=-='*5 + '\n')
        # print('=-='*5)
        # print('Test Acc.: {:.2f}'.format(test_acc))
        # print('=-='*5)

if __name__ == "__main__":
    test_TXTMAT()
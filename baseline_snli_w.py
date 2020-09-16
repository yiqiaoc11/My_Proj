'''
baseline model for Stanford natural language inference
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
import torchvision
from torch.utils.tensorboard import SummaryWriter

class encoder(nn.Module):

    def __init__(self, num_embeddings, embedding_size, hidden_size, para_init):
        super(encoder, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.para_init = para_init

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_size)
        self.input_linear = nn.Linear(
            self.embedding_size, self.hidden_size, bias=False)  # linear transformation
        nn.init.normal_(self.embedding.weight.data[1:,:], self.para_init)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                
                m.weight.data.normal_(0, self.para_init)
                # m.bias.data.uniform_(-0.01, 0.01)

    def forward(self, sent1, sent2):
        '''
               sent: batch_size x length (Long tensor)
        '''
        # # for train_fast.py
        # batch_size = sent1[0].size(0)
        # sent1 = self.embedding(sent1[0])
        # sent2 = self.embedding(sent2[0])        
        batch_size = sent1.size(0)
        sent1 = self.embedding(sent1)
        sent2 = self.embedding(sent2)

        sent1 = sent1.view(-1, self.embedding_size)
        sent2 = sent2.view(-1, self.embedding_size)

        sent1_linear = self.input_linear(sent1).view(
            batch_size, -1, self.hidden_size)
        sent2_linear = self.input_linear(sent2).view(
            batch_size, -1, self.hidden_size)

        return sent1_linear, sent2_linear




class atten(nn.Module):
    '''
        intra sentence attention
    '''
    def __init__(self, hidden_size, label_size, batch_size, max_len, para_init):
        super(atten, self).__init__()

        self.hidden_size = hidden_size
        self.label_size = label_size
        self.para_init = para_init
        self.batch_size = batch_size

        
        self.mlp_f = self._mlp_layers(self.hidden_size, self.hidden_size)
        self.mlp_g = self._mlp_layers(2 * self.hidden_size, self.hidden_size, b=False) # print(1 if m.mlp_f[1].bias is None else 0)
        self.mlp_h = self._mlp_layers(2 * self.hidden_size, self.hidden_size)

        self.final_linear = nn.Linear(
            self.hidden_size, self.label_size, bias=True)

        self.log_prob = nn.LogSoftmax(-1) #

        '''initialize parameters'''
        for m in self.modules():
            # print m
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, self.para_init)
                if m.bias is not None:
                    m.bias.data.normal_(0, self.para_init)

    def _mlp_layers(self, input_dim, output_dim, b=True):
        mlp_layers = []
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            input_dim, output_dim, bias=b))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(p=0.2))
        mlp_layers.append(nn.Linear(
            output_dim, output_dim, bias=b))
        mlp_layers.append(nn.ReLU())
        return nn.Sequential(*mlp_layers)   # * used to unpack list

    # def init_mask(self, sent1_lens, sent2_lens, sent1_size, sent2_size, mask, zero=True):

    #     if zero is True:
    #         for i in range(sent1_lens.size(0)):
    #             mask[i, sent1_lens[i]:, :] = 0.0
    #             mask[i, :, sent2_lens[i]:] = 0.0
    #     else:
    #         for i in range(sent1_lens.size(0)):
    #             mask[i, sent1_lens[i]:, :] = 1.0
    #             mask[i, :, sent2_lens[i]:] = 1.0
    #     return


    def forward(self, sent1_linear, sent2_linear, sent1_lens, sent2_lens, mask):
        '''
            sent_linear: batch_size x length x hidden_size
        '''        

        len1 = sent1_linear.size(1)
        len2 = sent2_linear.size(1)
        # pdb.set_trace()
        
        '''attend'''
        f1 = self.mlp_f(sent1_linear.view(-1, self.hidden_size)) # 这个转换: .view(-1, self.hidden_size) 是有必要, 观察到2/3维的参数是有不同的
        f2 = self.mlp_f(sent2_linear.view(-1, self.hidden_size))

        f1 = f1.view(-1, len1, self.hidden_size)
        # batch_size x len1 x hidden_size
        f2 = f2.view(-1, len2, self.hidden_size)
        # batch_size x len2 x hidden_size

        # mask = torch.ones(sent1_linear.size(0), sent1_size, sent2_size).cuda()
        # mask = m[:sent1_linear.size(0), :sent1_size, :sent2_size]
        # self.init_mask(sent1_lens, sent2_lens, sent1_size, sent2_size, mask)

        score1 = torch.bmm(f1, torch.transpose(f2, 1, 2)) + (1.0 - mask) * -10000.0
        prob1 = nn.Softmax(-1)(score1.view(-1, len2)).view(-1, len1, len2) * mask
        # batch_size x len1 x len2

        score2 = torch.transpose(score1.contiguous(), 1, 2)
        score2 = score2.contiguous() 
        # e_{ji} batch_size x len2 x len1
        prob2 = nn.Softmax(-1)(score2.view(-1, len1)).view(-1, len2, len1) * torch.transpose(mask, 1, 2)
        # batch_size x len2 x len1

        # self.init_mask(sent1_lens, sent2_lens, sent1_size, sent2_size, mask, False)
        sent1_combine = torch.cat((mask[:, :, 0].unsqueeze(-1) * sent1_linear, torch.bmm(prob1, sent2_linear)), 2)
        # batch_size x len1 x (hidden_size x 2)
        sent2_combine = torch.cat((torch.transpose(mask[:, 0, :].unsqueeze(1), 1, 2) * sent2_linear, torch.bmm(prob2, sent1_linear)), 2) #有问题 sent2_linear后面都是void data!
        # batch_size x len2 x (hidden_size x 2)
        # pdb.set_trace()
        '''sum'''
        g1 = self.mlp_g(sent1_combine.view(-1, 2 * self.hidden_size))
        g2 = self.mlp_g(sent2_combine.view(-1, 2 * self.hidden_size))

        # g1 = self.mlp_g(f1.view(-1, 1 * self.hidden_size))
        # g2 = self.mlp_g(f2.view(-1, 1 * self.hidden_size))        
        g1 = g1.view(-1, len1, self.hidden_size)
        # batch_size x len1 x hidden_size
        g2 = g2.view(-1, len2, self.hidden_size)
        # batch_size x len2 x hidden_size


        sent1_output = torch.sum(g1, 1)  # batch_size x 1 x hidden_size
        sent1_output = torch.squeeze(sent1_output, 1)
        sent2_output = torch.sum(g2, 1)  # batch_size x 1 x hidden_size
        sent2_output = torch.squeeze(sent2_output, 1)

        input_combine = torch.cat((sent1_output, sent2_output), 1)
        # batch_size x (2 * hidden_size)
        h = self.mlp_h(input_combine)
        # batch_size * hidden_size

        # if sample_id == 15:
        #     print '-2 layer'
        #     print h.data[:, 100:150]

        h = self.final_linear(h)

        # print 'final layer'
        # print h.data

        log_prob = self.log_prob(h)
        # pdb.set_trace()

        return log_prob

# w = SummaryWriter()
# net = atten(400, 3, 2, 30, .1)
# input_x = torch.zeros([2, 20, 400], dtype=torch.float)
# input_y = torch.zeros([2, 20, 400], dtype=torch.float)
# input_c = torch.tensor(20)
# input_d = torch.tensor(20)
# input_e = torch.zeros([2, 20, 20], dtype=torch.float)
# w.add_graph(net, (input_x, input_y, input_c, input_d, input_e))
# w.close()
# current best-dev:
#     0 0.657
#     1 0.678
#     2 0.703
#     3 0.710
#     4 0.723
#     5 0.725
#     6 0.733
#     8 0.735
#     9 0.739
#     10 0.744
#     11 0.745
#     12 0.747
#     13 0.749
#     14 0.754
#     15 0.758
#     17 0.760
#     18 0.762
#     20 0.767
#     21 0.771
#     22 0.771
#     23 0.776
#     25 0.779
#     26 0.781
#     29 0.781
#     30 0.782
#     31 0.785
#     35 0.786
#     39 0.787
#     40 0.791
#     43 0.791
#     44 0.792
#     45 0.795
#     54 0.796
#     56 0.798
#     61 0.798
#     63 0.799
#     64 0.800
#     65 0.800
#     70 0.802
#     71 0.803
#     75 0.805
#     82 0.806
#     103 0.807
#     109 0.807
#     110 0.807
#     111 0.807
#     113 0.810
#     143 0.810
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.backends.thnn import backend # backend.LSTMCell etc...
import pdb
import random
import math
from torch.nn import init
import numpy as np


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):

    def __init__(self, vocab, word2vec_size, embedding_size, infer_size, batch_size, embedding_dim, hidden_size=100, dropout=.1):
        super(RNN, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = hidden_size #LSTM input

        self.word2vec = nn.Embedding.from_pretrained(vocab.word2vector_weights)

        self.embedding = nn.Embedding(embedding_size, embedding_dim, padding_idx=0) #researve zero vector
        nn.init.uniform_(self.embedding.weight.data[1:,:], -.05, .05)

        self.infer_embedding = nn.Embedding(infer_size, embedding_dim, padding_idx=0)
        nn.init.uniform_(self.embedding.weight.data[1:,:], -1.0, 1.0)
        self.freezeLayer(self.infer_embedding)


        self.x_proj = nn.Linear(embedding_dim, self.hidden_size)
        self.lstm_p = nn.LSTM(self.input_size, hidden_size)
        self.init_parameters(self.lstm_p)

        self.lstm_h = nn.LSTM(self.input_size, hidden_size)
        self.init_parameters(self.lstm_h)

        # self.reset_parameters(self.lstm_h)
        # The linear layer that maps from hidden state space to tag space
        self.i2o = nn.Linear(hidden_size, 3) #只需要最后一个hidden state
        
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sentence_p, sentence_h, h_len, hidden, vocab_size, infer_size):
        """s
        >=vocab_size 与 <vocab_size 为两个部分放入两个不同embedding中以mask方式实现, 同时padding_idx生成 0 embeddings,
        使两个部分不影响.
        sentence_p  # 44 × 32 × 50
        sentence_h  #  6 × 32 × 50
        """
       
        sentence_p1, sentence_p2, sentence_p3 = self.splitEmbedding(sentence_p, vocab_size, infer_size)
        embed_p1 = self.embedding(sentence_p1)
        embed_p2 = self.word2vec(sentence_p2)
        embed_p3 = self.infer_embedding(sentence_p3)
        embedded_p = embed_p1 + embed_p2 + embed_p3
       
        sentence_h1, sentence_h2, sentence_h3 = self.splitEmbedding(sentence_h, vocab_size, infer_size)
        embed_h1 = self.embedding(sentence_h1)
        embed_h2 = self.word2vec(sentence_h2)
        embed_h3 = self.infer_embedding(sentence_h3)
        embedded_h = embed_h1 + embed_h2 + embed_h3 #combine two tensors

        embedded_p = self.x_proj(embedded_p)
        output_p, h = self.lstm_p(embedded_p, hidden) # 44 × 32 × 100  seq. × batch × features

        embedded_h = self.x_proj(embedded_h)
        output_h, _ = self.lstm_h(embedded_h, h)

        # output = torch.cat([output_h[j,i,:].unsqueeze(0) for i, j in enumerate(h_len)], 0)
        p_output = torch.cat([output_h[j,i,:].unsqueeze(0) for i, j in enumerate(h_len) if j > 0], 0)
        # p_output = self.dp(p_output)
        output = self.i2o(F.tanh(p_output))

        output = self.softmax(output)

        return output

    def initHidden(self):
        h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size)).to(device)
        c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size)).to(device)
        return (h0, c0)

    def freezeLayer(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def init_parameters(self, model, Xavier=True):
        stdv = 1.0 / math.sqrt(model.hidden_size)
        if Xavier:
            nn.init.xavier_normal_(model.weight_hh_l0)
            nn.init.xavier_normal_(model.weight_ih_l0)
            model.bias_hh_l0.data.uniform_(-stdv, stdv)
            model.bias_ih_l0.data.uniform_(-stdv, stdv)
        else:
            stdv = 1.0 / math.sqrt(model.hidden_size)
            for weight in model.parameters():
                weight.data.uniform_(-stdv, stdv)

    def splitEmbedding(self, tensor, bd, level):
        embedding_idx = torch.clamp(tensor-bd+1, min=0, max=level)%level # e.g. +1 腾出1个位置给embedding生成0
        word2vec_idx = torch.clamp(tensor, max=bd)%bd #replace >=bd by 0
        rand_idx = torch.clamp(tensor-bd-level+2, min=0)
        return embedding_idx, word2vec_idx, rand_idx

# class RNN(nn.Module):
#     """
#     Rochtaschel 2015 shared coding
#     """ 
#     def __init__(self, w2v_embedding_size, embedding_size, pretrain_wts, batch_size, input_size=50, hidden_size=100, dropout=1, word2vec=False):
#         super(RNN, self).__init__()
#         self.batch_size = batch_size
#         self.hidden_size = hidden_size
#         if word2vec:
#             self.w2v_embedding = nn.Embedding(w2v_embedding_size, input_size)
#             self.w2v_embedding.weight.data.copy_(pretrain_wts)

#         self.embedding = nn.Embedding(embedding_size, input_size)

#         self.lstm_p = nn.LSTM(input_size, hidden_size)
#         self.lstm_h = nn.LSTM(input_size, hidden_size)
#         # The linear layer that maps from hidden state space to tag space
#         self.i2o = nn.Linear(hidden_size, 3) #只需要最后一个hidden state
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, sentence_p, sentence_h, hidden):
#         embedded_p = self.embedding(sentence_p)  # 44 × 32 × 50
#         embedded_h = self.embedding(sentence_h)  #  6 × 32 × 50

#         output_p, (h, c) = self.lstm_p(embedded_p, hidden) # 44 × 32 × 100  seq. × batch × features
#         # (h, _) = self.initHidden()

#         output_h, _ = self.lstm_h(embedded_h, (h, c))
#         # combined_input = torch.transpose(torch.cat((output_s, output_q), 0), 0, 1)
#         # output = self.i2o(combined_input.contiguous().view(32, -1))

#         output = self.i2o(F.tanh(output_h[-1]))
#         output = self.softmax(output)
#         return output

#     def initHidden(self):
#         h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
#         c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
#         return (h0, c0)


# # class LayerNorm(nn.Module):
# #     def __init__(self, features, eps=1e-5):
# #         super().__init__()
# #         # self.gamma = nn.Parameter(torch.ones(features))
# #         # self.beta = nn.Parameter(torch.zeros(features))
# #         self.gamma = Variable(torch.ones(features))
# #         self.beta = Variable(torch.zeros(features))
# #         self.eps = eps

# #     def forward(self, x):
# #         mean = x.mean(-1, keepdim=True)  # -1 表示最后一维 
# #         std = x.std(-1, keepdim=True)

# #         return self.gamma * (x - mean) / (std + self.eps) + self.beta

# # class RNN(nn.Module):
# #     """add generated network based on the Rochtaschel's LSTM
# #     for metanetwork:
# #         x^_t = [x_t, h_t-1]
# #         h^_t 
# #     arguments: batch_size: data width within a batch for one LSTM cell
# #                u_hidden_size: hidden features for upper net (basic net)
# #                m_hidden_size: hidden features for meta net
# #                output_size: output features for upper net
# #     """
# #     def __init__(self, embedding_size, batch_size, u_hidden_size, m_hidden_size, output_size):
# #         super(RNN, self).__init__()
# #         combined_mlp_size = embedding_size + u_hidden_size # for [x^_t h^_t-1]

# #         self.ln = LayerNorm(u_hidden_size)
# #         self.lstm = nn.LSTMCell(self.combined_mlp_size, m_hidden_size)
# #         self.weight_h = torch.Tensor(self.batch_size, self.combined_mlp_size, u_hidden_size)
# #         self.layer_weights_hh_a = F.sigmoid(nn.Linear(m_hidden_size, combined_mlp_size))
# #         self.layer_weights_hh_b = F.sigmoid(nn.Linear(m_hidden_size, m_hidden_size))
# #         self.layer_weights_ho_a = F.sigmoid(nn.Linear(m_hidden_size, combined_mlp_size))
# #         self.layer_weights_ho_b = F.sigmoid(nn.Linear(m_hidden_size, m_hidden_size))
# #         self.layer_weights_hi_a = F.sigmoid(nn.Linear(m_hidden_size, combined_mlp_size))
# #         self.layer_weights_hi_b = F.sigmoid(nn.Linear(m_hidden_size, m_hidden_size))
# #         self.layer_weights_hc_a = F.sigmoid(nn.Linear(m_hidden_size, combined_mlp_size))
# #         self.layer_weights_hc_b = F.sigmoid(nn.Linear(m_hidden_size, m_hidden_size))
        
# #         self.layer_shifts_hh = F.sigmoid(nn.Linear(m_hidden_size, m_hidden_size))
# #         self.layer_shifts_ho = F.sigmoid(nn.Linear(m_hidden_size, m_hidden_size))
# #         self.layer_shifts_hi = F.sigmoid(nn.Linear(m_hidden_size, m_hidden_size))
# #         self.layer_shifts_hc = F.sigmoid(nn.Linear(m_hidden_size, m_hidden_size))

# #         torch.nn.init.uniform_(self.fw, -.01, .01)

# #     def forward(self, data, programme, u_hidden, m_hidden):
# #         #u_hidden = uh0, uc0  m_hidden = mh0, mc0, meta1([x1, uh0], (mh0, mc0)) -> (mh1, mc1)  basic1(x1, (uh0, uc0)) -> (uh1, mh1)
# #         embedded = self.Embedding(data) # 44 × 32 × 50
# #         for i in range(embedded.size()[0]):
# #             m_h, m_c = self.lstm(torch.cat((embedded[i], u_hidden[0]), 1), m_hidden) #Inputs: input, (h_0, c_0)
# #             m_hidden = (m_h, m_c)

# #             whh_a = self.layer_weights_hh_a(m_h)
# #             whh_b = self.layer_weights_hh_b(m_h)
# #             who_a = self.layer_weights_ho_a(m_h)
# #             who_b = self.layer_weights_ho_b(m_h)
# #             whi_a = self.layer_weights_hi_a(m_h)
# #             whi_b = self.layer_weights_hi_b(m_h)
# #             whc_a = self.layer_weights_hc_a(m_h)
# #             whc_b = self.layer_weights_hc_b(m_h)
            
# #             weight_hh = whh_a * whh_b
# #             weight_ho = who_a * who_b
# #             weight_hi = whi_a * whi_b
# #             weight_hc = whc_a * whc_b

# #             shh = self.layer_shifts_hh(m_h)
# #             sho = self.layer_shifts_ho(m_h)
# #             shi = self.layer_shifts_hi(m_h)
# #             shc = self.layer_shifts_hc(m_h)

# #             *u = backend.LSTMCell(embedded[i], u_hidden, )
# #             u_hidden = *u





        
#           # self.fw = self.gate(m_h, self.combined_mlp_size, self.m_hidden_size, self.fw) #alpha_gamma = combined_mlp_size, beta_delta = m_hidden_size

#     # def gate(self, input_weights, alpha_gamma, beta_delta, fast_weights): #update F1, F2
#     #     alpha = torch.unsqueeze(input_weights[:, :alpha_gamma], 2)
#     #     beta  = torch.unsqueeze(input_weights[:, alpha_gamma:alpha_gamma + beta_delta], 1)
#     #     gamma = torch.unsqueeze(input_weights[:, alpha_gamma + beta_delta:2*alpha_gamma + beta_delta], 2)
#     #     delta = torch.unsqueeze(input_weights[:, 2*alpha_gamma + beta_delta:], 1)

#     #     H = F.tanh(alpha) * F.tanh(beta)
#     #     T = F.sigmoid(gamma) * F.sigmoid(delta)
#     #     fast_weights = T.mul(H) + torch.mul(Variable(torch.ones(alpha_gamma, beta_delta)) - T, fast_weights)

#     #     return fast_weights

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import roc_auc_score
from utils import load_emb_from_txt, save_emb_to_txt


class SphericalEmbeddingModel(nn.Module):

    def __init__(self, emb_dimension, word_size, document_size, category_size,
                        text_margin=0.25, intra_margin = 0.9, inter_margin = 0.0):
        super(SphericalEmbeddingModel, self).__init__()

        self.emb_dimension = emb_dimension
        self.word_size = word_size
        self.document_size = document_size
        self.category_size = category_size

        self.text_margin = text_margin
        self.intra_margin = intra_margin
        self.inter_margin = inter_margin

        self.u_embeddings = nn.Embedding(word_size, emb_dimension, padding_idx=0)
        self.v_embeddings = nn.Embedding(word_size, emb_dimension, padding_idx=0)
        self.d_embeddings = nn.Embedding(document_size, emb_dimension)
        self.c_embeddings = nn.Embedding(category_size, emb_dimension)
        
        initrange = 0.5 / self.emb_dimension
        nn.init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        nn.init.uniform_(self.v_embeddings.weight.data, -initrange, initrange)
        nn.init.uniform_(self.d_embeddings.weight.data, -initrange, initrange)
        nn.init.uniform_(self.c_embeddings.weight.data, -initrange, initrange)
        self.normalize_embeddings()

    def riemannian_gradient(self, weight):
        weight.grad -= torch.mul(weight.data, weight.grad).sum(dim=1, keepdim=True) * weight.data

    def normalize_embeddings(self):
        self.u_embeddings.weight.data = F.normalize(self.u_embeddings.weight.data, dim=-1)
        self.v_embeddings.weight.data = F.normalize(self.v_embeddings.weight.data, dim=-1)
        self.d_embeddings.weight.data = F.normalize(self.d_embeddings.weight.data, dim=-1)
        self.c_embeddings.weight.data = F.normalize(self.c_embeddings.weight.data, dim=-1)

    def init_keywords(self, keywords):
        self.keywords = nn.Parameter(torch.LongTensor(keywords).unsqueeze(dim=1), requires_grad=False)
        self.keyword_size = 1

    def forward_local(self, pos_u, pos_v, neg_v):
        pos_u_emb = F.normalize(self.v_embeddings(pos_u), dim=-1) 
        pos_v_emb = F.normalize(self.u_embeddings(pos_v), dim=-1)
        neg_v_emb = F.normalize(self.u_embeddings(neg_v), dim=-1)

        pos_score = torch.sum(torch.mul(pos_u_emb, pos_v_emb), dim=1, keepdim=True)
        neg_score = torch.bmm(neg_v_emb, pos_u_emb.unsqueeze(2)).squeeze()

        margin_loss = torch.clamp(pos_score - neg_score - self.text_margin, max=0)
        return - torch.sum(torch.sum(margin_loss, dim=1), dim=0)

    def forward_global(self, pos_d, pos_w, neg_w):
        pos_d_emb = F.normalize(self.d_embeddings(pos_d), dim=-1)
        pos_w_emb = F.normalize(self.u_embeddings(pos_w), dim=-1)
        neg_w_emb = F.normalize(self.u_embeddings(neg_w), dim=-1)
        
        pos_score = torch.sum(torch.mul(pos_d_emb, pos_w_emb), dim=1, keepdim=True)
        neg_score = torch.bmm(neg_w_emb, pos_d_emb.unsqueeze(2)).squeeze()

        margin_loss = torch.clamp(pos_score - neg_score - self.text_margin, max=0)
        return - torch.sum(torch.sum(margin_loss, dim=1), dim=0)    

    def forward_category(self):
        u_emb = F.normalize(self.u_embeddings(self.keywords), dim=-1)
        c_emb = F.normalize(self.c_embeddings.weight, dim=-1)
        
        pos_score = torch.bmm(u_emb, c_emb.unsqueeze(dim=2)).squeeze(dim=-1)
        neg_score = torch.matmul(c_emb, c_emb.transpose(0, 1))
        neg_score.masked_fill_(torch.eye(c_emb.shape[0], dtype=torch.bool, device=neg_score.device), 0)

        intra_margin_loss = torch.sum(torch.clamp(pos_score - self.intra_margin, max=0), dim=1)
        inter_margin_loss = torch.sum(torch.clamp(-(neg_score - self.inter_margin), max=0), dim=1)
        return - torch.sum(intra_margin_loss + inter_margin_loss, dim=0)

    def forward_selftraining(self):
        d_emb = F.normalize(self.d_embeddings.weight, dim=-1)
        c_emb = F.normalize(self.c_embeddings.weight, dim=-1)
        
        score = torch.softmax(torch.matmul(d_emb, c_emb.transpose(0, 1)), dim=1)

        target = torch.pow(score, 2) / torch.sum(score, dim=0, keepdim=True)
        target = target / torch.sum(target, dim=1, keepdim=True)

        loss = - target.detach() * torch.log(score + 1e-16)
        return torch.sum(torch.sum(loss, dim=1), dim=0)

    def extend_category(self, extend_size=1):
        u_emb = F.normalize(self.u_embeddings.weight, dim=-1)
        c_emb = F.normalize(self.c_embeddings.weight, dim=-1)
        
        score = torch.matmul(c_emb, u_emb.transpose(0, 1))
        sorted_words = torch.argsort(score, dim=1, descending=True)
        new_keywords = [[] for c in range(self.category_size)]

        for c in range(self.category_size):
            rank = 0
            while True:
                target_word = int(sorted_words[c, rank].item())
                if target_word not in self.keywords[c, :]:
                    new_keywords[c].append(target_word)
                    if len(new_keywords[c]) == extend_size:  break
                rank += 1

        new_keywords = torch.LongTensor(new_keywords).to(self.keywords.device)
        self.keywords.data = torch.cat([self.keywords, new_keywords], dim=1)
        self.keyword_size += extend_size

    def update_intermargin(self):
        c_emb = F.normalize(self.c_embeddings.weight.detach(), dim=-1)
        score = torch.matmul(c_emb, c_emb.transpose(0, 1))
        self.inter_margin = (score.sum() - self.category_size) / (self.category_size * (self.category_size - 1))

    def load_pretrained_wordemb(self, filename, word2id):
        with open(filename, 'r') as f:
            metainfos = f.readline().split()
            emb_size, emb_dim = int(metainfos[0]), int(metainfos[1])
            for line in f:
                fields = line.split()
                word = ' '.join(fields[:-emb_dim])
                if word in word2id:
                    self.u_embeddings.weight.data[word2id[word]] = torch.Tensor([float(x) for x in fields[-emb_dim:]])

    def get_embeddings(self, npflag=False):
        if npflag:
            w_emb = F.normalize(self.u_embeddings.weight, dim=-1).detach().cpu().numpy()
            d_emb = F.normalize(self.d_embeddings.weight, dim=-1).detach().cpu().numpy()
            c_emb = F.normalize(self.c_embeddings.weight, dim=-1).detach().cpu().numpy()
        else:
            w_emb = F.normalize(self.u_embeddings.weight, dim=-1).detach()
            d_emb = F.normalize(self.d_embeddings.weight, dim=-1).detach()
            c_emb = F.normalize(self.c_embeddings.weight, dim=-1).detach()
        return w_emb, d_emb, c_emb

    def load_embeddings(self, w_filename, d_filename, c_filename=None, npy_flag=True):
        if os.path.isfile(w_filename):
            self.u_embeddings.weight.data = torch.Tensor(np.load(w_filename)) if npy_flag \
                                            else torch.Tensor(load_emb_from_txt(w_filename))
        if os.path.isfile(d_filename):
            self.d_embeddings.weight.data = torch.Tensor(np.load(d_filename)) if npy_flag \
                                            else torch.Tensor(load_emb_from_txt(d_filename))
        if os.path.isfile(c_filename):
            self.c_embeddings.weight.data = torch.Tensor(np.load(c_filename)) if npy_flag \
                                            else torch.Tensor(load_emb_from_txt(c_filename)) 

    def save_embeddings(self, w_filename, d_filename, c_filename, npy_flag=True):
        w_emb, d_emb, c_emb = self.get_embeddings(npflag=True)
        
        if npy_flag:
            np.save(w_filename, w_emb)
            np.save(d_filename, d_emb)
            np.save(c_filename, c_emb)

        else:
            w_ids = [id2word[i] for i in range(w_emb.shape[0])]
            d_ids = [str(i) for i in range(d_emb.shape[0])]
            c_ids = [id2word[k] for k in self.keywords[:, 0]]
            
            save_emb_to_txt(w_filename, w_ids, w_emb)
            save_emb_to_txt(w_filename, d_ids, d_emb)
            save_emb_to_txt(w_filename, c_ids, c_emb)

    def get_confdocs(self, temperature, conf_ratio, k=30, j=30, rel_type='vmfd'):
        w_emb, d_emb, c_emb = self.get_embeddings()

        if rel_type == 'vmfw':
            wscore_mat = torch.matmul(w_emb, c_emb.transpose(1, 0))
            dscore_mat = torch.matmul(d_emb, c_emb.transpose(1, 0))

            # retrieving j-nearest words for each document
            dwsim_mat = torch.matmul(d_emb, w_emb.transpose(1, 0))
            jnn_score_mat, jnn_index_mat = torch.sort(dwsim_mat, dim=1)

            # retrieving k-nearest documents for each document
            ddsim_mat = torch.matmul(d_emb, d_emb.transpose(1, 0))
            knn_score_mat, knn_index_mat = torch.sort(ddsim_mat, dim=1)

            jnn_indices, jnn_scores = jnn_index_mat[:,-j:], jnn_score_mat[:,-j:]
            knn_indices, knn_scores = knn_index_mat[:,-k:], knn_score_mat[:,-k:]
 
            dscore_mat = torch.sum(jnn_scores.unsqueeze(dim=2) * wscore_mat[jnn_indices], dim=1) / j
            dscore_mat = torch.sum(knn_scores.unsqueeze(dim=2) * dscore_mat[knn_indices], dim=1) / k
            dscore_mat = torch.softmax(dscore_mat/temperature, dim=-1)

            score_list, pred_list = torch.max(dscore_mat, dim=1)
            tau = torch.sort(score_list, descending=True)[0][int(conf_ratio * score_list.shape[0])]
            confdoc_mask = score_list > tau
        
            hard_pseudolabels = pred_list[confdoc_mask]
            soft_pseudolabels = dscore_mat[confdoc_mask]

        elif rel_type == 'vmfd':
            dscore_mat = torch.matmul(d_emb, c_emb.transpose(1, 0))
            dscore_mat = torch.softmax(dscore_mat/temperature, dim=-1)

            score_list, pred_list = torch.max(dscore_mat, dim=1)
            tau = torch.sort(score_list, descending=True)[0][int(conf_ratio * score_list.shape[0])]
            confdoc_mask = score_list > tau

            hard_pseudolabels = pred_list[confdoc_mask]
            soft_pseudolabels = dscore_mat[confdoc_mask]
            
        else:
            raise ValueError('Invalid relevance score type')

        return confdoc_mask, hard_pseudolabels, soft_pseudolabels


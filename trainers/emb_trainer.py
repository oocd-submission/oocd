import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_reader import Word2vecDataset
from models.textemb import SphericalEmbeddingModel
from sklearn.metrics import roc_auc_score, average_precision_score


class EmbeddingTrainer:
    def __init__(self, data, 
                 emb_dimension=100, batch_size=1024, 
                 n_iters=10, n_negatives=5, emb_lr=0.025, 
                 max_keywords=10, extend_size=1,
                 lambda_local=1.0, lambda_global=1.5, lambda_topic=1.0):
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.data = data
        self.dataset = Word2vecDataset(self.data, n_negatives)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.word_size = len(data.word2id)
        self.document_size = data.sentences_count
        self.category_size = len(data.targets)

        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.n_iters = n_iters
        self.emb_lr = emb_lr
        self.max_keywords = max_keywords
        self.extend_size = extend_size

        self.lambda_local = lambda_local
        self.lambda_global = lambda_global
        self.lambda_topic = lambda_topic

        self.topic_model = SphericalEmbeddingModel(self.emb_dimension, self.word_size, self.document_size, self.category_size)
        self.topic_model.init_keywords([data.word2id[init_keyword] for target, init_keyword in data.targets])
        self.topic_model.load_pretrained_wordemb('./output/pretrained_wiki_w_emb.txt', self.data.word2id)
        self.topic_model.to(self.device)

    def train(self):
        optimizer = optim.Adam(self.topic_model.parameters(), lr=self.emb_lr)
        for iteration in range(self.n_iters):
            total_losses = {'local': [], 'global': [], 'topic': [], 'self': []}
            self.topic_model.update_intermargin()
            self.dataset.pairs = self.data.gatherTrainingPairs()

            for i, sample_batched in enumerate(tqdm(self.dataloader)):
                ptype = sample_batched[0].to(self.device)
                target = sample_batched[1].to(self.device)
                pos = sample_batched[2].to(self.device)
                neg = sample_batched[3].to(self.device)

                if (ptype == 0).sum() > 0:
                    optimizer.zero_grad()
                    loss_local = self.lambda_local * self.topic_model.forward_local(target[ptype==0], pos[ptype==0], neg[ptype==0])
                    loss_local.backward()
                    #self.topic_model.riemannian_gradient(self.topic_model.u_embeddings.weight)
                    #self.topic_model.riemannian_gradient(self.topic_model.v_embeddings.weight)
                    optimizer.step()
                    total_losses['local'].append(float(loss_local.item()))
        
                if (ptype == 1).sum() > 0:
                    optimizer.zero_grad()
                    loss_global = self.lambda_global * self.topic_model.forward_global(target[ptype==1], pos[ptype==1], neg[ptype==1])
                    loss_global.backward()
                    #self.topic_model.riemannian_gradient(self.topic_model.u_embeddings.weight)
                    #self.topic_model.riemannian_gradient(self.topic_model.d_embeddings.weight)
                    optimizer.step()
                    total_losses['global'].append(float(loss_global.item()))
                    
                optimizer.zero_grad()
                loss_topic = self.lambda_topic * self.topic_model.forward_category()
                loss_topic.backward()
                #self.topic_model.riemannian_gradient(self.topic_model.u_embeddings.weight)
                #self.topic_model.riemannian_gradient(self.topic_model.c_embeddings.weight)
                optimizer.step()
                total_losses['topic'].append(float(loss_topic.item()))

            if iteration >= 0 and self.topic_model.keyword_size < self.max_keywords: 
                self.topic_model.extend_category(self.extend_size)

            print("(Iteration %3d) Local Loss: %.4f, Global Loss: %.4f, Topic Loss: %.4f" % (
                        iteration + 1,
                        sum(total_losses['local'])/len(total_losses['local']), 
                        sum(total_losses['global'])/len(total_losses['global']), 
                        sum(total_losses['topic'])/len(total_losses['topic'])))


import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data_reader import DocumentDataset
from models.cnn import ConvClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import save_emb_to_txt, targetf1_score, bestf1_score


class ClassifierTrainer:
    def __init__(self, data, 
                emb_dimension=100, rep_dimension=100, filter_size=20, filter_widths=[3, 4, 5], 
                max_length=1000, drop_prob=0.5, batch_size=32, n_iters=50, 
                initial_plr=0.001, initial_slr=0.0005, valid_ratio=0.1):

        self.use_cuda = torch.cuda.is_available() 
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.n_classes = len(data.targets)
        self.word_size = len(data.word2id)
        self.emb_dimension = emb_dimension
        self.max_length = max_length 
        self.batch_size = batch_size
        self.n_iters = n_iters
        self.initial_plr = initial_plr
        self.initial_slr = initial_slr
        self.valid_ratio = valid_ratio

        self.data = data
        self.test_dataset = DocumentDataset(self.data.docs, self.data.labels)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

        self.classifier = ConvClassifier(self.n_classes, filter_size, filter_widths, drop_prob, self.word_size, emb_dimension, rep_dimension)
        self.classifier.to(self.device)

    def init_trainer(self, topic_model, temperature=0.001, conf_ratio=0.5, k=30, j=30, rel_type='vmfd'):
        mask, hardlabels, softlabels = topic_model.get_confdocs(temperature, conf_ratio, k, j, rel_type)
        mask, hardlabels, softlabels = mask.cpu().numpy(), hardlabels.cpu().numpy(), softlabels.cpu().numpy()
        docs = self.data.docs[mask]
    
        valid_indices = np.random.choice(len(docs), size=int(self.valid_ratio*len(docs)), replace=False)
        train_indices = np.array([idx for idx in range(len(docs)) if idx not in valid_indices])
        train_docs, valid_docs = docs[train_indices], docs[valid_indices]
        train_hardlabels, valid_hardlabels = hardlabels[train_indices], hardlabels[valid_indices]
        train_softlabels, valid_softlabels = softlabels[train_indices], softlabels[valid_indices]

        self.train_dataset = DocumentDataset(train_docs, train_hardlabels, train_softlabels)
        self.valid_dataset = DocumentDataset(valid_docs, valid_hardlabels, valid_softlabels)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        print('\n### The number of confident documents in the training & validation set : ' + str(int(mask.sum())))

    # loss for pre-training
    def compute_ptloss(self, scores, hardlabels, softlabels, hardflag=False):
        if hardflag:
            return self.ce_loss(scores, hardlabels)
        else:
            scores = torch.softmax(scores, dim=1)
            return - (softlabels * torch.log(scores + 1e-12)).sum(dim=1).mean(dim=0)

    # loss for self-training
    def compute_stloss(self, scores):
        scores = torch.softmax(scores, dim=1)
        target = scores.pow(1.2) 
        #target = scores.pow(2) / (scores.sum(dim=0, keepdim=True) + 1e-12)
        target = target / target.sum(dim=1, keepdim=True)
        #return - (target.detach() * torch.log(scores + 1e-12)).sum(dim=1).mean(dim=0)   
        return self.kl_loss(torch.log(scores + 1e-12), target.detach())

    def train(self):
        perf_results = []
        pt_optimizer = optim.Adam(self.classifier.parameters(), lr=self.initial_plr)
        st_optimizer = optim.Adam(self.classifier.parameters(), lr=self.initial_slr)
        
        opt_type = 'pre-train'
        iteration, valid_bestacc, noimprv_cnt = 0, 0, 0
        while True:
            total_loss = []
            self.classifier.train()
            for data, hardlabels, softlabels in self.train_loader:
                data = data.to(self.device)
                hardlabels, softlabels = hardlabels.to(self.device), softlabels.to(self.device)
                scores = self.classifier.compute_logits(data)

                if opt_type == 'pre-train':
                    loss = self.compute_ptloss(scores, hardlabels, softlabels, hardflag=False)
                    pt_optimizer.zero_grad()
                    loss.backward()
                    pt_optimizer.step()
                else: # opt_type == 'self-train'
                    loss = self.compute_stloss(scores)      
                    st_optimizer.zero_grad()
                    loss.backward()
                    st_optimizer.step()
 
                total_loss.append(float(loss.item()))
        
            self.classifier.eval()    
            with torch.no_grad():
                valid_preds, valid_labels = [], []
                test_probs, test_preds, test_labels = [], [], []

                for data, labels, _ in self.valid_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    logits = self.classifier.compute_logits(data)
                    probs = torch.softmax(logits, dim=1)
                    _, preds = torch.max(probs, dim=1)

                    valid_preds.append(preds)
                    valid_labels.append(labels)
                
                valid_preds = torch.cat(valid_preds)
                valid_labels = torch.cat(valid_labels)

                for data, labels in self.test_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    logits = self.classifier.compute_logits(data)
                    probs = torch.softmax(logits, dim=1)
                    if True:
                        _, preds = torch.max(probs, dim=1)
                        probs = torch.sum(probs * torch.log(probs + 1e-12), dim=1)
                    else:
                        probs, preds = torch.max(probs, dim=1)
                    
                    test_probs.append(probs)
                    test_preds.append(preds)
                    test_labels.append(labels)
                    
                test_probs = torch.cat(test_probs)
                test_preds = torch.cat(test_preds)
                test_labels = torch.cat(test_labels)

                train_loss = sum(total_loss)/len(total_loss)
                valid_acc = (valid_preds == valid_labels).sum().item()/(valid_labels != -1).sum().item()
                test_acc = (test_preds == test_labels).sum().item()/(test_labels != -1).sum().item()
                test_auroc = roc_auc_score(test_labels.cpu().numpy() != -1, test_probs.cpu().numpy())
                test_aupr = average_precision_score(test_labels.cpu().numpy() == -1, -test_probs.cpu().numpy())
                test_targetf1 = targetf1_score(test_labels.cpu().numpy() == -1, -test_probs.cpu().numpy())
                test_bestf1 = bestf1_score(test_labels.cpu().numpy() == -1, -test_probs.cpu().numpy())

 
            perf_results.append((valid_acc, test_acc, test_auroc, test_aupr, test_targetf1, test_bestf1))
            print("(%13s %3d) [TRAIN] Loss: %.4f, [VALID] Clf-Acc: %.4f, [TEST] Clf-Acc: %.4f, AUROC: %.4f, AUPR: %.4f, TargetF1: %.4f, BestF1: %.4f" \
                % ('Pre-Training' if opt_type == 'pre-train' else 'Self-Training', iteration + 1, \
                    train_loss, valid_acc, test_acc, test_auroc, test_aupr, test_targetf1, test_bestf1))
            
            if opt_type == 'pre-train':
                iteration += 1
                if noimprv_cnt == 5:
                    opt_type = 'self-train'
                    iteration = 0
                elif valid_acc > valid_bestacc:
                    valid_bestacc = valid_acc
                    noimprv_cnt = 0
                else:
                    noimprv_cnt += 1
            else: # opt_type == 'self-train'
                if iteration + 1 == self.n_iters:
                    break
                else:
                    iteration += 1

        result = perf_results[-1]
        print("\n== PERFORMANCE REPORT ==")
        print("Clf-Acc\t%.4f\tAUROC\t%.4f\tAUPR\t%.4f\tTargetF1\t%.4f\tBestF1\t%.4f" % (result[1], result[2], result[3], result[4], result[5]))

    def save_representation(self, rep_filename, npy_flag=True):
        self.classifier.eval()
        with torch.no_grad():
            total_vecs = []
            for data, _ in self.test_loader:
                data = data.to(self.device)
                dvecs = self.classifier(data)
                total_vecs.append(dvecs)
                    
            total_vecs = torch.cat(total_vecs, dim=0).cpu().numpy()
        
        if npy_flag:
            np.save(rep_filename, total_vecs)
        else:
            d_ids = [str(i) for i in range(len(total_vecs))]
            save_emb_to_txt(rep_filename, d_ids, total_vecs)


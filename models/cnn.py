import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvClassifier(nn.Module):
    def __init__(self, n_classes, filter_size, filter_widths, drop_prob, vocab_size, emb_dimension, rep_dimension):
        super(ConvClassifier, self).__init__()
        
        self.n_classes = n_classes
        self.filter_widths = filter_widths
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension
        self.rep_dimension = rep_dimension       
 
        self.word_embeddings = nn.Embedding(vocab_size, emb_dimension, padding_idx=0)
        torch.nn.init.xavier_uniform_(self.word_embeddings.weight)

        self.conv1 = nn.Conv1d(emb_dimension, filter_size, filter_widths[0])
        self.conv2 = nn.Conv1d(emb_dimension, filter_size, filter_widths[1])
        self.conv3 = nn.Conv1d(emb_dimension, filter_size, filter_widths[2])
        self.dropout = nn.Dropout(drop_prob)

        self.out_linear = nn.Linear(len(filter_widths)*filter_size, rep_dimension)
        self.clf_linear = nn.Linear(rep_dimension, n_classes)
        torch.nn.init.xavier_uniform_(self.out_linear.weight)
        torch.nn.init.xavier_uniform_(self.clf_linear.weight)
    
    def conv_block(self, x, conv_layer):
        out = conv_layer(x)
        out = F.relu(out)
        out = F.max_pool1d(out, out.shape[2]).squeeze(dim=2)
        return out

    def forward(self, x):
        emb = self.word_embeddings(x)
        emb = emb.transpose(1, 2)

        out1 = self.conv_block(emb, self.conv1)
        out2 = self.conv_block(emb, self.conv2)
        out3 = self.conv_block(emb, self.conv3)
        
        out = torch.cat((out1, out2, out3), dim=1)
        out = self.out_linear(self.dropout(out))
        return out

    def compute_logits(self, x):
        out = self.forward(x)
        logits = self.clf_linear(self.dropout(out))
        return logits


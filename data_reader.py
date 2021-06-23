import numpy as np
from torch.utils.data import Dataset

np.random.seed(12345)

class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, inputFileName, targetFileName, labelFileName, min_count, max_length=500, window_size=5):

        self.negatives = []
        self.discards = []
        self.targets = []
        self.labels = []

        self.word2id = dict()
        self.id2word = dict()
        self.target2id = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.max_length = max_length
        self.window_size = window_size

        self.inputFileName = inputFileName
        self.targetFileName = targetFileName
        self.labelFileName = labelFileName

        self.read_words(min_count)
        self.read_targets() 
        self.read_labels()
        
        self.initTableNegatives()
        self.initTableDiscards()

        self.pairs = self.gatherTrainingPairs()
        self.docs = self.gatherTrainingDocs()

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="utf8"):
            line = line.split()
            if len(line) > 0:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        self.word2id['<pad>'] = 0
        self.id2word[0] = '<pad>'
        self.word_frequency[0] = 1

        wid = 1
        init_keywords = [init_keyword for target, init_keyword in self.targets]

        for w, c in word_frequency.items():
            if (w not in init_keywords) and (c < min_count):
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        print("\n### The number of word embeddings: " + str(len(self.word2id)))

    def read_targets(self):
        tid = 0
        for line in open(self.targetFileName, encoding="utf8"):
            targets, init_keyword = line.split()[:-1], line.split()[-1]
            self.targets.append((targets[0].lower(), init_keyword))
            self.target2id.update({target.lower(): tid for target in targets})
            tid += 1

    def read_labels(self):
        for line in open(self.labelFileName, 'r'):
            category_name = line.split()[0].lower()
            self.labels.append(self.target2id.get(category_name, -1))
        self.labels = np.array(self.labels)

    def gatherTrainingDocs(self):
        docs = []
        for line in open(self.inputFileName, encoding="utf8"):
            words = line.split()
            if len(words) > 1:
                doc = [self.word2id[w] for w in words if w in self.word2id] + [0] * self.max_length
                docs.append(doc[:self.max_length])
        return np.array(docs)

    def gatherTrainingPairs(self):
        idx = 0
        total_pairs = []
        for line in open(self.inputFileName, encoding="utf8"):
            words = line.split()
            if len(words) > 1:
                word_ids = [self.word2id[w] for w in words if w in self.word2id
                            and np.random.rand() < self.discards[self.word2id[w]]]

                if len(word_ids) > 1: 
                    boundary = np.random.randint(1, self.window_size)
                    total_pairs.append([[0, u, v] for i, u in enumerate(word_ids) for j, v in
                            enumerate(word_ids[max(i - boundary, 0):i + boundary]) if i != j])

                if len(word_ids) > 0:
                    total_pairs.append([[1, idx, u] for u in word_ids])

                idx += 1
        return np.concatenate(total_pairs, axis=0)

    def initTableDiscards(self):
        t = 0.001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)


class DocumentDataset(Dataset):
    def __init__(self, docs, hardlabels, softlabels=None):
        self.docs = docs
        self.hardlabels = hardlabels
        self.softlabels = softlabels 

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        if self.softlabels is not None:
            return self.docs[idx], self.hardlabels[idx], self.softlabels[idx]
        else:
            return self.docs[idx], self.hardlabels[idx]


class Word2vecDataset(Dataset):
    def __init__(self, data, n_negatives, window_size=5):
        self.pairs = data.pairs
        self.n_negatives = n_negatives
        self.negpos = 0
        self.negatives = data.negatives
        self.window_size = window_size

    def getNegatives(self, target, size): 
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        return pair[0], pair[1], pair[2], self.getNegatives(pair[2], 2)


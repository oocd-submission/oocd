import numpy as np


def load_emb_from_txt(filename):
    emb_vecs = []
    with open(filename, 'r') as f:
        metainfos = f.readline().split()
        emb_size, emb_dim = int(metainfos[0]), int(metainfos[1])
        for line in f:
            fields = line.split()
            emb_vecs.append(np.array([float(x) for x in fields[-emb_dim:]]))
    emb_vecs = np.stack(emb_vecs, axis=0)
    return emb_vecs

def save_emb_to_txt(filename, emb_ids, emb_vecs):
    with open(filename, 'w') as f:
        f.write('%d %d\n' % (emb_vecs.shape[0], emb_vecs.shape[1]))
        for i, e in enumerate(emb_ids):
            v = ' '.join(map(lambda x: str(x), emb_vecs[i]))
            f.write('%s %s\n' % (e, v))

def targetf1_score(labels, scores):
    indices = np.argsort(scores)[::-1]
    sorted_labels, sorted_scores = labels[indices], scores[indices]
    true_indices = np.where(sorted_labels == 1)[0]

    T, P = sum(labels), sum(labels)
    TP = sum(sorted_labels[:T])
    precision = TP / P
    recall = TP / T
    if TP != 0:
        targetf1 = 2 * (precision * recall) / (precision + recall)
    else:
        targetf1 = 0.0
    return targetf1

def bestf1_score(labels, scores):
    indices = np.argsort(scores)[::-1]
    sorted_labels, sorted_scores = labels[indices], scores[indices]
    true_indices = np.where(sorted_labels == 1)[0]

    T, bestf1 = sum(labels), 0.0
    for _TP, _P in enumerate(true_indices):
        TP, P = _TP + 1, _P + 1
        precision = TP / P
        recall = TP / T
        f1 = 2 * (precision * recall) / (precision + recall)
        if f1 > bestf1: bestf1 = f1
    return bestf1


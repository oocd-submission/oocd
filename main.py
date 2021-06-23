import os
import argparse
import numpy as np
import torch

from models.textemb import SphericalEmbeddingModel
from trainers.emb_trainer import EmbeddingTrainer
from trainers.clf_trainer import ClassifierTrainer
from data_reader import DataReader


def main(args):
   
    data_filename = os.path.join("data", args.dataset, args.dataset + "_corpus.txt")
    label_filename = os.path.join("data", args.dataset, args.dataset + "_labels.txt")
    target_filename = os.path.join("data", args.dataset, args.target + "_targets.txt")     

    data = DataReader(inputFileName=data_filename, targetFileName=target_filename, labelFileName=label_filename, 
                            min_count=args.min_count, max_length=args.max_length, window_size=args.window_size)

    # STEP 1
    emb = EmbeddingTrainer(data,
                        emb_dimension=args.emb_dimension, batch_size=args.emb_batch_size, 
                        n_iters=args.emb_iters, n_negatives=args.n_negatives, emb_lr=args.emb_lr, 
                        lambda_local=args.lambda_local, lambda_global=args.lambda_global, lambda_topic=args.lambda_topic)
    print('STEP 1: Embedding-based Confidence Scoring')
    emb.train()

    wemb_filename = os.path.join("output", args.dataset, args.target + "_w_emb.npy")
    demb_filename = os.path.join("output", args.dataset, args.target + "_d_emb.npy")
    cemb_filename = os.path.join("output", args.dataset, args.target + "_c_emb.npy")
    emb.topic_model.save_embeddings(wemb_filename, demb_filename, cemb_filename)

    # STEP 2
    clf = ClassifierTrainer(data, 
                        emb_dimension=args.emb_dimension, rep_dimension=args.rep_dimension, 
                        filter_size=args.filter_size, filter_widths=args.filter_widths, 
                        max_length=args.max_length, drop_prob=args.drop_prob,
                        batch_size=args.clf_batch_size, n_iters=args.clf_iters, 
                        initial_plr=args.clf_plr, initial_slr=args.clf_slr)
    clf.init_trainer(emb.topic_model, args.temperature, args.conf_ratio, args.k, args.j, args.rel_type)
    print('STEP 2: Classifier-based Confidence Scoring')
    clf.train() 


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Invalid boolean value')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
   
    # for OPTIMIZATION 
    parser.add_argument('--gpuidx', default=0, type=int, help='gpu index')
    parser.add_argument('--dataset', default='nyttoy', type=str, help='target dataset')
    parser.add_argument('--target', default='inlier', type=str, help='target category names')
    parser.add_argument('--emb_batch_size', default=1024, type=int, help='batch size for training the embedding')
    parser.add_argument('--emb_iters', default=10, type=int, help='number of iterations')
    parser.add_argument('--emb_lr', default=0.001, type=float, help='initial learning rate') 
    parser.add_argument('--clf_batch_size', default=32, type=int, help='batch size for training the classifier')
    parser.add_argument('--clf_iters', default=50, type=int, help='number of iterations for self-training')
    parser.add_argument('--clf_plr', default=0.001, type=float, help='initial learning rate for pre-training') 
    parser.add_argument('--clf_slr', default=0.00001, type=float, help='initial learning rate for self-training') 

    # for CONFDOC_IDENTIFIER    
    parser.add_argument('--temperature', default=0.1, type=float, help='temperature for pseudo-labeling') 
    parser.add_argument('--conf_ratio', default=0.8, type=float, help='ratio of confident documents to be retrieved') 
    parser.add_argument('--k', type=int, default=30, help='number of neighbor documents')
    parser.add_argument('--j', type=int, default=30, help='number of neighbor words')
    parser.add_argument('--rel_type', type=str, default='vmfd', help='relevance score type')

    # for EMB_TRAINER    
    parser.add_argument('--emb_dimension', default=100, type=int, help='dimensionality of embeddings')
    parser.add_argument('--min_count', default=5, type=int, help='minimum count of words')
    parser.add_argument('--window_size', default=5, type=int, help='window size for skip-gram')
    parser.add_argument('--n_negatives', default=2, type=int, help='number of negative words')

    parser.add_argument('--lambda_local', default=1.0, type=float, help='weight for local loss') 
    parser.add_argument('--lambda_global', default=1.5, type=float, help='weight for global loss') 
    parser.add_argument('--lambda_topic', default=1.0, type=float, help='weight for topic loss') 

    # for CLF_TRAINER
    parser.add_argument('--max_length', default=500, type=int, help='maximum length of documents')
    parser.add_argument('--filter_size', default=32, type=int, help='size of conv filters')
    parser.add_argument('--filter_widths', default=[2, 3, 4], nargs='+', type=int, help='widths of conv filters')
    parser.add_argument('--drop_prob', default=0.5, type=float, help='dropout probability')
    parser.add_argument('--rep_dimension', default=100, type=int, help='dimensionality of representations')
    
    args = parser.parse_args()

    print(args)

    # GPU setting
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuidx)
    
    # Random seed initialization
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    main(args=args)

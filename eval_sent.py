from __future__ import absolute_import, division, unicode_literals

import logging
import sys

import torch
import os
import pickle

import data_utils
from model import BaselineNet

# Set PATHs
PATH_TO_SENTEVAL = 'SentEval/'
PATH_TO_DATA = 'SentEval/data/senteval_data/'

# TODO: command line input
embedding_path = os.path.join('data','glove','glove.filtered.300d.txt') #TODO: regenerate based on task and full GloVe
model_name = 'baseline'
checkpoint_path = os.path.join('output','baseline','23065025','checkpoints','model_iter_40000.pt')
output_dir = os.path.join('output','baseline','22221624_test')

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device_name = 'cpu'
device = torch.device(device_name)
print("Device: "+device_name)

# import Senteval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def prepare(params, samples):
    '''
    TODO: Prepare the vocabulary and embedding
    '''
    
    return

def batcher(params, batch):
    # Retrieve parameters
    net = params['model']
    dataloader = params['dataloader']
    
    # Encode batch through model
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    batch = dataloader.prepare_sentences(batch)
    embeddings = torch.Tensor(net.encode(batch)).to(device)

    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 10, 'batch_size': 512}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Obtain GloVe word embeddings
    print("Loading GloVe embedding from "+embedding_path)
    glove_emb = data_utils.EmbeddingGlove(embedding_path)

    # Build vocabulary
    vocab = data_utils.Vocabulary()
    vocab.count_glove(glove_emb)
    vocab.build()

    # Empty dataloader for converting sentence list to batch
    dataloader = data_utils.DataLoaderSnli([], vocab)
    
    # Load network
    if model_name == 'baseline':
        net = BaselineNet(glove_emb.embedding).to(device)
    # Load checkpoint
    print("Initialising model from "+checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state_dict)
    print("Network architecture:\n\t{}".format(str(net)))

    params_senteval['dataloader'] = dataloader
    params_senteval['model']      = net

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = []
    for task in transfer_tasks:
        try:
            r = se.eval(transfer_tasks)
            results.append(r)
        except Exception as e:
            print("Failed to perform task '{}':\n{}".format(task,str(e)))
    pickle.dump(results, open(os.path.join(output_dir,'SentEval_results.p'), "wb"))
    print(results)

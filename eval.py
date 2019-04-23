import argparse
import numpy as np
import os
import torch.nn.functional as F
import torch.optim as optim
import pickle
import datetime
import matplotlib
import matplotlib.pyplot as plt
import torch
import types
import data_utils
from tensorboardX import SummaryWriter
from subprocess import call

from model import BaselineNet
import data_utils
import visual


# Default constants
MODEL_NAMES = ['baseline']
MODEL_NAME_DEFAULT = 'baseline'
CHECKPOINT_PATH_DEFAULT = None
OUTPUT_DIR_DEFAULT = 'evaluation'
DATA_PATH_DEFAULT = os.path.join('data','snli_1.0','snli_1.0_test.txt')
EMBEDDING_PATH_DEFAULT = os.path.join('data','glove','glove.filtered.300d.txt')

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the micro and macro prediction accuracy

    Args:
        predictions: 2D float array of size [batch_size, n_classes]
        labels: 1D int array of size [batch_size]
                with ground truth labels for each sample in the batch
    Returns:
        accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    """

    class_count = []
    class_acc = []

    pred = predictions.argmax(dim=1)

    for t in set(targets.tolist()):
        pred_sub  = pred[targets==t]
        target_sub = targets[targets==t]
        class_count.append(len(target_sub))
        correct_pred = pred_sub - target_sub == 0
        acc = correct_pred.sum().float() / float(len(correct_pred))
        class_acc.append(acc)
    class_count = np.array(class_count)
    class_acc = np.array(class_acc)

    # classes have equal weight
    accuracy_macro = np.mean(class_acc)
    # classes are weighted by their count (normal cccuracy)
    accuracy_micro = class_count @ class_acc / class_count.sum()

    return accuracy_macro, accuracy_micro

def eval():
    '''
    Performs test evaluation on the model.
    '''
    ## Read terminal arguments
    model_name      = FLAGS.model_name
    checkpoint_path = FLAGS.checkpoint_path
    output_dir      = FLAGS.output_dir
    data_path       = FLAGS.data_path
    embedding_path  = FLAGS.embedding_path

    assert checkpoint_path is not None,     "checkpoint_path is a required argument"
    assert os.path.isfile(checkpoint_path), "Checkpoint does not exist"
    assert os.path.isfile(embedding_path),  "Embedding does not exist"
    assert model_name in MODEL_NAMES,       "Model name is unknown"

    # Further process terminal arguments
    os.makedirs(output_dir, exist_ok=True) # create output directory

    # Obtain GloVe word embeddings
    print("Loading GloVe embedding from "+embedding_path)
    glove_emb = data_utils.EmbeddingGlove(embedding_path)

    # Build vocabulary
    vocab = data_utils.Vocabulary()
    vocab.count_glove(glove_emb)
    vocab.build()

    # Obtain SNLI train and dev dataset
    dataset = {}
    dataloader = {}
    dataset    = data_utils.DatasetSnli(data_path)
    dataloader = data_utils.DataLoaderSnli(dataset, vocab)

    # Load network
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print("Device: "+device_name)
    if model_name == 'baseline':
        net = BaselineNet(glove_emb.embedding).to(device)
    # Load checkpoint
    print("Initialising model from "+checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(state_dict)
    print("Network architecture:\n\t{}".format(str(net)))

    # Evaluate SNLI per class
    prem, hyp, label = dataloader.next_batch(len(dataset))
    prem = prem.to(device)
    hyp  = hyp.to(device)
    label = label.to(device)

    prediction = net.forward(prem, hyp)
    accuracy_macro, accuracy_micro = accuracy(prediction, label)

    print("Macro accuracy:\t{}\nMicro accuracy:\t{}".format(accuracy_macro,accuracy_micro))
            
    

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))
    print()


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    # Run the training operation
    eval()


FLAGS = types.SimpleNamespace()
if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = MODEL_NAME_DEFAULT,
                        help='Name of the model type to train (baseline, )')
    parser.add_argument('--checkpoint_path', type = str, default = CHECKPOINT_PATH_DEFAULT,
                        help='Path to a model checkpoint')
    parser.add_argument('--output_dir', type = str, default = OUTPUT_DIR_DEFAULT,
                        help='Directory to write output to')
    parser.add_argument('--data_path', type=str, default=DATA_PATH_DEFAULT,
                        help='Path to the SNLI test dataset')
    parser.add_argument('--embedding_path', type=str, default=EMBEDDING_PATH_DEFAULT,
                        help='Path to word embedding file')
    FLAGS, unparsed = parser.parse_known_args()

    main()

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
OUTPUT_DIR_DEFAULT = 'output'
MODEL_NAME_DEFAULT = 'baseline'
MODEL_NAMES = ['baseline']
CHECKPOINT_PATH_DEFAULT = None
DATA_TRAIN_PATH_DEFAULT = os.path.join('data','snli_1.0','snli_1.0_train.txt')
DATA_DEV_PATH_DEFAULT = os.path.join('data','snli_1.0','snli_1.0_dev.txt')
EMBEDDING_PATH_DEFAULT = os.path.join('data','glove','glove.filtered.300d.txt')
BATCH_SIZE_DEFAULT = 64
MAX_STEPS_DEFAULT = None
LEARNING_RATE_DEFAULT = 0.1
ACTIVATE_BOOL_DEFAULT = False

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
        predictions: 2D float array of size [batch_size, n_classes]
        labels: 1D int array of size [batch_size]
                with ground truth labels for each sample in the batch
    Returns:
        accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    """

    correct_prediction = predictions.argmax(dim=1) - targets == 0
    accuracy = correct_prediction.sum().float() / float(len(correct_prediction))

    return accuracy

def train():
    '''
    Performs training and evaluation of the model.
    '''
    start_time = datetime.datetime.now()

    ## Read terminal arguments
    model_name      = FLAGS.model_name
    activate_board  = FLAGS.activate_board
    checkpoint_path = FLAGS.checkpoint_path
    data_train_path = FLAGS.data_train_path
    data_dev_path   = FLAGS.data_dev_path
    embedding_path  = FLAGS.embedding_path
    # Hyperparameters
    batch_size      = FLAGS.batch_size
    max_steps       = FLAGS.max_steps
    learning_rate   = FLAGS.learning_rate
    o_dir           = FLAGS.output_dir

    # Further process terminal arguments
    if model_name not in MODEL_NAMES:
        raise NotImplementedError
    now = datetime.datetime.now()
    time_stamp = "{:02g}{:02g}{:02g}{:02g}".format(now.day, now.hour, now.minute, now.second)
    output_dir      = os.path.join(o_dir,model_name,time_stamp)
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    checkpoint_dir  = os.path.join(output_dir, 'checkpoints')
    os.makedirs(tensorboard_dir, exist_ok=True) # create output and tensorboard directory
    os.makedirs(checkpoint_dir, exist_ok=True)  # create checkpoint directory

    if checkpoint_path is not None:
        if not os.path.isfile(checkpoint_path):
            print("Could not find checkpoint: "+checkpoint_path)
            return
    
    # Standard hyperparams
    weight_decay = .99
    eval_freq    = 100
    check_freq   = 1000

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
    for set_name,set_path in [('train',data_train_path),('dev',data_dev_path)]:
        dataset[set_name] = data_utils.DatasetSnli(set_path)
        dataloader[set_name] = data_utils.DataLoaderSnli(dataset[set_name], vocab)

    # Initialise network
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    print("Device: "+device_name)
    if model_name == 'baseline':
        net = BaselineNet(glove_emb.embedding).to(device)
    # Load checkpoint
    if checkpoint_path is not None:
        print("Initialising model from "+checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location=device)
        net.load_state_dict(state_dict)
    loss_fn = F.cross_entropy
    print("Network architecture:\n\t{}\nLoss module:\n\t{}".format(str(net), str(loss_fn)))
    
    # Evaluation vars
    writer = SummaryWriter(log_dir=tensorboard_dir)
    if activate_board:
        call('gnome-terminal -- tensorboard --logdir '+tensorboard_dir, shell=True) # start tensorboard
    train_loss = []
    gradient_norms = []
    train_acc = []
    test_acc = []
    iteration = 0

    # Training
    optimizer = optim.SGD(net.sequential.parameters(), 
        lr=learning_rate, weight_decay=weight_decay)
    
    last_dev_acc = 0
    current_dev_accs = []
    epoch = 0
    while True:
        # Stopping criterion
        iteration += 1
        # Max iterations
        if max_steps is not None:
            if iteration > max_steps:
                print("Training stopped: maximum number of iterations reached")
                break
        # Adapt learning rate; early stopping
        if dataloader['train']._epochs_completed > epoch:
            epoch = dataloader['train']._epochs_completed
            print("Epoch {}".format(epoch))
            if current_dev_accs == []:
                current_dev_accs = [0]
            current_dev_acc = np.mean(current_dev_accs)
            if current_dev_acc < last_dev_acc:
                learning_rate /= 5
                if learning_rate < 1e-5:
                    print("Training stopped: learning rate dropped below 1e-5")
                    break
                for g in optimizer.param_groups:
                    g['lr'] = learning_rate
                print("Learning rate dropped to {}".format(learning_rate))
                writer.add_scalar('learning_rate', learning_rate, iteration)
            writer.add_scalar('epoch_dev_acc', current_dev_acc, epoch)
            last_dev_acc = current_dev_acc
            current_dev_accs = []

        # Sample a mini-batch
        prem, hyp, label = dataloader['train'].next_batch(batch_size)
        prem = prem.to(device)
        hyp  = hyp.to(device)
        label = label.to(device)

        # Forward propagation
        prediction = net.forward(prem, hyp)
        loss = loss_fn(prediction, label)
        acc = accuracy(prediction, label)
        train_acc.append( (iteration, acc.tolist()) )
        train_loss.append( (iteration, loss.tolist()) )
        writer.add_scalars('accuracy', {'train': train_acc[-1][1]}, iteration)
        writer.add_scalar('train_loss', train_loss[-1][1], iteration)

        # Backprop
        optimizer.zero_grad()
        loss.backward()

        # Weight update in linear modules
        optimizer.step()

        with torch.no_grad():
            norm = 0
            for params in net.sequential.parameters():
                norm += params.grad.reshape(-1).pow(2).sum()
            gradient_norms.append( (iteration, norm.reshape(-1).tolist()[0]) )
            writer.add_scalar('gradient_norm', gradient_norms[-1][1], iteration)

            # Evaluation
            if iteration % eval_freq == 0 or iteration == max_steps:
                prem, hyp, label = dataloader['dev'].next_batch(batch_size)
                prem = prem.to(device)
                hyp  = hyp.to(device)
                label = label.to(device)
                prediction = net.forward(prem,hyp)
                acc = accuracy(prediction, label)
                test_acc.append( (iteration, acc.tolist()) )
                writer.add_scalars('accuracy', {'dev': test_acc[-1][1]}, iteration)
                print("Iteration: {}\t\tTest accuracy: {}\t\tTrain accuracy: {}".format(iteration, acc, train_acc[-1][1]))
            
            # Checkpoint
            if iteration % check_freq == 0 or iteration == max_steps:
                print("Saving checkpoint")
                torch.save(net.state_dict(), os.path.join(checkpoint_dir, "model_iter_"+str(iteration)+".pt"))
                # Save or return raw output
                metrics = {"train_loss": train_loss,
                            "gradient_norms": gradient_norms,
                            "train_acc": train_acc,
                            "test_acc": test_acc}
                # Save
                pickle.dump(metrics, open(os.path.join(output_dir, "metrics.p"), "wb"))
                visual.make_plots(output_dir, metrics)
    
    writer.close()

    end_time = datetime.datetime.now()
    print("Done. Start and End time:\n\t{}\n\t{}".format(start_time, end_time))

    return metrics



def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Print all Flags to confirm parameter settings
    print_flags()

    # Run the training operation
    train()


FLAGS = types.SimpleNamespace()
if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default = MODEL_NAME_DEFAULT,
                        help='Name of the model type to train (baseline, )')
    parser.add_argument('--checkpoint_path', type = str, default = CHECKPOINT_PATH_DEFAULT,
                        help='Path to a checkpoint file')
    parser.add_argument('--output_dir', type = str, default = OUTPUT_DIR_DEFAULT,
                        help='Directory to write output directory to')
    parser.add_argument('--data_train_path', type=str, default=DATA_TRAIN_PATH_DEFAULT,
                        help='Path to the SNLI train dataset')
    parser.add_argument('--data_dev_path', type=str, default=DATA_DEV_PATH_DEFAULT,
                        help='Path to the SNLI development dataset')
    parser.add_argument('--embedding_path', type=str, default=EMBEDDING_PATH_DEFAULT,
                        help='Path to word embedding file')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Mini-batch size during training')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Maximum number of training iterations')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate for training')
    parser.add_argument('--activate_board', type=bool, default=ACTIVATE_BOOL_DEFAULT,
                        help='Automatically activate tensorboard')
    FLAGS, unparsed = parser.parse_known_args()

    main()

from matplotlib import pyplot as plt
import os
import numpy as np

def partition(l, n):
        '''
        Partitions a list into n parts
        '''
        if len(l) <= n:
                return [[e] for e in l]

        p = []
        indices = np.linspace(0,len(l),n+1, dtype=int)
        for i in range(1,len(indices)):
                p.append(l[indices[i-1]:indices[i]])
        return p

def make_plots(output_dir, metrics):
        # Obtain metrics
        train_loss     = metrics['train_loss']
        gradient_norms = metrics['gradient_norms']
        train_acc      = metrics['train_acc']
        test_acc       = metrics['test_acc']

        n_part = 200 # number of points in the averaged graphs

        # Save plots
        # Loss
        iter = [i for (i,q) in train_loss]
        loss = [q for (i,q) in train_loss]
        fig, ax = plt.subplots()
        ax.plot(iter, loss)
        ax.set(xlabel='Iteration', ylabel='Loss',title='Batch training loss')
        ax.grid()
        fig.savefig(os.path.join(output_dir, "loss.png"))

        iter = [it[0] for it in partition(iter, n_part)]
        loss = [np.mean(l) for l in partition(loss, n_part)]
        fig, ax = plt.subplots()
        ax.plot(iter, loss)
        ax.set(xlabel='Iteration', ylabel='Average loss',title='Averaged training loss')
        ax.grid()
        fig.savefig(os.path.join(output_dir, "loss_avg.png"))

        # gradient norm
        iter = [i for (i,q) in gradient_norms]
        norm = [q for (i,q) in gradient_norms]
        fig, ax = plt.subplots()
        ax.plot(iter, norm)
        ax.set(xlabel='Iteration', ylabel='Norm',title='Gradient norm')
        ax.grid()
        fig.savefig(os.path.join(output_dir, "gradient_norm.png"))

        iter = [it[0] for it in partition(iter, n_part)]
        norm = [np.mean(n) for n in partition(norm, n_part)]
        fig, ax = plt.subplots()
        ax.plot(iter, norm)
        ax.set(xlabel='Iteration', ylabel='Average norm',title='Averaged gradient norm')
        ax.grid()
        fig.savefig(os.path.join(output_dir, "gradient_norm_avg.png"))

        # accuracies
        iter_train = [i for (i,q) in train_acc]
        accu_train = [q for (i,q) in train_acc]
        iter_dev = [i for (i,q) in test_acc]
        accu_dev = [q for (i,q) in test_acc]
        fig, ax = plt.subplots()
        ax.plot(iter_train, accu_train, label='Train')
        ax.plot(iter_dev, accu_dev, label='Dev')
        ax.set(xlabel='Iteration', ylabel='Accuracy',title='Train and dev accuracy')
        ax.legend()
        ax.grid()
        fig.savefig(os.path.join(output_dir, "accuracy.png"))

        iter_train = [it[0] for it in partition(iter_train, n_part)]
        accu_train = [np.mean(a) for a in partition(accu_train, n_part)]
        iter_dev = [it[0] for it in partition(iter_dev, n_part)]
        accu_dev = [np.mean(n) for n in partition(accu_dev, n_part)]
        fig, ax = plt.subplots()
        ax.plot(iter_train, accu_train, label='Train')
        ax.plot(iter_dev, accu_dev, label='Dev')
        ax.set(xlabel='Iteration', ylabel='Average accuracy',title='Averaged train and dev accuracy')
        ax.legend()
        ax.grid()
        fig.savefig(os.path.join(output_dir, "accuracy_avg.png"))

        plt.close(fig)
        
import os

import data_utils


'''
Data is prepared for the actual experiments:
- Filtered GloVe file is made with only words from SNLI
- Very small subset of SNLI for sanity check
'''

def filter_glove():
    '''
    Filters the large glove file. Only takes the words that occur in the SNLI dataset and writes them to a new glove file.
    '''
    # load SNLI datasets and count its words in the vocabulary
    vocab = data_utils.Vocabulary()
    for setname in ['train','dev','test']:
        print("Loading {} dataset".format(setname))
        data_path = os.path.join('data','snli_1.0','snli_1.0_'+setname+'.txt')
        dataset = data_utils.DatasetSnli(data_path)
        print("Counting words")
        vocab.count_snli(dataset)

    # build the vocabulary
    print("Building vocabulary")
    vocab.build()

    # filtering GloVe file
    print("Filtering GloVe file")
    file_name_from = os.path.join('data','glove','glove.840B.300d.txt')
    file_name_to   = os.path.join('data','glove','glove.filtered.300d.txt')
    file_to = open(file_name_to, "w")
    countTot = 0
    countAdd = 0
    with open(file_name_from,"r") as file_from:
        for line in file_from:
            word = line.split()[0]
            # Add line to filtered file if the word is in the vocab
            if word in vocab.w2i.keys():
                try:
                    emb = [float(x) for x in line.split()[1:]]
                    if len(emb) == 300:
                        file_to.write(line)
                        countAdd += 1
                    else:
                        print('Unexpected line length (not 300): "{}"'.format(line))
                except:
                    print('Unexpected line (multiple words?): "{} ..."'.format(line[:20]))
            countTot += 1
            if countTot % 25000 == 0:
                print("{} words processed ({} selected)".format(countTot, countAdd))
        print("{} words processed ({} selected)".format(countTot, countAdd))
    file_to.close()


def snli_subset():
    '''
    Makes two very small subsets of the SNLI data, of different sizes
    Size of train, dev and test set are the same
    '''
    # Make directories
    dir_name  = lambda size : os.path.join('data','snli_sub'+str(size))
    sizes = [1, 8, 16, 64, 256]
    for size in sizes:
        os.makedirs(dir_name(size), exist_ok=True)

    for setname in ['train','dev','test']:
        print("Loading {} dataset".format(setname))
        file_name = 'snli_1.0_'+setname+'.txt'
        data_path = os.path.join('data','snli_1.0',file_name)
        with open(data_path, 'r') as data_file:
            header = data_file.readline()
            # write to subset data files
            for size in sizes:
                with open(os.path.join(dir_name(size), file_name), 'w') as subset_file:
                    subset_file.write(header)
                    while size > 0:
                        line = data_file.readline()
                        if line[0] != "-": # ambiguous
                            size -= 1
                            subset_file.write(line)

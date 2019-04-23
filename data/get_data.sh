#!/bin/bash

# GloVe
echo "Getting GloVe"
wget -O glove.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.zip -d glove/
rm glove.zip

# SNLI
echo "Getting SNLI"
wget -O snli.zip https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli.zip
rm snli.zip

# Data preparations
echo "Preparing data
cd ..
python prepare.py


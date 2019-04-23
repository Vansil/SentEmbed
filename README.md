# SentEmbed
Practical for Statistical Semantics for Natural Language Semantics (UvA)

Trained models can be found in the directory 'output'. The model in 'output/baseline/experiment_23075505' is the best performing baseline model in my experiment. The graphs in the report correspond to this model. As an LSTM model, 'unilstm/noeffect_23115828' was investigated. This was without success, as the name suggests.

To apply a model by interactive inference in the terminal:

    python infer.py --checkpoint_path=output/path/to/checkpoint
    
To evaluate a model on the SNLI test set:

    python eval.py --checkpoint_path=output/path/to/checkpoint
    
To train a model (e.g. baseline) with standard parameters and track progress in tensorboard:

    python train.py --model_name=baseline --activate_board=True
    
Especially in the last function, many more options can be set from the terminal. To list them, use ```--help```.

The required data (SNLI, SNLI subsets, GloVe embeddings, GloVe filtered on SNLI) can be downloaded and prepared by calling ```data/get_data.sh```.

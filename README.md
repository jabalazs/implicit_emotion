# [Implicit Emotion Shared Task](http://implicitemotions.wassa2018.com/)

# Requirements
* Python 3
* Pytorch 0.4.0

# Installation
## Code

* Clone this repo.
```
git clone --recurse-submodules https://github.com/jabalazs/implicit_emotion.git
cd implicit_emotion
```

**We recommend doing all the steps below within a conda environment**

* Make sure you already have python 3 and pytorch installed

* Install the other dependencies
```
pip install -r requirements.txt
```

* Install Matplotlib and Scikit learn
```
conda install matplotlib scikit-learn
```
## Data
* Get the data by running the following command and typing your password when prompted
```
./get_data.sh <USERNAME>
```

* Run the preprocessing script
```
./preprocess.sh
```

## Pre-trained Embeddings
The code expects a directory named `word_embeddings` in `data`
containing embeddings in `.txt` format with their default name,
e.g., `glove.840B.300d.txt`. See [`config.py`](src/config.py) for more details
on naming conventions and directory structure.

TODO: create script to automate this process.

---
To test if you installed everything correctly run `python run.py --help`.

# Preliminary results

```
date                 hash        commit      server_name  seed        emb                corpus      model       char_emb_dim  wem         wcam        batch       optim       lr_0        ulr         acc         best_epoch
-------------------  ----------  ----------  -----------  ----------  -----------------  ----------  ----------  ------------  ----------  ----------  ----------  ----------  ----------  ----------  ----------  ----------
2018-06-07 10:07:12  a3d35de     9953057     yatima       43          glove              iest        bilstm      100           char_lstm   cat         64          adam        0.001       1           0.6235      1
2018-06-07 08:28:48  d51d588     9953057     yatima       43          glove              iest        bilstm      100           char_lstm   scalar_gat  64          adam        0.001       1           0.6231      1
2018-06-07 08:45:36  a13a0ba     9953057     yatima       44          glove              iest        bilstm      100           char_lstm   scalar_gat  64          adam        0.001       1           0.6224      1
2018-06-06 14:15:12  7f6cdc0     9953057     yatima       42          glove              iest        bilstm      100           char_lstm   scalar_gat  64          adam        0.001       1           0.6203      1
2018-06-06 11:55:54  bc44c5a     9953057     yatima       43          glove              iest        bilstm      100           char_lstm   vector_gat  64          adam        0.001       1           0.6187      1
2018-06-06 08:07:34  9dec60f     5a4a315     yatima       42          glove              iest        bilstm      100           char_lstm   vector_gat  64          adam        0.001       1           0.6186      1
```

## Best Performance
Currently the best validation accuracy (0.6235) is obtained at the end of the second epoch by running:
```bash
python run.py -lr=0.001 --lstm_hidden_size=1024 --word_encoding_method=char_lstm --word_char_aggregation_method=cat --char_emb_dim=100 --update_learning_rate --seed=43
```

# TODO

## Data
* Try augmenting the training dataset, if that's allowed
* Try removing examples which have the `un[#TRIGGERWORD#]` variant, or conditioning on it. See [this](https://groups.google.com/forum/#!topic/implicit-emotions-shared-task-wassa-2018/2wIdY_lmCoY) thread.
* Related to the previous point, in some examples the trigger word is a hashtag: `#[#TRIGGERWORD#]`. Maybe we could use this as a feature.
* Tokenize and exploit emojis. See [#1](https://github.com/jabalazs/implicit_emotion/issues/1).

## Model tuning

* Try using other pre-trained word embeddings
* ~Shuffle training examples at each epoch~
* Try using attention for aggregating character-level representations into word representations
* Try using learning rate decay
* Try different regularization methods
  - l2-norm (see `weight_decay` [here](https://pytorch.org/docs/stable/optim.html))
  - dropout
  - [batch norm](https://pytorch.org/docs/stable/nn.html?highlight=crossentropy#batchnorm1d) 
 
## Engineering
### Implement proper way of doing early stopping
Now it's only being applied when passing the `--update_learning_rate` and `--learning_rate_decay=<number>` flags, and the learning rate goes below 1e-5 (as in [InferSent](http://www.aclweb.org/anthology/D17-1070))

The idea would be to control the stopping policies from a centralized place, so we could for example tell the training procedure to stop if any of the following conditions are met:

* Epoch greater than 10
* Learning rate lower than 1e-5
* Validation accuracy has not improved in the last 2 epochs

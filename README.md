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
date                 hash        commit      seed        corpus      char_emb_dim  sent_l      wcam        ulr         acc         best_epoch
-------------------  ----------  ----------  ----------  ----------  ------------  ----------  ----------  ----------  ----------  ----------
2018-06-08 11:23:26  49736fd     36ee296     42          iest_emoji  150           1           cat         1           0.6287      1
2018-06-08 08:38:15  aed323b     ccc3e4d     44          iest        150           1           vector_gat  1           0.6259      1
2018-06-08 10:24:28  40a35e4     36ee296     44          iest_emoji  150           1           vector_gat  1           0.625       1
2018-06-07 16:35:42  69fd297     ccc3e4d     42          iest        150           1           cat         1           0.6239      1
2018-06-07 15:47:05  8a44651     ccc3e4d     42          iest        150           1           scalar_gat  1           0.6236      1
2018-06-07 10:07:12  a3d35de     9953057     43          iest        100           1           cat         1           0.6235      1
```

## Best Performance
Currently the best validation accuracy (0.6287) is obtained at the end of the second epoch by running:
```bash
python run.py --corpus=iest_emoji -lr=0.001 --lstm_hidden_size=1024 --word_encoding_method=char_lstm --word_char_aggregation_method=cat --char_emb_dim=150 --update_learning_rate --seed=42
```

# TODO

## Data
* Try augmenting the training dataset, if that's allowed
* Try removing examples which have the `un[#TRIGGERWORD#]` variant, or conditioning on it. See [this](https://groups.google.com/forum/#!topic/implicit-emotions-shared-task-wassa-2018/2wIdY_lmCoY) thread.
* Related to the previous point, in some examples the trigger word is a hashtag: `#[#TRIGGERWORD#]`. Maybe we could use this as a feature.
* ~Tokenize and exploit emojis. See [#1](https://github.com/jabalazs/implicit_emotion/issues/1).~

## Model tuning

* ~Try using other pre-trained word embeddings~ `glove.840B.300d` seem to perform best
* ~Shuffle training examples at each epoch~
* Try using attention for aggregating character-level representations into word representations
* Try different learning rate decays schedules
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

### Show models that have a saved checkpoint in database
Right now I have to look at the experiment hash and navigate to the corresponding directory to see whether it was saved during training or not.

An option could be to save every model, but given that they are around 350MB in size this might not be practical.

Another solution would be to perform a check somewhere at some point to update this field for every row. However this might create a great overhead in projects with lots of experiments.

Another option, and the simplest one, is to create the database entry indicating the experiment doesn't have a corresponding checkpoint, and then change this field when the model is saved. The problem with this approach is that if the model is deleted manually the database will keep showing as if there was one.

# [Implicit Emotion Shared Task](http://implicitemotions.wassa2018.com/)

[Competition website](https://competitions.codalab.org/competitions/19214)

# Installation

1. Clone this repo.
   ```
   git clone --recurse-submodules https://github.com/jabalazs/implicit_emotion.git
   cd implicit_emotion
   ```

2. Create a conda environment
   
   If you don't have conda installed, we recommend using [miniconda](https://conda.io/miniconda.html).
   
   You can then easily create and activate a new conda environment with Python 3.6
   by executing 
   
   ```
   conda create -n iest python=3.6
   source activate iest
   ```
   
   Where you can replace `iest` by any environment name you like.

3. Run `./install.sh`

   This will install pytorch, a few dependencies for our code, AllenNLP (ELMo) and
   all of its dependencies. See https://github.com/allenai/allennlp for more ways
   to install AllenNLP. Also note that for replicability purposes we will install
   the same version we used for development: [`ac2e0b9b6`](https://github.com/allenai/allennlp/tree/ac2e0b9b6e4668984ebd8c05578d9f4894e94bee).

   > By default, AllenNLP will be cloned in this repo. If you want to install it
     somewhere else please modify the install script
     [install.sh](install.sh), and change the `ALLENNLP_PATH`
     variable in [src/config.py](src/config.py) accordingly.

4. (Optional) Install java

   We used a forked version of [`ark-tweet-nlp`](https://github.com/jabalazs/ark-tweet-nlp/tree/7e37f5badcc28d1b5ad595d26721db3832fd1dde)
   for obtaining POS tags without using its built-in tokenization feature. This
   repo already comes with the resulting jar (`ark-tweet-nlp-0.3.2.jar`) in
   [`utils/ark-tweet-nlp`](utils/ark-tweet-nlp).
   
   If you want to use this feature you need java. You can easily install it
   within your conda environment with
   ```
   conda install -c cyclus java-jdk
   ```
   
   > You can also change the pre-trained POS tagging model by modifying the
     `PRETRAINED_MODEL_NAME` variable in [`utils/run_postagger.sh`](utils/run_postagger.sh)
     with one of the models provided in [`utils/ark-tweet-nlp`](utils/ark-tweet-nlp).

## Data
* Get the data by running the following command and typing your password when prompted
```
./get_data.sh <USERNAME>
```

* Run the preprocessing script
```
./preprocess.sh
```


## External Data
TODO: create script to automate this process.

### Pre-trained Embeddings
The code expects a directory named `word_embeddings` in `data`
containing embeddings in `.txt` format with their default name,
e.g., `glove.840B.300d.txt`. See [`config.py`](src/config.py) for more details
on naming conventions and directory structure.

### Elmo options and weights
To use Elmo we need 2 files provided by their developers. Run the following commands within the project dir, assuming you already created the `word_embeddings` dir:
```
cd data/word_embeddings
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json
wget https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
```

See [this tutorial](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md) for more details

---
To test if you installed everything correctly run `python run.py --help`.

# Preliminary results

```
date                 hash        commit      seed        corpus      char_emb_dim  lstm_out_dim  dpout       sent_l      wem         wcam        ulr         acc         best_epoch
-------------------  ----------  ----------  ----------  ----------  ------------  ------------  ----------  ----------  ----------  ----------  ----------  ----------  ----------
2018-06-20 17:32:01  5047214     6eb5bac     42          iest_emoji  200           2048          0.1         1           elmo                    1           0.6745      6
2018-06-20 13:04:45  73e08f6     6eb5bac     43          iest_emoji  200           2048          0.1         1           elmo                    1           0.6677      6
2018-06-20 10:40:43  75c72f2     6eb5bac     444         iest_emoji  200           2048          0.1         1           elmo                    1           0.6653      5
2018-06-15 07:57:18  a37d98a     ea5e5c3     444         iest_emoji  200           2048          0.1         1           elmo                    1           0.6548      4
2018-06-13 12:20:02  375a9ba     c4b07c8     43          iest_emoji  200           2048          0.1         1           elmo                    1           0.6543      6
2018-06-13 10:41:58  edc78f7     f62ac12     43          iest_emoji  200           1024          0.1         1           elmo                    1           0.6524      6
2018-06-18 11:21:23  03bf597     ea5e5c3     46          iest_emoji  200           2048          0.1         1           elmo                    1           0.6519      8
```

## Best Performance
Currently the best validation accuracy (0.6745) is obtained at the end of the 5th epoch by running:
```bash
python run.py --corpus=iest_emoji -lr=0.001 --lstm_hidden_size=2048 --word_encoding_method=elmo --update_learning_rate --seed=42 -cem=200
```
**Note**: The big bump in performance from 0.6548 up was due to the organizers fixing mislabeled examples in the dataset.

# TODO

## Data
* Try augmenting the training dataset, if that's allowed
* ~Try removing examples which have the `un[#TRIGGERWORD#]` variant, or conditioning on it. See [this](https://groups.google.com/forum/#!topic/implicit-emotions-shared-task-wassa-2018/2wIdY_lmCoY) thread.~ The organizers fixed this.
* Related to the previous point, in some examples the trigger word is a hashtag: `#[#TRIGGERWORD#]`. Maybe we could use this as a feature.
* ~Tokenize and exploit emojis. See [#1](https://github.com/jabalazs/implicit_emotion/issues/1).~

## Model tuning

* ~Try using other pre-trained word embeddings~ `glove.840B.300d` seem to perform best
* ~Shuffle training examples at each epoch~
* ~Try using attention for aggregating character-level representations into word representations~ Preliminary experiments were not promising. It might be worth to try attention over the raw aggregation of the character-level vectors, instead of doing it over the pre-trained GloVe embedding.
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

## Implement better optimizer class
This class should be able to:
* Accept specific parameters for the underlying pytorch optimizers
* Implement different learning rate schedules
  - Transformer's linear increase & O(sqrt(n)) decrease ([paper](https://papers.nips.cc/paper/7181-attention-is-all-you-need))
  - Triangular learning rates ([paper](https://arxiv.org/abs/1506.01186))
  - Slanted triangular learning rates ([paper](https://arxiv.org/abs/1801.06146))
  - Linearly decreasing learning rates
* Switch the underlying pytorch optimizer during training


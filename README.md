# [Implicit Emotion Shared Task](http://implicitemotions.wassa2018.com/)

This code was developed during the WASSA 2018 Implicit Emotion Shared Task by
the team IIIDYT. You can read our paper
[here](https://arxiv.org/abs/1808.08672).

You can also read more details about the shared task in the official IEST
[website](http://implicitemotions.wassa2018.com/), and in the competition
[website](https://competitions.codalab.org/competitions/19214).

If you find this code useful please consider citing our paper:

```bibtex
@InProceedings{balazs2018iiidyt,
  author       = {Balazs, Jorge A. and 
                  Marrese-Taylor, Edison and
                  Matsuo, Yutaka},
  title        = {{IIIDYT at IEST 2018: Implicit Emotion Classification
                   with Deep Contextualized Word Representations}},
  booktitle    = {Proceedings of the 9th Workshop on Computational
                  Approaches to Subjectivity, Sentiment and Social
                  Media Analysis},
  year         = {2018},
  address      = {Brussels, Belgium},
  month        = {November},
  organization = {Association for Computational Linguistics}
}
```

# Recommended Installation

1. Clone this repo.
   ```bash
   git clone https://github.com/jabalazs/implicit_emotion.git
   cd implicit_emotion
   ```

2. Create a conda environment
   
   If you don't have conda installed, we recommend using [miniconda](https://conda.io/miniconda.html).
   
   You can then easily create and activate a new conda environment with Python 3.6
   by executing:
   
   ```bash
   conda create -n iest python=3.6
   source activate iest
   ```
   
   Where you can replace `iest` by any environment name you like.

3. Run 

    ```bash
    scripts/install.sh
    ```

   This will install pytorch, a few dependencies for our code, AllenNLP (ELMo) and
   all of its dependencies. See https://github.com/allenai/allennlp for more ways
   to install AllenNLP. Also note that for replicability purposes we will install
   the same ELMo version we used for development: [`ac2e0b9b6`](https://github.com/allenai/allennlp/tree/ac2e0b9b6e4668984ebd8c05578d9f4894e94bee).

   > By default, AllenNLP will be cloned in this repo. If you want to install it
   > somewhere else please modify the install script
   > [scripts/install.sh](scripts/install.sh), and change the `ALLENNLP_PATH`
   > variable in [src/config.py](src/config.py) accordingly.

   > The installation script will install Pytorch 0.4.0 with CUDA 8.0 by
   > default. Please make sure that you have compatible GPU drivers, or change
   > the installation script so it installs the correct version of CUDA. You can
   > run `nvidia-smi` to see the version of your driver, and check the
   > compatibility with CUDA in
   > [this](https://stackoverflow.com/a/30820690/3941813) chart.

4. (Optional) Install java for obtaining POS tags

   We used a forked version of [`ark-tweet-nlp`](https://github.com/jabalazs/ark-tweet-nlp/tree/7e37f5badcc28d1b5ad595d26721db3832fd1dde)
   for obtaining POS tags without using its built-in tokenization feature. This
   repo already comes with the compiled jar (`ark-tweet-nlp-0.3.2.jar`) in
   [`utils/ark-tweet-nlp`](utils/ark-tweet-nlp).
   
   If you want to use this feature you need java. You can easily install it
   within your conda environment with

   ```bash
   conda install -c cyclus java-jdk
   ```
   
   > You can also change the pre-trained POS tagging model by modifying the
   > `PRETRAINED_MODEL_NAME` variable in [`utils/run_postagger.sh`](utils/run_postagger.sh)
   > with one of the models provided in [`utils/ark-tweet-nlp`](utils/ark-tweet-nlp).

## Data

1. To get the data you need some credentials provided by the organizers of the
   shared task. Please contact them at iest@wassa2018.com, or at the email
   addresses listed in the offical shared task
   [website](http://implicitemotions.wassa2018.com/organizers/), to get the
   credentials for downloading the data.

   > Alternatively, you could download the tweets according to their IDs,
   > already published in the official
   > [website](http://implicitemotions.wassa2018.com/data/), and not requiring
   > any credentials. However, the organizers haven't published the code they
   > used for replacing username mentions, newlines, urls, and trigger-words, so
   > you might not end up with the same dataset that was used during the
   > shared task.

2. Once you have your `USERNAME` and `PASSWORD`, get the data by running the
   following command, and typing your password when prompted:

   ```bash
   scripts/get_data.sh USERNAME
   ```

   This script will download the following:

   - train \ dev \ test splits (~23 MB unzipped) into `data/`
   - pre-trained GloVe embeddings (~2.2 GB zipped, ~5.6 GB unzipped) into
     `data/word_embeddings`
   - pre-trained ELMo weights (~360 MB) into `data/word_embeddings`  

   > If you already downloaded the GloVe embeddings for another project, we
   > recommend commenting the line where they are dowloaded in the `get_data.sh`
   > script, and creating a symbolic link instead (our code will not modify the
   > original embeddings file).

   > If you want to save the data in a different directory, you can do so as
   > long as you modify the paths in the
   > [`scripts/preprocess.sh`](scripts/preprocess.sh) and
   > [`scripts/get_pos.sh`](scripts/get_pos.sh) preprocessing scripts, and in
   > [`src/config.py`](src/config.py).

3. Run the preprocessing script

   ```bash
   scripts/preprocess.sh
   ```

4. (Optional) If you installed java and want to obtain the pos tags, execute:

   ```bash
   scripts/get_pos.sh
   ```

---

To test if you installed everything correctly run `python run.py --help`. This
command should display the options with which you can run the code, or an error
if something failed during the installation process.

# Training

To train a best-performing model, run:

```bash
python run.py --write_mode=BOTH --save_model
```

This will run for 10 epochs and will save the best checkpoint according to
validation accuracy.

Checkpoints and other output files are saved in a directory named after the
`hash` of the current run in [`data/results/`](data/results/). See [this
section](#experiment-results-directory-structure) for more details.


> The `hash` will depend on hyperparameters that impact performance, and the
> current commit hash. For example, changing `learning_rate`, `lstm_hidden_size`,
> `dropout`, would produce different hashes, whereas changing `write_mode`, or
> `save_model` or similars, would not.

# Testing

To test a trained model, run:

```bash
python run.py --model_hash=<partial_model_hash> --test
```

Where you have to replace `<partial_model_hash>` by the hash of the model you
wish to test, corresponding to the name of its directory located in
[`data/results/`](data/results/).

A classification report will be printed on screen, and files containing the
prediction labels and probabilities will be created in `data/results/<hash>`
([details](#test_results)).


# Experiment Results Directory Structure

**After the validation phase of the first epoch you should have the following
structure**:

```
data/results/<hash>
├── architecture.txt
├── best_dev_predictions.txt
├── best_dev_probabilities.csv
├── best_model.pth
├── best_model_state_dict.pth
├── events.out.tfevents.1535523327
└── hyperparams.json
```

* `architecture.txt` contains the architecture as represented by PyTorch. For
  example:

  ```
  IESTClassifier(
      (char_embeddings): Embedding(1818, 50)
      (word_encoding_layer): WordEncodingLayer(method=elmo)
      (word_dropout): Dropout(p=0.5)
      (sent_encoding_layer): SentenceEncodingLayer(
        (sent_encoding_layer): BLSTMEncoder(
          (enc_lstm): LSTM(1024, 2048, dropout=0.2, bidirectional=True)
        )
      )
      (sent_dropout): Dropout(p=0.2)
      (pooling_layer): PoolingLayer(
        (pooling_layer): MaxPoolingLayer()
      )
      (dense_layer): Sequential(
        (0): Linear(in_features=4096, out_features=512, bias=True)
        (1): ReLU()
        (2): Dropout(p=0.5)
        (3): Linear(in_features=512, out_features=6, bias=True)
      )
  )
  ```

* `best_dev_predictions.txt` contains 9591 rows with a single column containing
  the predicted label for the best epoch for the dev (trial) examples. This is
  how its `head` looks like:

  ```txt
  surprise
  disgust
  anger
  disgust
  surprise
  sad
  fear
  anger
  disgust
  joy
  ```

* `best_dev_probabilities.csv` contains 9591 comma-separated rows, with 6
  columns corresponding to the probability of the example belonging to one of
  the 6 emotion classes. This is how its `head` looks like:

  ```
  0.11771341,0.11176660,0.01460518,0.06944314,0.03313787,0.65333384
  0.28856099,0.36440939,0.03660037,0.04640478,0.13095142,0.13307315
  0.25572592,0.18113331,0.10848557,0.16581367,0.24334571,0.04549576
  0.32709056,0.36550751,0.09679491,0.02511709,0.02381524,0.16167462
  0.00716987,0.12533677,0.02027770,0.00367037,0.24190214,0.60164315
  0.00020374,0.01509732,0.00003231,0.00025629,0.98010033,0.00430998
  0.19336309,0.19491623,0.40776601,0.05498404,0.05227452,0.09669605
  0.63258690,0.02079128,0.01398562,0.21799077,0.10592879,0.00871667
  0.09783797,0.29105908,0.08870091,0.12649100,0.22437103,0.17154007
  0.16136999,0.00479498,0.01619518,0.73850989,0.07212262,0.00700729
  ```

  This is the header of the file (not included in the file itself):

  ```
  anger,disgust,fear,joy,sad,surprise
  ```

* `best_model.pth`: The whole model, serialized by running `torch.save(model,
  PATH)`.
* `best_model_state_dict.pth`: The model weigths, serialized by running
  `torch.save(model.state_dict(), PATH)`.

  > For more on Pytorch serialization, see: [Serialization
  > Semantics](https://pytorch.org/docs/0.4.0/notes/serialization.html).

* `events.out.tfevents.1535523327`: TensorBoard file generated by
  [`tensorboardX`](https://github.com/lanpa/tensorboardX).

* `hyperparams.json`: Hyperparameters with which the model was trained, and some
  extra information, such as the date the model was run and its hash. For
  example:

  ```json
  {
    "epochs": 10,
    "batch_size": 64,
    "optim": "adam",
    "seed": 45,
    ...
    "max_lr": 0.001,
    "datetime": "2018-08-29 17:36:09",
    "commit": "76e0af0150fc35f9be6cd993dc35b2dc7a4bb87d",
    "hash": "dc700889fa1bbae360bbd7afa68cd9d02c154d62"
  }
  ```

<a id="test_results"></a>
**After testing a model, two new files will be created in `data/results/<hash>`**:

* `test_predictions.txt`: equivalent to `best_dev_predictions.txt`, but obtained
  from evaluating the trained model on the test dataset.

* `test_probabilities.csv`: equivalent to `best_dev_probabilites.txt`, but obtained
  from evaluating the trained model on the test dataset.

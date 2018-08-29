# [Implicit Emotion Shared Task](http://implicitemotions.wassa2018.com/)

[Competition website](https://competitions.codalab.org/competitions/19214)

# Installation

1. Clone this repo.
   ```
   git clone https://github.com/jabalazs/implicit_emotion.git
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

   > The installation script will install Pytorch 0.4.0 with CUDA 8.0 by
   > default. Please make sure that you have compatible GPU drivers, or change
   > the installation script so it installs the correct version of CUDA. You can
   > run `nvidia-smi` to see the version of your driver, and check the
   > compatibility with CUDA in
   > [this](https://stackoverflow.com/a/30820690/3941813) chart.

4. (Optional) Install java for obtaining POS tags

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

1. Get the data by running the following command and typing your password when prompted
   ```
   ./get_data.sh <USERNAME>
   ```

   This script will get the train \ dev \ test splits (~23 MB unzipped),
   pre-trained GloVe embeddings (~2.2 GB zipped, ~5.6 GB unzipped), and
   pre-trained ELMo weights (~360 MB).

   > If you already downloaded the GloVe embeddings for another project, we
   > recommend commenting the line where they are dowloaded in the `get_data.sh`
   > script, and creating a symbolic link instead (our code will not modify the
   > original embeddings file).

   > If you want to save the data in a different directory, you can do so as
   > long as you modify the paths in the `preprocess.sh` and `get_pos.sh`
   > preprocessing scripts, and in [`src/config.py`](src/config.py).

2. Run the preprocessing script
   ```
   ./preprocess.sh
   ```

3. (Optional) If you installed java and want to obtain the pos tags, execute:
   ```
   ./get_pos.sh
   ```

---

To test if you installed everything correctly run `python run.py --help`. This
command should display the options with which you can run the code, or an error
if something failed during the installation process.

# Training

To train a best-performing model, run:

```
python run.py --corpus=iest_emoji --log_interval=50  --lstm_hidden_size=2048 -wem=elmo --seed=43 --epochs=10 --dropout=0.5 --sent_enc_dropout=0.2 --write_mode=BOTH --save_model
```

This will run for 10 epochs and will save the following files in
[`data/results/`](data/results/):

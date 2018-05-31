import os

from glob import glob

import torch
import colored_traceback

import numpy as np

from tqdm import tqdm

from src.corpus.iestcorpus import IESTCorpus
from src.corpus.embeddings import Embeddings
from src.utils.logger import Logger
# from src.utils.io import load_or_create
from src.train import Trainer
from src.optim import OptimWithDecay
from src import config

from src.models.base import (
                             NLIClassifier,
                             WordEncodingLayer,
                             WordCharEncodingLayer,
                             SentenceEncodingLayer,
                            )

from base_args import base_parser, CustomArgumentParser

colored_traceback.add_hook(always=True)


base_parser.description = 'PyTorch MultiNLI Inner Attention Classifier'
arg_parser = CustomArgumentParser(parents=[base_parser],
                                  description='PyTorch MultiNLI')

arg_parser.add_argument('--model', type=str, default="infersent",
                        choices=SentenceEncodingLayer.SENTENCE_ENCODING_METHODS,
                        help='Model to use')

arg_parser.add_argument('--corpus', type=str, default="multinli",
                        choices=list(config.corpora_dict.keys()),
                        help='Name of the corpus to use.')

arg_parser.add_argument('--mismatched_dev', action='store_true',
                        help='Whether to use the mismatched dev dataset for '
                             'MultiNLI')

arg_parser.add_argument('--embeddings', type=str, default="glove",
                        choices=list(config.embedding_dict.keys()),
                        help='Name of the embeddings to use.')

arg_parser.add_argument('--lstm_hidden_size', type=int, default=2048,
                        choices=list(config.embedding_dict.keys()),
                        help='Hidden dimension size for the word-level LSTM')

arg_parser.add_argument('--force_reload', action='store_true',
                        help='Whether to reload pickles or not (makes the '
                        'process slower, but ensures data coherence)')
arg_parser.add_argument('--char_emb_dim', type=int, default=50,
                        help='Char embedding dimension')
arg_parser.add_argument('--pooling_method', type=str, default='max',
                        choices=['mean', 'sum', 'last', 'max'],
                        help='Pooling scheme to use as raw sentence '
                             'representation method.')
arg_parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout applied to layers. 0 means no dropout.')

arg_parser.add_argument('--model_hash', type=str, default=None,
                        help='Hash of the model to load, can be a partial hash')

arg_parser.add_argument('--update_learning_rate_nie', '-ulrn', action='store_true')

arg_parser.add_argument('--word_encoding_method', '-wem', type=str, default="embed",
                        choices=WordEncodingLayer.WORD_ENCODING_METHODS,
                        help='How to obtain word representations')

arg_parser.add_argument('--word_char_aggregation_method', '-wcam',
                        choices=WordCharEncodingLayer.AGGREGATION_METHODS,
                        default=None,
                        help='Way in which character-level and word-level word '
                             'representations are aggregated')


def validate_args(hp):
    """hp: argparser parsed arguments. type: Namespace"""
    assert not (hp.update_learning_rate and hp.update_learning_rate_nie)

    if hp.word_encoding_method == 'char_lstm' and not hp.word_char_aggregation_method:
        raise ValueError(f'Need to pass a word_char_aggregation_method when '
                         f'using char_lstm word_encoding_method. '
                         f'Choose one from {WordCharEncodingLayer.AGGREGATION_METHODS}')

    if hp.mismatched_dev and hp.corpus != 'multinli':
        raise ValueError('mismatched_dev flag passed but a corpus other than '
                         'multinli is being used. Either disable '
                         'mismatched_dev or choose multinli as corpus')


def main():
    hp = arg_parser.parse_args()
    validate_args(hp)

    logger = Logger(hp, model_name='Baseline', write_mode=hp.write_mode)
    if hp.write_mode != 'NONE':
        logger.write_hyperparams()

    torch.manual_seed(hp.seed)
    torch.cuda.manual_seed_all(hp.seed)  # silently ignored if there are no GPUs

    CUDA = False
    if torch.cuda.is_available() and not hp.no_cuda:
        CUDA = True

    # these must match the ones found in config.corpora_dict
    if hp.corpus == 'iest':
        CorpusClass = IESTCorpus
    else:
        raise NotImplementedError('Corpus not implemented')

    corpus = CorpusClass(force_reload=hp.force_reload,
                         train_data_proportion=hp.train_data_proportion,
                         valid_data_proportion=hp.dev_data_proportion,
                         batch_size=hp.batch_size)

    if hp.model_hash:
        experiment_path = os.path.join(config.RESULTS_PATH, hp.model_hash + '*')
        ext_experiment_path = glob(experiment_path)
        assert len(ext_experiment_path) == 1, 'Try provinding a longer model hash'
        model_path = os.path.join(ext_experiment_path[0], 'best_model.pth')
        import ipdb; ipdb.set_trace(context=10)
        model = torch.load(model_path)

    # Load pre-trained embeddings
    embeddings = Embeddings(config.embedding_dict[hp.embeddings],
                            k_most_frequent=None,
                            force_reload=hp.force_reload)

    # Get subset of embeddings corresponding to our vocabulary
    embedding_matrix = embeddings.generate_embedding_matrix(corpus.word2index)
    print(f'{len(embeddings.unknown_tokens)} words from vocabulary not found '
          f'in {hp.embeddings} embeddings. ')

    # Initialize torch Embedding object with subset of pre-trained embeddings
    torch_embeddings = torch.nn.Embedding(*embedding_matrix.shape)
    torch_embeddings.weight = torch.nn.Parameter(torch.Tensor(embedding_matrix))

    # Repeat process for character embeddings with the difference that they are
    # not pretrained

    # Initialize character embedding matrix randomly
    char_vocab_size = len(corpus.char2index)
    char_embedding_matrix = np.random.uniform(-0.05, 0.05,
                                              size=(char_vocab_size,
                                                    hp.char_emb_dim))
    char_torch_embeddings = torch.nn.Embedding(*char_embedding_matrix.shape)
    char_torch_embeddings.weight = torch.nn.Parameter(
                                            torch.Tensor(char_embedding_matrix))

    # Define some specific parameters for the model
    num_classes = len(corpus.label_ids)
    batch_size = corpus.train_batches.batch_size

    if hp.model == 'stacked':
        # Nie's model: Stacked LSTM
        hidden_sizes = [512, 1024, 2048]
        # hidden_sizes = [128, 256, 512]

    elif hp.model == 'infersent':
        # Conneau's InferSent
        hidden_sizes = hp.lstm_hidden_size

    model = NLIClassifier(num_classes, batch_size,
                          torch_embeddings=torch_embeddings,
                          char_embeddings=char_torch_embeddings,
                          word_encoding_method=hp.word_encoding_method,
                          word_char_aggregation_method=hp.word_char_aggregation_method,
                          sent_encoding_method=hp.model,
                          hidden_sizes=hidden_sizes,
                          use_cuda=CUDA,
                          pooling_method=hp.pooling_method,
                          batch_first=True,
                          dropout=hp.dropout)

    if CUDA:
        model.cuda()
    logger.write_current_run_details(str(model))

    optimizer = OptimWithDecay(model.parameters(),
                               method=hp.optim,
                               initial_lr=hp.learning_rate,
                               max_grad_norm=hp.grad_clipping,
                               lr_decay=hp.learning_rate_decay,
                               start_decay_at=hp.start_decay_at,
                               decay_every=hp.decay_every)

    loss_function = torch.nn.CrossEntropyLoss()

    # Whether to use matched or mismatched MultiNLI data
    dev_batches = corpus.dev_batches
    if hp.mismatched_dev:
        dev_batches = corpus.dev_mismatched_batches

    trainer = Trainer(model, corpus.train_batches,
                      dev_batches,
                      optimizer, loss_function, num_epochs=hp.epochs,
                      update_learning_rate=hp.update_learning_rate,
                      use_cuda=CUDA, log_interval=hp.log_interval)

    try:
        best_accuracy = None
        for epoch in tqdm(range(hp.epochs), desc='Epoch'):
            total_loss = 0
            trainer.train_epoch(epoch)

            eval_dict = trainer.evaluate()
            if hp.update_learning_rate:
                optim_updated, new_lr = trainer.optimizer.updt_lr_accuracy(epoch, eval_dict['accuracy'])
                if new_lr < 10e-5:
                    break
                if optim_updated:
                    tqdm.write(f'Learning rate decayed to {new_lr}')

            elif hp.update_learning_rate_nie:
                optim_updated, new_lr = trainer.optimizer.update_learning_rate_nie(epoch)
                if optim_updated:
                    tqdm.write(f'Learning rate decayed to {new_lr}')

            accuracy = eval_dict['accuracy']
            if not best_accuracy or accuracy > best_accuracy:
                best_accuracy = accuracy
                logger.update_results({'best_valid_acc': best_accuracy,
                                       'best_epoch': epoch})
                if hp.save_model:
                    logger.torch_save_file('best_model_state_dict.pth',
                                           model.state_dict(),
                                           progress_bar=tqdm)
                    logger.torch_save_file('best_model.pth',
                                           model,
                                           progress_bar=tqdm)

        logger.insert_in_googlesheets()
    except KeyboardInterrupt:
        logger.insert_in_googlesheets()


if __name__ == '__main__':
    main()

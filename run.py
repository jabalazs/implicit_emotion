import os

from glob import glob

import torch
import colored_traceback
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from src.corpus.iest_corpus import IESTCorpus
from src.corpus.embeddings import Embeddings

from src.utils.logger import Logger
from src.utils.ops import np_softmax

from src.train import Trainer
from src.optim.optim import OptimWithDecay, ScheduledOptim
from src.optim.schedulers import SlantedTriangularScheduler, TransformerScheduler
from src import config

from src.models.iest import (
                             IESTClassifier,
                             WordEncodingLayer,
                             WordCharEncodingLayer,
                             SentenceEncodingLayer,
                            )

from src.layers.pooling import PoolingLayer

from base_args import base_parser, CustomArgumentParser

colored_traceback.add_hook(always=True)


base_parser.description = 'PyTorch MultiNLI Inner Attention Classifier'
arg_parser = CustomArgumentParser(parents=[base_parser],
                                  description='PyTorch MultiNLI')

arg_parser.add_argument('--model', type=str, default="bilstm",
                        choices=SentenceEncodingLayer.SENTENCE_ENCODING_METHODS,
                        help='Model to use')

arg_parser.add_argument('--corpus', type=str, default="iest",
                        choices=list(config.corpora_dict.keys()),
                        help='Name of the corpus to use.')

arg_parser.add_argument('--embeddings', type=str, default="glove",
                        choices=list(config.embedding_dict.keys()),
                        help='Name of the embeddings to use.')

arg_parser.add_argument('--lstm_hidden_size', type=int, default=2048,
                        help='Hidden dimension size for the word-level LSTM')

arg_parser.add_argument('--sent_enc_layers', type=int, default=1,
                        help='Number of layers for the word-level LSTM')

arg_parser.add_argument('--force_reload', action='store_true',
                        help='Whether to reload pickles or not (makes the '
                        'process slower, but ensures data coherence)')
arg_parser.add_argument('--char_emb_dim', '-ced', type=int, default=50,
                        help='Char embedding dimension')
arg_parser.add_argument('--pooling_method', type=str, default='max',
                        choices=PoolingLayer.POOLING_METHODS,
                        help='Pooling scheme to use as raw sentence '
                             'representation method.')

arg_parser.add_argument('--sent_enc_dropout', type=float, default=0.0,
                        help='Dropout between sentence encoding lstm layers. '
                             '0 means no dropout.')

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

arg_parser.add_argument('--lowercase', '-lc', action='store_true',
                        help='Whether to lowercase data or not. WARNING: '
                             'REMEBER TO CLEAR THE CACHE BY PASSING '
                             '--force_reload or deleting .cache')

arg_parser.add_argument("--warmup_iters", "-wup", default=4000, type=int,
                        help="During how many iterations to increase the learning rate")


def validate_args(hp):
    """hp: argparser parsed arguments. type: Namespace"""
    assert not (hp.update_learning_rate and hp.update_learning_rate_nie)

    if hp.word_encoding_method == 'char_lstm' and not hp.word_char_aggregation_method:
        raise ValueError(f'Need to pass a word_char_aggregation_method when '
                         f'using char_lstm word_encoding_method. '
                         f'Choose one from {WordCharEncodingLayer.AGGREGATION_METHODS}')


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

    corpus = IESTCorpus(hp.corpus,
                        force_reload=hp.force_reload,
                        train_data_proportion=hp.train_data_proportion,
                        dev_data_proportion=hp.dev_data_proportion,
                        batch_size=hp.batch_size,
                        lowercase=hp.lowercase)

    if hp.model_hash:
        experiment_path = os.path.join(config.RESULTS_PATH, hp.model_hash + '*')
        ext_experiment_path = glob(experiment_path)
        assert len(ext_experiment_path) == 1, 'Try provinding a longer model hash'
        model_path = os.path.join(ext_experiment_path[0], 'best_model.pth')
        # FIXME: This will get replaced by the model being loaded below
        model = torch.load(model_path)

    # Load pre-trained embeddings
    embeddings = Embeddings(config.embedding_dict[hp.embeddings],
                            k_most_frequent=None,
                            force_reload=hp.force_reload)

    # Get subset of embeddings corresponding to our vocabulary
    embedding_matrix = embeddings.generate_embedding_matrix(corpus.lang.token2id)
    print(f'{len(embeddings.unknown_tokens)} words from vocabulary not found '
          f'in {hp.embeddings} embeddings. ')

    # Repeat process for character embeddings with the difference that they are
    # not pretrained

    # Initialize character embedding matrix randomly
    char_vocab_size = len(corpus.lang.char2id)
    char_embedding_matrix = np.random.uniform(-0.05, 0.05,
                                              size=(char_vocab_size,
                                                    hp.char_emb_dim))

    # Define some specific parameters for the model
    num_classes = len(corpus.label2id)
    batch_size = corpus.train_batches.batch_size

    hidden_sizes = hp.lstm_hidden_size
    model = IESTClassifier(num_classes, batch_size,
                           embedding_matrix=embedding_matrix,
                           char_embedding_matrix=char_embedding_matrix,
                           word_encoding_method=hp.word_encoding_method,
                           word_char_aggregation_method=hp.word_char_aggregation_method,
                           sent_encoding_method=hp.model,
                           hidden_sizes=hidden_sizes,
                           use_cuda=CUDA,
                           pooling_method=hp.pooling_method,
                           batch_first=True,
                           dropout=hp.dropout,
                           sent_enc_dropout=hp.sent_enc_dropout,
                           sent_enc_layers=hp.sent_enc_layers)

    if CUDA:
        model.cuda()

    if hp.write_mode != 'NONE':
        logger.write_architecture(str(model))

    logger.write_current_run_details(str(model))

    if hp.model == 'transformer':
        core_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        transformer_scheduler = TransformerScheduler(
            1024,
            factor=1,
            warmup_steps=hp.warmup_iters
        )
        optimizer = ScheduledOptim(core_optimizer, transformer_scheduler)
    else:
        # optimizer = OptimWithDecay(model.parameters(),
        #                            method=hp.optim,
        #                            initial_lr=hp.learning_rate,
        #                            max_grad_norm=hp.grad_clipping,
        #                            lr_decay=hp.learning_rate_decay,
        #                            start_decay_at=hp.start_decay_at,
        #                            decay_every=hp.decay_every)

        core_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0,
        )

        max_iter = corpus.train_batches.num_batches * hp.epochs
        slanted_triangular_scheduler = SlantedTriangularScheduler(
            max_iter,
            max_lr=0.005,
            cut_fraction=0.1,
            ratio=32
        )
        optimizer = ScheduledOptim(core_optimizer, slanted_triangular_scheduler)

    loss_function = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model, corpus.train_batches, corpus.dev_batches,
                      optimizer, loss_function, num_epochs=hp.epochs,
                      use_cuda=CUDA, log_interval=hp.log_interval)

    writer = SummaryWriter(logger.run_savepath)
    try:
        best_accuracy = None
        for epoch in tqdm(range(hp.epochs), desc='Epoch'):
            total_loss = 0

            trainer.train_epoch(epoch, writer)
            eval_dict = trainer.evaluate(epoch, writer)

            if hp.update_learning_rate:
                if hp.model != 'transformer':
                    optim_updated, new_lr = trainer.optimizer.updt_lr_accuracy(epoch, eval_dict['accuracy'])
                    lr_threshold = 1e-5
                    if new_lr < lr_threshold:
                        tqdm.write(f'Learning rate smaller than {lr_threshold}, '
                                   f'stopping.')
                        break
                    if optim_updated:
                        tqdm.write(f'Learning rate decayed to {new_lr}')

            # elif hp.update_learning_rate_nie:
            #     optim_updated, new_lr = trainer.optimizer.update_learning_rate_nie(epoch)
            #     if optim_updated:
            #         tqdm.write(f'Learning rate decayed to {new_lr}')

            accuracy = eval_dict['accuracy']
            if not best_accuracy or accuracy > best_accuracy:
                best_accuracy = accuracy
                logger.update_results({'best_valid_acc': best_accuracy,
                                       'best_epoch': epoch})

                if hp.write_mode != 'NONE':
                    probs = np_softmax(eval_dict['output'])
                    probs_filepath = os.path.join(logger.run_savepath,
                                                  'best_eval_probs.csv')
                    np.savetxt(probs_filepath, probs,
                               delimiter=',', fmt='%.8f')

                    labels_filepath = os.path.join(logger.run_savepath,
                                                   'predictions.txt')
                    labels = [label + '\n' for label in eval_dict['labels']]
                    with open(labels_filepath, 'w', encoding='utf-8') as f:
                        f.writelines(labels)

                if hp.save_model:
                    logger.torch_save_file('best_model_state_dict.pth',
                                           model.state_dict(),
                                           progress_bar=tqdm)
                    logger.torch_save_file('best_model.pth',
                                           model,
                                           progress_bar=tqdm)

        # logger.insert_in_googlesheets()
    except KeyboardInterrupt:
        # logger.insert_in_googlesheets()
        pass


if __name__ == '__main__':
    main()

"""
OtimWithDecay taken from
https://github.com/pytorch/examples/blob/master/OpenNMT/onmt/Optim.py
"""

import math

import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self.optimizer = optimizer

        self.step_num = 0
        self.lr = 0

    def step(self):
        self.step_num += 1
        rate = self.get_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.lr = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_rate(self, step=None):
        if step is None:
            step = self.step_num
        return self.factor * (self.model_size ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))


def slanted_triangular_lr(step, max_step, max_lr=0.01, cut_fraction=0.1, ratio=32):
    """Calculate triangular-shaped learning rate schedule

    From Howard & Ruder's (2018) paper:
    Universal Language Model Fine-tuning for Text Classification
    https://arxiv.org/abs/1801.06146

    Parameters
    ----------
    step : int
        Current step during training
    max_step : int
        Last training step (probably should equal num_batches * num_epochs)
    max_lr : float, optional
        Maximum desired learning rate
    cut_fraction : int, optional
        Fraction of steps during which to increase the learning rate
    ratio : int, optional
        How many times bigger is the maximum learning rate as compared to the
        minimum one


    Returns
    -------
    learning_rate : float
        The learning rate for a given step

    """
    cut = math.floor(max_step * cut_fraction)
    if step < cut:
        p = step / cut
    else:
        p = 1 - ((step - cut) / (cut * (1 / cut_fraction - 1)))
    learning_rate = max_lr * (1 + p * (ratio - 1)) / ratio

    return learning_rate


class OptimWithDecay(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        elif self.method == 'rmsprop':
            self.optimizer = optim.RMSprop(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, initial_lr, max_grad_norm=None,
                 lr_decay=1, start_decay_at=5, decay_every=2):

        self.params = list(params)  # careful: params may be a generator
        self.last_ppl = None
        self.lr = initial_lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.decay_every = decay_every
        self.start_decay = False

        self._makeOptimizer()

        self.last_accuracy = 0

        self.step_num = 0

    def step(self):
        # Compute gradients norm.
        self.step_num += 1
        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()

    def update_learning_rate_nie(self, epoch):
        updated = False
        if epoch == self.start_decay_at:
            self.start_decay = True

        if (self.start_decay and
           (epoch - self.start_decay_at) % self.decay_every == 0.0):

            self.lr = self.lr * self.lr_decay
            updated = True
            # print("Decaying learning rate to %g" % self.lr)

        self._makeOptimizer()
        return updated, self.lr

    def updt_lr_accuracy(self, epoch, accuracy):
        """This is the lr update policy that Conneau used for infersent"""
        updated = False
        if accuracy < self.last_accuracy:
            self.lr = self.lr / 5
            updated = True

        self.last_accuracy = accuracy

        self._makeOptimizer()
        return updated, self.lr


if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt

    max_step = 2000
    plt.plot(np.arange(1, max_step), [slanted_triangular_lr(step, max_step) for step in range(1, max_step)])
    plt.show()


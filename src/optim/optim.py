"""
OtimWithDecay taken from
https://github.com/pytorch/examples/blob/master/OpenNMT/onmt/Optim.py
"""

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


# class OptimSlantedTriangular(object):

#     """Optimizer with slanted triangular learning rate schedule"""

#     def __init__(self, core_optimizer, max_step, max_lr=0.01, cut_fraction=0.1,
#                  ratio=32):
#         """

#         Parameters
#         ----------
#         core_optimizer : torch.optim.Optimizer
#             torch object containing the model parameters
#         max_step : int
#         max_lr : float, optional
#         cut_fraction : float, optional
#         ratio : int, optional


#         """
#         self.optimizer = core_optimizer
#         self.max_step = max_step
#         self.max_lr = max_lr
#         self.cut_fraction = cut_fraction
#         self.ratio = ratio

#         self.lr = None
#         self.step_num = 0

#     def step(self):
#         self.step_num += 1
#         rate = self.get_rate()
#         for p in self.optimizer.param_groups:
#             p['lr'] = rate
#         self.lr = rate
#         self.optimizer.step()

#     def get_rate(self, step=None):
#         if step is None:
#             step = self.step_num

#         return slanted_triangular_lr(
#             step,
#             self.max_step,
#             max_lr=self.max_lr,
#             cut_fraction=self.cut_fraction,
#             ratio=self.ratio
#         )


class ScheduledOptim(object):

    """Optimizer with predefined learning rate schedule"""

    def __init__(self, core_optimizer, lr_scheduler):
        """

        Parameters
        ----------
        core_optimizer : torch.optim.Optimizer
        lr_scheduler : Scheduler instance


        """
        self.optimizer = core_optimizer
        self.lr_scheduler = lr_scheduler

        self.step_num = 0
        self.lr = None

    def step(self):
        self.step_num += 1
        rate = self.get_rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.lr = rate
        self.optimizer.step()

    def get_rate(self, step=None):
        if step is None:
            step = self.step_num

        return self.lr_scheduler.get_rate(step)

    def zero_grad(self):
        self.optimizer.zero_grad()


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


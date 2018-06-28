import math


# def slanted_triangular_lr(step, max_step, max_lr=0.01, cut_fraction=0.1, ratio=32):

#     cut = math.floor(max_step * cut_fraction)
#     if step < cut:
#         p = step / cut
#     else:
#         p = 1 - ((step - cut) / (cut * (1 / cut_fraction - 1)))
#     learning_rate = max_lr * (1 + p * (ratio - 1)) / ratio

#     return learning_rate


class SlantedTriangularScheduler(object):

    """Scheduler producing a slanted triangular learning rate schedule

       From Howard & Ruder's (2018) paper:
       Universal Language Model Fine-tuning for Text Classification
       https://arxiv.org/abs/1801.06146
    """

    def __init__(self, max_step, max_lr, cut_fraction, ratio):
        """

        Parameters
        ----------
        max_step : int
            Last training step (probably should equal num_batches * num_epochs)
        max_lr : float, optional
            Maximum desired learning rate
        cut_fraction : int, optional
            Fraction of steps during which to increase the learning rate
        ratio : int, optional
            How many times bigger is the maximum learning rate as compared to the
            minimum one

        """
        self.max_step = max_step
        self.max_lr = max_lr
        self.cut_fraction = cut_fraction
        self.ratio = ratio

    def get_rate(self, step):
        """
        Parameters
        ----------
            step : int
                Current step during training

        Returns
        -------
        learning_rate : float
            The learning rate for a given step
        """
        cut = math.floor(self.max_step * self.cut_fraction)

        if step < cut:
            p = step / cut
        else:
            p = 1 - ((step - cut) / (cut * (1 / self.cut_fraction - 1)))
        learning_rate = self.max_lr * (1 + p * (self.ratio - 1)) / self.ratio

        return learning_rate

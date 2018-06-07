import torch
import numpy as np

from tqdm import tqdm
from sklearn.metrics import classification_report

from .utils.torch import to_var

from . import config


class Trainer(object):
    def __init__(self, model, train_batches, dev_batches, optimizer,
                 loss_function, num_epochs=10,
                 use_cuda=True,
                 log_interval=20):
        self.model = model

        self.train_batches = train_batches
        self.dev_batches = dev_batches

        self.optimizer = optimizer
        self.loss_function = loss_function

        self.num_epochs = num_epochs

        self.use_cuda = use_cuda
        self.log_interval = log_interval

    def train_epoch(self, epoch):
        self.model.train()  # Depends on using pytorch
        num_batches = self.train_batches.num_batches

        total_loss = 0
        for batch_index in tqdm(range(num_batches), desc='Batch'):
            self.model.zero_grad()
            batch = self.train_batches[batch_index]
            ret_dict = self.model(batch)

            # FIXME: This part depends both on the way the batch is built and
            # on using pytorch. Think of how to avoid this. Maybe by creating
            # a specific MultNLI Trainer Subclass?
            labels = batch['labels']
            labels = to_var(torch.LongTensor(labels), self.use_cuda,
                            requires_grad=False)

            # FIXME: this line assumes that the loss_function expects logits
            # and that ret_dict will contain that key, but what if our problem
            # is not classification?
            batch_loss = self.loss_function(ret_dict['logits'], labels)

            batch_loss.backward()
            self.optimizer.step()

            # We ignore batch 0's output for prettier logging
            if batch_index != 0:
                total_loss += batch_loss.item()

            if (batch_index % self.log_interval == 0 and batch_index != 0):

                avg_loss = total_loss / self.log_interval
                tqdm.write(f'Epoch: {epoch}, batch: {batch_index}, loss: {avg_loss}')
                total_loss = 0

        self.train_batches.shuffle_examples()

    def evaluate(self):
        self.model.eval()
        num_batches = self.dev_batches.num_batches
        outputs = []
        true_labels = []
        tqdm.write("Evaluating...")
        for batch_index in range(num_batches):
            batch = self.dev_batches[batch_index]
            out = self.model(batch)
            # FIXME: Shouldn't call data()
            outputs.append(out['logits'].cpu().data.numpy())
            true_labels.extend(batch['labels'])

        output = np.vstack(outputs)
        pred_labels = output.argmax(axis=1)
        true_labels = np.array(true_labels)
        tqdm.write(classification_report(true_labels, pred_labels,
                                         target_names=config.LABELS))
        num_correct = (pred_labels == true_labels).sum()
        num_total = len(pred_labels)
        accuracy = num_correct / num_total
        tqdm.write(f'\nAccuracy: {accuracy:.3f}\n')

        # Generate prediction list
        pred_labels = pred_labels.tolist()
        pred_labels = [config.ID2LABEL[label] for label in pred_labels]
        ret_dict = {'accuracy': accuracy,
                    'labels': pred_labels,
                    'output': output}
        return ret_dict

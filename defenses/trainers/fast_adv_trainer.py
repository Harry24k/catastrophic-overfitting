import torch
import torch.nn as nn

from torchattacks import FFGSM

from .adv_trainer import AdvTrainer

r"""
'Fast is better than free: Revisiting adversarial training'
[https://arxiv.org/abs/2001.03994]

Attributes:
    self.model : model.
    self.device : device where model is.
    self.optimizer : optimizer.
    self.scheduler : scheduler (* Automatically Updated).
    self.max_epoch : total number of epochs.
    self.max_iter : total number of iterations.
    self.epoch : current epoch.
        * Starts from 0.
        * Automatically updated.
    self.iter : current iter.
        * Starts from 0.
        * Automatically updated.
    self.record_keys : items to record (= items returned by do_iter).

Arguments:
    model (nn.Module): model to train.
    eps (float): strength of the attack or maximum perturbation.
    alpha (float): alpha in the paper.

"""

class FastAdvTrainer(AdvTrainer):
    def __init__(self, model, eps, alpha, **kwargs):
        super(FastAdvTrainer, self).__init__("FastAdvTrainer", model, **kwargs)
        # Set Records (* Must be same as the items returned by do_iter)
        self.record_keys = ["Loss", ]
        self.atk = FFGSM(model, eps, alpha)
    
    def _do_iter(self, train_data):
        r"""
        Overridden.
        """
        images, labels = train_data
        X = images.to(self.device)
        Y = labels.to(self.device)

        X_adv = self.atk(X, Y)

        pre = self.model(X_adv)
        cost = nn.CrossEntropyLoss()(pre, Y)

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        return cost.item(), 
    
    
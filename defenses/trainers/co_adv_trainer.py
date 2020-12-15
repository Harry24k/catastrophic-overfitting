import torch
import torch.nn as nn

from torchattacks.attack import Attack

from .adv_trainer import AdvTrainer

r"""
'Understanding Catastrophic Overfitting in Single-step Adversarial Training'
[https://arxiv.org/abs/2010.01799]

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


class COAdvTrainer(AdvTrainer):
    def __init__(self, model, eps, alpha, c, inf_batch=1024):
        super(COAdvTrainer, self).__init__("COAdvTrainer", model)
        # Set Records (* Must be same as the items returned by do_iter)
        self.record_keys = ["Loss", "Eps",]
        self.atk = CustomFFGSM(model, eps, alpha)
        
        # Number of checkpoints.
        self.c = c
        
        # Number of inference batch size.
        self.inf_batch = inf_batch
    
    def _do_iter(self, train_data):
        r"""
        Overridden.
        """
        images, labels = train_data
        X = images.to(self.device)
        Y = labels.to(self.device)
        batch_size = len(X)

        # Generate adversarial images.
        X_adv, logit_clean = self.atk(X, Y)
        pert = (X_adv-X).detach()

        # Get correctly classified indexes.
        _, pre_clean = torch.max(logit_clean.data, 1)
        correct = (pre_clean == Y)
        correct_idx = torch.masked_select(torch.arange(batch_size).to(self.device), correct)
        wrong_idx = torch.masked_select(torch.arange(batch_size).to(self.device), ~correct)
        
        # Use misclassified images as final images.
        X_adv[wrong_idx] = X[wrong_idx].detach()
        
        # Make checkpoints.
        # e.g., (batch_size*(c-1))*3*32*32 for CIFAR10.
        Xs = (torch.cat([X]*(self.c-1)) + \
              torch.cat([torch.arange(1, self.c).to(self.device).view(-1, 1)]*batch_size, dim=1).view(-1, 1, 1, 1) * \
              torch.cat([pert/self.c]*(self.c-1)))
        Ys = torch.cat([Y]*(self.c-1))
                
        # Inference checkpoints for correct images.
        idx = correct_idx
        idxs = []
        self.model.eval()
        with torch.no_grad():
            for k in range(self.c-1):
                # Stop iterations if all checkpoints are correctly classiffied.
                if len(idx) == 0:
                    break
                # Stack checkpoints for inference.
                elif (self.inf_batch >= (len(idxs)+1)*len(idx)):
                    idxs.append(idx + k*batch_size)
                else:
                    pass
                
                # Do inference.
                if (self.inf_batch < (len(idxs)+1)*len(idx)) or (k==self.c-2):
                    # Inference selected checkpoints.
                    idxs = torch.cat(idxs).to(self.device)
                    pre = self.model(Xs[idxs]).detach()
                    _, pre = torch.max(pre.data, 1)
                    correct = (pre == Ys[idxs]).view(-1, len(idx))
                    
                    # Get index of misclassified images for selected checkpoints.
                    max_idx = idxs.max() + 1
                    wrong_idxs = (idxs.view(-1, len(idx))*(1-correct*1)) + (max_idx*(correct*1))
                    wrong_idx, _ = wrong_idxs.min(dim=0)
                    
                    wrong_idx = torch.masked_select(wrong_idx, wrong_idx < max_idx)
                    update_idx = wrong_idx%batch_size
                    X_adv[update_idx] = Xs[wrong_idx]
                    
                    # Set new indexes by eliminating updated indexes.
                    idx = torch.tensor(list(set(idx.cpu().data.numpy().tolist())\
                                            -set(update_idx.cpu().data.numpy().tolist())))
                    idxs = []
        
        # Train final images.
        self.model.train()
        
        pre = self.model(X_adv.detach().to(self.device))
        cost = nn.CrossEntropyLoss()(pre, Y)

        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()
        
        # Calculate average perturbation.
        delta, _ = (X_adv.to(self.device)-X).abs().view(batch_size,-1).max(dim=1)
        
        return cost.item(), delta.mean().item(), 
    
# Modified from torchattacks.
class CustomFFGSM(Attack):
    def __init__(self, model, eps=8/255, alpha=10/255):
        super(CustomFFGSM, self).__init__("CustomFFGSM", model)
        self.eps = eps
        self.alpha = alpha

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        labels = self._transform_label(images, labels)
        
        loss = nn.CrossEntropyLoss()

        adv_images = images + torch.randn_like(images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        adv_images.requires_grad = True

        outputs = self.model(adv_images)
        cost = self._targeted*loss(outputs, labels)

        grad = torch.autograd.grad(cost, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = adv_images + self.alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images, outputs

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import random

import torch
import torch.optim as optim

from defenses.loader import base_loader
from defenses.model import get_model
from defenses.trainer import COAdvTrainer
from defenses.trainer import FastAdvTrainer

def run(name, method, model, gpu, scheduler, epochs, eps, alpha, c, inf_batch, path, save_type):

    torch.cuda.set_device(gpu)
    
    DATA_NAME = "CIFAR10"
    
    # Set Train, Test Loader
    train_loader, test_loader = base_loader(data_name=DATA_NAME,
                                            shuffle_train=True)
    train_loader_ns, _ = base_loader(data_name=DATA_NAME,
                                     shuffle_train=False) #Train w/o Suffle

    # Get First batch
    train_set = iter(train_loader_ns).next()
    test_set = iter(test_loader).next()

    # Set Model
    model = get_model(name=model, num_classes=10).cuda()

    # Set Trainer
    if method == "fast":
        trainer = FastAdvTrainer(model, eps=eps, alpha=alpha, c=c)

    elif method == "proposed":
        trainer = COAdvTrainer(model, eps=eps, alpha=alpha, c=c, inf_batch=inf_batch)

    else:
        raise ValueError(method + " is not supported method.")
        
    trainer.record_rob(train_set, test_set, eps=eps, alpha=alpha, steps=7)

    optimizer="SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)"
    
    if scheduler=="Cyclic":
        scheduler="Cyclic(0, 0.3)"
        scheduler_type="Iter"
    elif scheduler=="Stepwise":
        scheduler="Step([60, 120, 160], 0.1)"
        scheduler_type="Epoch"
    else:
        raise ValueError("%s is not supported scheduler."%(scheduler))
        
    # Train Model
    if save_type == "None":
        save_type = None
        
    trainer.train(train_loader=train_loader, max_epoch=epochs,
                  optimizer=optimizer, scheduler=scheduler, scheduler_type=scheduler_type,
                  save_type=save_type, save_path=path+name,
                  save_overwrite=False, record_type="Epoch")

    trainer.save_all(path+name)

    print("Train Done!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, help='Name for this training script')
    parser.add_argument('--method', type=str, choices=['fast', 'proposed'], help='Training method')
    parser.add_argument('--model', default='PRN18', type=str, choices=['PRN18', 'WRN28'], help='Model Structure')
    parser.add_argument('--gpu', default=0, type=int, help='GPU to be used')
    parser.add_argument('--scheduler', default='Stepwise', type=str, choices=['Stepwise', 'Cyclic'], help='Scheduler type')
    parser.add_argument('--epochs', default=200, type=int, help='Numbers of epochs')
    parser.add_argument('--eps', default=8, type=float, help='Maximum perturbation (ex.8)')
    parser.add_argument('--alpha', default=10, type=float, help='Stepsize (ex.12)')
    parser.add_argument('--c', default=3, type=int, help='Number of checkpoints')
    parser.add_argument('--inf-batch', default=1024, type=int, help='Number of batches during checkpoints inference')
    parser.add_argument('--path', default="./", type=str, help='Save path')
    parser.add_argument('--save-type', default="None", type=str, choices=['None', 'Epoch'], help='Epoch to save the model every epoch')
    args = parser.parse_args()

    run(args.name,
        args.method,
        args.model,
        args.gpu,
        args.scheduler,
        args.epochs,
        args.eps/255,
        args.alpha/255,
        args.c,
        args.inf_batch,
        args.path,
        args.save_type)

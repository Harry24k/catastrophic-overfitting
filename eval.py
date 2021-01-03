from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch

from torchattacks import FGSM, PGD, MultiAttack

from defenses.loader import base_loader
from defenses.model import get_model


def run(name, model, root, data_path, gpu, method, eps, alpha, steps, restart):

    torch.cuda.set_device(gpu)

    # Set Model
    model = get_model(name=model, num_classes=10).cuda()
    
    DATA_NAME = "CIFAR10"
    
    # Set Train, Test Loader
    train_loader, test_loader = base_loader(data_name=DATA_NAME,
                                            shuffle_train=True)

    model.load_state_dict(torch.load(root+name))
    model = model.cuda().eval()

    if method == "FGSM":
        fgsm = FGSM(model, eps=eps)
        fgsm.set_return_type('int')
        fgsm.save(data_loader=test_loader,
                  save_path=data_path, verbose=True)
    elif method == "PGD":
        pgd = PGD(model, eps=eps, alpha=alpha, steps=steps, random_start=True)
        multi = MultiAttack([pgd]*restart)
        multi.set_return_type('int')
        multi.save(data_loader=test_loader,
                   save_path=data_path, verbose=True)
    else:
        raise ValueError(method + " is not supported method.")

    print("Test Done!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluation script", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, help='Name of saved model')
    parser.add_argument('--root', type=str, help='Root of saved model')
    parser.add_argument('--model', type=str, default='PRN18', choices=['PRN18', 'WRN28'], help='Model Structure')
    parser.add_argument('--data-path', type=str, default=None, help='Path for saving adversarial images')
    parser.add_argument('--gpu', default=0, type=int, help='GPU number to be used')
    parser.add_argument('--method', type=str, default='FGSM', choices=['FGSM', 'PGD'], help='Training method')
    parser.add_argument('--eps', default=8, type=float, help='Maximum perturbation (ex.8)')
    parser.add_argument('--alpha', default=2, type=float, help='Stepsize (ex.12)')
    parser.add_argument('--steps', default=1, type=int, help='Number of steps of PGD')
    parser.add_argument('--restart', default=1, type=int, help='Number of restart of PGD')
    args = parser.parse_args()

    run(args.name,
        args.model,
        args.root,
        args.data_path,
        args.gpu,
        args.method,
        args.eps/255,
        args.alpha/255,
        args.steps,
        args.restart)

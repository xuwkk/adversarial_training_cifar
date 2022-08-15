"""
evaluation on the best model
"""

from utils.resnet import ResNet18
import foolbox as fb
from utils.dataset import return_dataloader
from foolbox.utils import accuracy
import torch
from foolbox.attacks import LinfPGD

def evaluation_clean(fmodel, loader):
    ACCURACY = 0.
    for batch_idx, (image, label) in enumerate(loader):
        image, label = image.to(args.devices), label.to(args.devices)
        ACCURACY += accuracy(fmodel, image, label) * len(image)
    
    print(ACCURACY/len(loader.dataset))

def evaluation_adver(fmodel, loader, attack, epsilon):
    ACCURACY = 0.
    for batch_idx, (image, label) in enumerate(loader):
        image, label = image.to(args.devices), label.to(args.devices)
        raw, clipped, is_adv = attack(fmodel, image, label, epsilons = epsilon)
        ACCURACY += len(image) - is_adv.sum().item()

    print(ACCURACY/len(loader.dataset))

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument("--network_type", type=str, default="resnet")
    parser.add_argument("--folder_dir", type = str, default = "trained_model/trades/beta_1.0")
    parser.add_argument("--batch_size", type = int, default = 256)
    parser.add_argument("--devices", type = str, default = "cuda:1")
    parser.add_argument("--small_set", type = bool, default = False)
    
    # attack params
    parser.add_argument("--epsilon", type = float, default = 8/255)
    parser.add_argument("--alpha", type = float, default = 2/255)
    parser.add_argument("--num_iter", type = int, default = 20)
    parser.add_argument("--random", type = bool, default = True)

    args = parser.parse_args()
    
    # model
    if args.network_type == 'resnet':
        network = ResNet18().to(args.devices)
    
    network.load_state_dict(torch.load(f"{args.folder_dir}/best.pth"))
    network.eval()

    fmodel = fb.PyTorchModel(network, bounds=(0,1), device=args.devices)
    
    # data
    _, test_loader = return_dataloader(args)

    # attack
    attack_pgd_linf = LinfPGD(random_start=args.random, steps=args.num_iter, abs_stepsize = args.alpha)
    
    print('Clean Accuracy:')
    evaluation_clean(fmodel, test_loader)
    print('PGD Attack Accuracy:')
    evaluation_adver(fmodel, test_loader, attack_pgd_linf, args.epsilon)
"""
find the best model
"""

from utils.resnet import ResNet18
import foolbox as fb
from utils.dataset import return_dataloader
from foolbox.utils import accuracy
import torch
from foolbox.attacks import LinfPGD
from tqdm import tqdm

def evaluation_clean(fmodel, loader):
    """
    clean accuracy
    """
    ACCURACY = 0.
    for batch_idx, (image, label) in enumerate(loader):
        image, label = image.to(args.devices), label.to(args.devices)
        ACCURACY += accuracy(fmodel, image, label) * len(image)
    
    return ACCURACY/len(loader.dataset)

def evaluation_adver(fmodel, loader, attack, epsilon):
    """
    adversarial accuracy
    """
    ACCURACY = 0. 
    for batch_idx, (image, label) in enumerate(loader):
        image, label = image.to(args.devices), label.to(args.devices)
        raw, clipped, is_adv = attack(fmodel, image, label, epsilons = epsilon)
        ACCURACY += len(image) - is_adv.sum().item()

    return ACCURACY/len(loader.dataset)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    
    # train
    parser.add_argument("--folder_dir", type = str, default = "trained_model/trades/beta_1.0")
    parser.add_argument("--batch_size", type = int, default = 512)
    parser.add_argument("--start_epo", type = int, default = 50)
    parser.add_argument("--end_epo", type = int, default = 110)
    parser.add_argument("--devices", type = str, default = "cuda:1")
    parser.add_argument("--small_set", type = bool, default = False)
    parser.add_argument("--seed", type = int, default = 5)
    
    # attack params
    parser.add_argument("--epsilon", type = float, default = 8/255)
    parser.add_argument("--alpha", type = float, default = 2/255)
    parser.add_argument("--num_iter", type = int, default = 20)
    parser.add_argument("--random", type = bool, default = True)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    # model
    network_trades = ResNet18().to(args.devices)
    
    # data
    _, test_loader = return_dataloader(args)

    # attack
    attack_pgd_linf = LinfPGD(random_start=args.random, steps=args.num_iter, abs_stepsize = args.alpha)

    best_accuracy = 0.

    for i in tqdm(range(args.start_epo, args.end_epo)):
        
        # load new model
        network_trades.load_state_dict(torch.load(f"{args.folder_dir}/epoch_{i}.pth"))
        network_trades.eval()
        fmodel = fb.PyTorchModel(network_trades, bounds=(0,1), device=args.devices)
        
        accuracy_clean = evaluation_clean(fmodel, test_loader)
        accuracy_adver = evaluation_adver(fmodel, test_loader, attack_pgd_linf, args.epsilon)
        total_accuracy = accuracy_clean + accuracy_adver
        
        print(total_accuracy)
        
        if best_accuracy < total_accuracy:
            best_accuracy = total_accuracy
            print('save')
            torch.save(network_trades.state_dict(), f"{args.folder_dir}/best.pth")
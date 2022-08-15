"""
adversarial training
"""

import torch
import os
import sys
sys.path.append('./')
from utils.attack import pgd
import torch.nn.functional as F
from trainer import SOLVER

class AT(SOLVER):
    def __init__(self, args):
        super().__init__(args)
        
        # model saving address
        self.path = f'trained_model/at/beta_{args.beta}'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
    
    def loss(self, image, delta, label):
        
        # clean loss
        loss_clean = F.cross_entropy(self.network(image), label)
        
        # robust loss
        loss_robust = F.cross_entropy(self.network(image + delta), label)
        
        loss_total = (1-self.beta) * loss_clean + self.beta * loss_robust  # beta: (0,1]
        
        return loss_total
    
    def train_epoch(self):
        self.network.train()
        
        for batch_idx, (image, label) in enumerate(self.train_loader):
            image = image.to(self.device)
            label = label.to(self.device)
            
            self.optim.zero_grad()
            
            # generate attack
            delta = pgd(self.args, self.network, image, label)
            loss_total = self.loss(image, delta, label)
            
            loss_total.backward()
            self.optim.step()
    
"""
main
"""

if __name__ == '__main__':
    
    """
    all the settings follow the paper: Theoretically Principled Trade-off between Robustness and Accuracy
    """
    
    import argparse
    parser = argparse.ArgumentParser()
    
    # train
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--epochs", type = int, default = 150)
    parser.add_argument("--devices", type = str, default = "cuda:0")
    parser.add_argument("--lr", type = float, default = 0.1)
    parser.add_argument("--beta", type = float, default = 0.5)
    parser.add_argument("--save", type = bool, default = False)
    parser.add_argument("--seed", type = int, default = 5)
    parser.add_argument("--small_set", type = bool, default = False)
    parser.add_argument("--tensorboard", type = bool, default = True)
    
    # attack params
    parser.add_argument("--epsilon", type = float, default = 0.031)
    parser.add_argument("--alpha", type = float, default = 0.007)
    parser.add_argument("--num_iter", type = int, default = 10)
    
    args = parser.parse_args()
    print(args)
    
    if args.beta <= 0 or args.beta < 1:
        sys.exit('Wrong beta range. beta should be in (0,1]')
    
    torch.manual_seed(args.seed)
    trainer = AT(args)
    
    # record the classification result for each sample each epoch
    for i in range(1,args.epochs+1):
        trainer.adjust_learning_rate(i)
        print('epoch: ', i, 'learning rate: ', round(trainer.optim.param_groups[0]['lr'],5))

        trainer.train_epoch()
        trainer.train_eval(i)
        trainer.test_eval(i)
        torch.save(trainer.network.state_dict(), f'{trainer.path}/epoch_{i}.pth')
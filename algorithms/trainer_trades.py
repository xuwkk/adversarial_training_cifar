"""
implement the TRADES algorithm from paper: Theoretically Principled Trade-off between Robustness and Accuracy
"""

import torch
import os
import sys
sys.path.append('./')
from utils.attack import trades
import torch.nn.functional as F
from trainer import SOLVER

class TRADES(SOLVER):
    def __init__(self, args):
        super().__init__(args)
        
        # model saving address
        self.path = f'trained_model/trades/beta_{args.beta}'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
    
    def loss(self, image, delta, label):
        """
        the trades loss
        """
        criterion_kl = torch.nn.KLDivLoss(reduction='batchmean')
        
        # clean loss
        y_clean = self.network(image)
        loss_clean = F.cross_entropy(y_clean, label)

        # robust loss
        loss_robust = criterion_kl(F.log_softmax(self.network(image+delta), dim=1), F.softmax(self.network(image), dim=1))
        loss_total = loss_clean + self.beta * loss_robust
        return loss_total
    
    def train_epoch(self):
        self.network.train()
        
        for batch_idx, (image, label) in enumerate(self.train_loader):
            image = image.to(self.device)
            label = label.to(self.device)
            
            self.optim.zero_grad()
            
            # generate attack
            delta = trades(self.args, self.network, image)
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
    parser.add_argument("--epochs", type = int, default = 110)
    parser.add_argument("--devices", type = str, default = "cuda:0")
    parser.add_argument("--lr", type = float, default = 0.1)
    parser.add_argument("--beta", type = float, default = 5.0)
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
    torch.manual_seed(args.seed)
    trainer = TRADES(args)
    
    # record the classification result for each sample each epoch
    for i in range(1,args.epochs+1):
        trainer.adjust_learning_rate(i)
        print('epoch: ', i, 'learning rate: ', round(trainer.optim.param_groups[0]['lr'],5))

        trainer.train_epoch()
        trainer.train_eval(i)
        trainer.test_eval(i)
        torch.save(trainer.network.state_dict(), f'{trainer.path}/epoch_{i}.pth')
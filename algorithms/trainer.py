"""
training algorithm class
"""

import torch
import os
import sys
sys.path.append('./')
from utils.resnet import ResNet18
import torch.optim as optim
from utils.dataset import return_dataloader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

class SOLVER:
    def __init__(self, args):
        
        self.args = args
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.device = args.devices
        self.lr = args.lr
        self.save = args.save
        self.beta = args.beta
        self.tensorboard = args.tensorboard
        
        # network instance
        self.network = ResNet18().to(self.device)

        # optimizer
        self.optim = optim.SGD(self.network.parameters(), lr = self.lr, momentum=0.9, weight_decay=2e-4)
        
        # data
        self.train_loader, self.test_loader = return_dataloader(args)
        
        # tensorboard
        if self.tensorboard:
            self.writer = SummaryWriter()
    
    def adjust_learning_rate(self, epoch):
        """decrease the learning rate"""
        lr = self.lr
        if epoch >= 75:
            lr = self.lr * 0.1
        if epoch >= 90:
            lr = self.lr * 0.01
        if epoch >= 100:
            lr = self.lr * 0.001
        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
    
    def train_eval(self, epoch):
        """
        to salve time, we only test the attack during evaluation
        """
        
        self.network.eval()
        train_loss = 0.
        correct = 0.
        
        with torch.no_grad():
            for X, label in self.train_loader:
                X, label = X.to(self.device), label.to(self.device)
                output = self.network(X)
                train_loss += F.cross_entropy(output, label).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(label.view_as(pred)).sum().item()
        train_loss /= len(self.train_loader.dataset)
        print("TRAIN LOSS: ", round(train_loss, 5), "ACCURACY: ", round(correct/len(self.train_loader.dataset), 5))
        
        if self.tensorboard:
            # tensorboard
            self.writer.add_scalars('CLEAN LOSS', {'TRAIN': train_loss/len(self.train_loader.dataset)}, epoch)
            self.writer.add_scalars('CLEAN ACCURACY', {'TRAIN': correct/len(self.train_loader.dataset)}, epoch)
    
    def test_eval(self, epoch):
        
        self.network.eval()
        test_loss = 0.
        correct = 0.
        
        with torch.no_grad():
            for X, label in self.test_loader:
                X, label = X.to(self.device), label.to(self.device)
                output = self.network(X)
                test_loss += F.cross_entropy(output, label).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(label.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        print("TEST LOSS: ", round(test_loss, 5), "ACCURACY: ", round(correct/len(self.test_loader.dataset), 5))
        
        if self.tensorboard:
            # tensorboard
            self.writer.add_scalars('CLEAN LOSS', {'VALID': test_loss/len(self.test_loader.dataset)}, epoch)
            self.writer.add_scalars('CLEAN ACCURACY', {'VALID': correct/len(self.test_loader.dataset)}, epoch)
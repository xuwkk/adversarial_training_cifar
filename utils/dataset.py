from torchvision import transforms
import torchvision
import torch
import torch.utils.data as data_utils

def return_dataloader(args):
    
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    if args.small_set:
        # use a small subset of the data
        indices_train = torch.arange(10000)
        trainset = data_utils.Subset(trainset, indices_train)
        indices_test = torch.arange(1000)
        testset = data_utils.Subset(testset, indices_test)
    
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    
    return train_loader, test_loader
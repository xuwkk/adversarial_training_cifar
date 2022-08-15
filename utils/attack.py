import torch
import copy
import torch.nn.functional as F

def initialize_feature(args, X):
    """
    initialize the attack vector in the feature domain
    """

    # randomly perturbation initialization
    delta = torch.rand_like(X, requires_grad=True) #0-1
    # Rescale into [-epsilon,epsilon]
    delta.data = delta.data * 2 * args.epsilon - args.epsilon
    # Make sure that the perturbed image is in [0,-1]
    delta.data.clamp_(min = 0-X, max = 1-X)

    # # zero perturbation initialization
    # delta = torch.zeros_like(X, requires_grad=True)
    
    return delta

def pgd(args, network, X, label):
    """
    perturbing the original image to maximize the classification error
    """
    
    network = copy.deepcopy(network)
    network.eval()  # disable BN and dropout
    
    epsilon = args.epsilon   # Perturbation strength   
    alpha = args.alpha       # lr
    num_iter = args.num_iter # Iteration number
    
    # Initialization
    delta = initialize_feature(args, X)  # random initialization, same to the original paper
    #delta = 0.001 * torch.randn_like(X).detach()
    delta.requires_grad_()
    
    for t in range(num_iter):
        with torch.enable_grad():
            cross_entropy_loss = F.cross_entropy(network(X.detach() + delta), label)
        
        # maximize the classification error
        cross_entropy_loss.backward()
        delta.data = (delta.detach() + alpha*delta.grad.detach().sign())
        
        # clamp within the [-epsilon,epsilon]
        delta.data.clamp_(min = -epsilon, max = epsilon)
        
        # clamp within [0,1]
        delta.data.clamp_(min = 0-X, max = 1-X)
        
        # zero out the gradient
        delta.grad.zero_()
        
    assert torch.all(torch.abs(delta.detach().view(X.shape[0], -1)) <= epsilon) 
    assert torch.all((X + delta.detach()).view(X.shape[0], -1) <= 1)
    assert torch.all((X + delta.detach()).view(X.shape[0], -1) >= 0)
    
    return delta.detach()

def trades(args, network, X):
    """
    maximize the KL divergence between the clean and adversarial outputs
    """
    
    criterion_kl = torch.nn.KLDivLoss(reduction='batchmean')  # maximize the KLD between the clean and adversarial samples
    
    network = copy.deepcopy(network)
    network.eval()  # disable BN and dropout  
    
    epsilon = args.epsilon   # Perturbation strength   
    alpha = args.alpha       # lr
    num_iter = args.num_iter # Iteration number
    
    # Initialization
    #delta = initialize_feature(args, X)
    
    # use the original paper setting: a small perturbation around the original image
    delta = 0.001 * torch.randn_like(X).detach()
    delta.requires_grad_()
    
    for t in range(num_iter):
        with torch.enable_grad():
            loss = criterion_kl(F.log_softmax(network(X.detach() + delta), dim=1), F.softmax(network(X.detach()), dim=1))
        loss.backward()
        delta.data = (delta.detach() + alpha*delta.grad.detach().sign())
    
        # clamp within the [-epsilon,epsilon]
        delta.data.clamp_(min = -epsilon, max = epsilon)
        
        # clamp within [0,1]
        delta.data.clamp_(min = 0-X, max = 1-X)
        
        delta.grad.zero_()
    
    assert torch.all(torch.abs(delta.detach().view(X.shape[0], -1)) <= epsilon) 
    assert torch.all((X + delta.detach()).view(X.shape[0], -1) <= 1)
    assert torch.all((X + delta.detach()).view(X.shape[0], -1) >= 0)
    
    return delta.detach()

def trades_1(args, network, X):
    """
    maximize the KL divergence between the clean and adversarial outputs
    """
    criterion_kl = torch.nn.KLDivLoss(size_average=False)
    
    network = copy.deepcopy(network)
    network.eval()  # set the network to evaluation mode to disable BN and dropout  
    
    epsilon = args.epsilon   # Perturbation strength   
    alpha = args.alpha       # lr
    num_iter = args.num_iter # Iteration number
    
    # Initialization
    #delta = initialize_feature(args, X)
    # use the original paper setting: a small perturbation around the original image
    delta = 0.001 * torch.randn_like(X).detach()
    delta.requires_grad_()
    
    for t in range(num_iter):
        with torch.enable_grad():
            loss = criterion_kl(F.log_softmax(network(X.detach() + delta), dim=1), F.softmax(network(X.detach()), dim=1))
        loss.backward()
        delta.data = (delta.detach() + alpha*delta.grad.detach().sign())
    
        # clamp within the [-epsilon,epsilon]
        delta.data.clamp_(min = -epsilon, max = epsilon)
        
        # clamp within [0,1]
        delta.data.clamp_(min = 0-X, max = 1-X)
        
        delta.grad.zero_()
    
    assert torch.all(torch.abs(delta.detach().view(X.shape[0], -1)) <= epsilon) 
    assert torch.all((X + delta.detach()).view(X.shape[0], -1) <= 1)
    assert torch.all((X + delta.detach()).view(X.shape[0], -1) >= 0)
    
    return delta.detach()
import constants as CONSTANTS
import torch

def GloVeLoss(output,target):
    
    def weighting_function(x):
        x[x>CONSTANTS.x_max] = CONSTANTS.x_max
        weights = torch.pow ( (x/CONSTANTS.x_max), CONSTANTS.alpha )
        return weights
    
    return torch.mean( torch.pow ( output - torch.log(target), 2 ) * weighting_function(target) )
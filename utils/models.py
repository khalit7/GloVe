import torch
import torch.nn as nn
import constants as CONSTANTS


class GloVeModel(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        
        self.embed1 = nn.Embedding(vocab_size,CONSTANTS.embed_size)
        self.embed2 = nn.Embedding(vocab_size,CONSTANTS.embed_size)
        
        self.b1 = nn.Embedding(vocab_size,1)
        self.b2 = nn.Embedding(vocab_size,1)
        
        
    def forward(self,x):
        '''
        x has the shape batch_size * 2
        '''
        batch_size = x.shape[0]
        
        first_embedding = self.embed1(x[:,0])  # has shape batch_size * CONSTANTS.embed_size
        second_embedding = self.embed2(x[:,1]) # has shape batch_size * CONSTANTS.embed_size
        
        first_biase =  self.b1(x[:,0]).view(batch_size) # has shape batch_size * 1
        second_biase = self.b2(x[:,1]).view(batch_size) # has shape batch_size * 1
        
        dot_product = torch.einsum("ij,ij->i",first_embedding,second_embedding) # has shape batch_size * 1

        output = dot_product + first_biase + second_biase
        
        return output
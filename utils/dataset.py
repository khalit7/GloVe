import torch
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer

from torch.utils.data import Dataset,DataLoader

import numpy as np

import constants as CONSTANTS


def _get_data_itr (split):
    
    data_itr = WikiText2(split=split)
    
    return data_itr

def _get_tokenizer():
    
    return get_tokenizer("basic_english")

def get_vocab(data_itr,tokenizer):
    
    vocab = build_vocab_from_iterator(map(tokenizer,data_itr),min_freq=CONSTANTS.min_freq,specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    
    
    return vocab


def create_co_occurance_matrix(data_itr,vocab,tokenizer):
    
    n = len(vocab)
    co_occurance_mat = np.zeros((n,n))
    for x in data_itr:
        tokens = tokenizer(x)
        for i in range( len(tokens) ) :
            middle_word_idx = vocab[ tokens[i] ]
            left_end  = max(i-CONSTANTS.context_size,0)
            right_end = min(i+CONSTANTS.context_size+1,len(tokens))
            # iterate through context words and add their co-occurance with the middle word by 1
            for t in tokens[left_end:i] + tokens[i+1:right_end]:
                context_word_idx = vocab[ t ]
                co_occurance_mat[middle_word_idx,context_word_idx]+=1
                
                
    assert (co_occurance_mat.T == co_occurance_mat).all(),"Co occurance matrix is not symmetric for some reason :/ check your code"
    
    return co_occurance_mat
    
    
class glove_data(Dataset):
    def __init__(self,vocab,co_occurance_mat):
        
        all_tokens = vocab.get_itos()
        
        X = []
        y = []
        for i in range( len(all_tokens) ):
            for j in range( i,len(all_tokens) ):
                # if the co-occurance is not zero, add it to the dataset
                i_idx = vocab[all_tokens[i]]
                j_idx = vocab[all_tokens[j]]
                if co_occurance_mat[i_idx][j_idx] > 0:
                    X.append( (i_idx,j_idx) )
                    y.append( co_occurance_mat[i_idx][j_idx] )
                    
        assert len(X) == len(y), "length of inputs is not equal to length of outputs"
                
        self.X = torch.tensor(X,dtype=torch.int)
        self.y = torch.tensor(y,dtype=torch.long)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        return self.X[idx],self.y[idx]
    
    
def get_dataloader_and_vocab(split,batch_size,vocab=None):
    
    print("getting data_itr, tokenizer, and vocab ... ",end=" ")
    data_itr = _get_data_itr (split) 
    tokenizer = _get_tokenizer()
    
    if vocab==None:
        vocab = get_vocab(data_itr,tokenizer)
    
    print("Done!")
    
    print("calculating co-occurance matrix ... ", end=" ")
    co_occurance_mat = create_co_occurance_matrix(data_itr,vocab,tokenizer)
    print("Done!")
    
    print("creating dataset ... ", end = " ")
    dataset = glove_data(vocab,co_occurance_mat)
    
    data_loader = DataLoader(dataset,batch_size=batch_size, shuffle=CONSTANTS.shuffle)
    
    print("Done!")
    
    
    return data_loader,vocab
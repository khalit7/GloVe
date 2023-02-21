from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer

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


def create_co_occurance_matrix(data_itr,vocab):
    
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
    
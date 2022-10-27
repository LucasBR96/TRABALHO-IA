#---------------------------------------------------------------------------
# CODE FROM THIS VERY PROJECT
import text_cleaning as txc
import vectorization as vct
vct.init_globals()

#---------------------------------------------------------------------------
# THIRD PARTY LIBRARIES
import pandas as pd
import numpy as np
import torch as tc
from torch.utils.data import Dataset , DataLoader
import diskcache as dkc
import random


#---------------------------------------------------------------------------
# PTYHON STANDARD LIBRARY
import random as rd
import functools as ft
import itertools as itt
from typing import *

#--------------------------------------------------------------------------
# CONSTANTS
df = pd.read_csv("DATA/hate_speech_pt.csv")[ [ "text" , "hatespeech_comb" ] ]

cache_1 = dkc.Cache()
@cache_1.memoize( typed = True )
def get_entry( pos : int ) -> Tuple[ Iterable[ int ] , int ]:

    rec : pd.Series = df.iloc[ pos ]
    
    text : str = rec[ 'text' ]
    clean_text : str = txc.clear_text( text )
    tok_text = txc.tokenize_text( clean_text )
    X = vct.vectorize( tok_text )

    Y = float( rec['hatespeech_comb'] )

    return ( X , Y )

def split_entries( idx_lst ):

    '''
    separa os indices das linhas classificadas como discurso de odio
    '''    

    hate_classification = df[ "hatespeech_comb" ]
    pos_idx , neg_idx = [] , []
    for i in idx_lst:
        hate = hate_classification.iloc[ i ]
        if hate:
            pos_idx.append( i )
        else:
            neg_idx.append( i )

    return pos_idx , neg_idx 

def pre_sample( eval = False ):

    n = len( df )
    if eval:
        return list( range( 0 , n , 5 ) )
    return [ i for i in range( n ) if i%5 != 0 ]

class HateSpeechDataset(Dataset):

    def __init__( self , eval = False ):

        self.eval = eval
        if eval:
            self.idx = pre_sample( True )
        else:
            idx_lst = pre_sample()
            self.pos_idx , self.neg_idx = split_entries( idx_lst )
        
    def __len__( self ):

        if self.eval:
            return len( self.idx )
        return 2*len( self.neg_idx )

    def __getitem__(self, index):

        if self.eval:
            pd_idx = self.idx[ index ]
        else:
            idx_lst = self.pos_idx if index%2 else self.neg_idx
            n = index//2
            m = len( idx_lst )
            pd_idx = idx_lst[ n%m ]

        X , Y = get_entry( pd_idx )
        return tc.from_numpy( X ).float() , Y

if __name__ == "__main__":

    
    dataset = HateSpeechDataset()
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)
    print(next(iter(dataloader)))
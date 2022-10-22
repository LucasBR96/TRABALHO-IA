#---------------------------------------------------------------------------
# CODE FROM THIS VERY PROJECT
import text_cleaning as tc
import vectorization as vct
vct.init_globals()

#---------------------------------------------------------------------------
# THIRD PARTY LIBRARIES
import pandas as pd
import numpy as np
import torch as tc
from torch.utils.data import Dataset
import diskcache as dkc

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
def get_entry( pos : int ) -> Tuple( Iterable[ int ] , int ):

    rec : pd.Series = df.iloc[ pos ]
    
    text : str = rec[ 'text' ]
    clean_text : str = tc.clear_text( text )
    tok_text = tc.tokenize_text( clean_text )
    X = vct.vectorize( tok_text )

    Y = rec['hatespeech_comb'].to_numpy()[0]

    return ( X , Y )
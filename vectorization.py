from typing import *
import numpy as np

word_freq_rank = dict() 
hash_freq_rank = dict()
vector_len : int

def init_globals():

    global word_freq_rank , hash_freq_rank, vector_len

    f = open( 'DATA/dummy.txt' )
    vector_len = int( f.readline() )
    
    word_freq_rank = {}
    words : List[ str ]
    words = f.readline().split()
    for i , word in enumerate( words ):
        word_freq_rank[ word ] = i

    vector_len += int( f.readline() )

    hash_freq_rank = {}
    words = f.readline().split()
    for i , word in enumerate( words ):
        hash_freq_rank[ word ] = i
    
    f.close()

tokenized_text = Tuple[ List[ str ] , List[ str ] ]
def vectorize( tok : tokenized_text ) -> Iterable[ float ]:

    txt_vector = np.zeros( vector_len )
    words , tags = tok

    for word in words:
        if word not in word_freq_rank:
            continue
        txt_vector[ word_freq_rank[ word ] ] = 1.
    
    n : int = len( word_freq_rank )
    for word in tags:
        if word not in hash_freq_rank:
            continue
        txt_vector[ hash_freq_rank[ word ] + n ] = 1.
    
    return txt_vector

if __name__ == "__main__":

    import text_cleaning as tc

    s = "Emma stone devia sentir nojo dela por pegar um PRETO fedido #Oscars"
    tok = tc.tokenize_text( tc.clear_text( s ) )

    init_globals()
    vec = vectorize( tok )
    print( vec )
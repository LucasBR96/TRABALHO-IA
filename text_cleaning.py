import re
from typing import List, Tuple
import unicodedata as und
import pandas as pd
import nltk

EXCESS_WHITE = r"( {2,}|\t+|\n+)"
PONCTUATION  = r"\W&&[^#]"
LINK         = r"(https?:?//\S+|@\w+)"
HASHTAG      = r"(?<!\S)#\S+"

STOP_WORDS = set( nltk.corpus.stopwords.words('portuguese') )

def normalize_str( entry_str : str ) -> str:

    text = entry_str.lower()

    #----------------------------------------------------------------
    # removing diacritics ã , ç -> a , c
    text = und.normalize( 'NFD' , text )
    shaved = ''.join( c for c in text if not und.combining( c ) )
    text   = und.normalize( 'NFC' , shaved )

    #-----------------------------------------------------------------
    #removing unecessary characters
    text = re.sub( EXCESS_WHITE , " " , text )  # excess whitespace
    text = re.sub( PONCTUATION , "" , text )    # ponctuation 
    
    return text

def remove_special_words( entry_str : str ) -> str:
    return re.sub( LINK , "" , entry_str )

def remove_stopwords( entry_str : str ) -> str:
    words : list[ str ] = entry_str.split( sep = " " )
    return " ".join( word for word in words if not( word in STOP_WORDS ) )

def clear_text( entry_str : str ) -> str:
    norm : str
    norm = remove_special_words( entry_str )
    norm = normalize_str( norm )
    return remove_stopwords( norm )

def tokenize_text( entry_str : str ) -> Tuple[ List[str] , List[str] ]:
    
    '''
    splits the clean text into common words and hashtags
    '''

    word : str
    common_words : List[str]
    hash_tags : List[str]
    seq : List[str]

    common_words , hash_tags = [] , []
    for word in entry_str.split():
        seq = common_words
        if word[ 0 ] == "#":
            word = word[ 1: ]
            seq  = hash_tags
        seq.append( word )
    return ( common_words , hash_tags )

if __name__ == "__main__":

    norm = clear_text( "ACTUALMENTE AQUELE RETARDADO QUE DEITOU NUMA BANHEIRA DE NUTELA #JOAQUINRESPONDE https://t.co/JdCLqC5zmY" )
    print( norm )

    print( tokenize_text( norm ) )

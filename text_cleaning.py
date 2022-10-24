import re
from typing import List, Tuple
import unicodedata as und
import pandas as pd
import nltk

EXCESS_WHITE = r"( {2,}|\t+|\n+)"
PONCTUATION  = r"[^a-zA-Z#@ ]"
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

    '''
    Removes usernames and http links
    '''

    return re.sub( LINK , "" , entry_str )

def remove_stopwords( entry_str : str ) -> str:

    '''
    Removes words that either dont add value to the sentence or have a secondary 
    role in the semantics of the sentence.

    ex:

    >>> s = "hey amazon my package never arrived please fix asap"
    >>> remove_stopwords( s )
    'amazon package never arrived fix asap'

    '''

    words : list[ str ] = entry_str.split( sep = " " )
    return " ".join( word for word in words if not( word in STOP_WORDS ) )

def clear_text( entry_str : str ) -> str:
    
    '''
    groups the functions 'remove_special_words', 'normalize_str' and 'remove_stopwords'
    in one method
    '''

    norm : str
    norm = remove_special_words( entry_str )
    norm = normalize_str( norm )
    return remove_stopwords( norm )

def tokenize_text( entry_str : str ) -> Tuple[ List[str] , List[str] ]:
    
    '''
    Separates regular words from hashtags from the cleaned text

    ex:

    >>> s = "As veias abertas da América Latina é um livro de história sombrio, escrito com emoção e de forma lírica. #literatura"
    >>> clean_s = clear_text( s )
    >>> clean_s
    'veias abertas america latina livro historia sombrio escrito emocao forma lirica #literatura'
    >>> tokenize_text( clean_s )
    (['veias', 'abertas', 'america', 'latina', 'livro', 'historia', 'sombrio', 'escrito', 'emocao', 'forma', 'lirica'], ['literatura'])
    '''

    word : str
    common_words : List[str]
    hash_tags : List[str]
    seq : List[str]

    common_words , hash_tags = [] , []
    for word in entry_str.split():
        seq = common_words
        if word[ 0 ] == "#":
            word = word[ 1: ]   #if it is a hashtag the symbol "#" is no longer nescessary
            seq  = hash_tags
        seq.append( word )
    return ( common_words , hash_tags )

if __name__ == "__main__":

    norm = clear_text( "A Emma Stone podia ir vestida de saco de lixo rodeado de sacos do lixo dentro de um camião de lixo, que nunca estaria mal #Oscars" )
    words , tags = tokenize_text( norm )

    print( len( words ) )
    print( *words )
    print( len( tags ) )
    print( *tags )
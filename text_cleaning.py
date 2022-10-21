import re
import unicodedata as und
import pandas as pd
import nltk

EXCESS_WHITE = r"( {2,}|\t+|\n+)"
PONCTUATION  = r"[\?!,\.\":\(\)\“\”\-]"
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

    norm : str = normalize_str( entry_str )
    norm = remove_special_words( norm )
    return remove_stopwords( norm )


norm = clear_text( "A coisa q mais me da odio na vida é mulher machista. Qnd escuto alguma falando merda fico com mt abuso - pq odeio gnt burra e ignorante tb" )
print( norm )
